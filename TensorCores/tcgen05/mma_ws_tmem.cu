#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
#error "tcgen05 requires SM100+"
#endif

// Minimal tcgen05.mma.ws test with tmem descriptors.
// Usage: ./mma_ws_tmem [iters]
// Build-time macro: MMA_KIND (default f16). Example: nvcc -DMMA_KIND=bf16 ...

#ifndef MMA_KIND
#define MMA_KIND f16
#endif

#define STR2(x) #x
#define STR(x) STR2(x)

__device__ __forceinline__ void tmem_alloc(uint32_t& taddr, int nCols) {
    asm volatile("tcgen05.alloc.cta_group.sync.aligned.shared::cta.b32 [%0], %1;" : "=r"(taddr) : "r"(nCols));
}

__device__ __forceinline__ void tmem_dealloc(uint32_t taddr, int nCols) {
    asm volatile("tcgen05.dealloc.cta_group.sync.aligned.b32 %0, %1;" :: "r"(taddr), "r"(nCols));
}

__device__ __forceinline__ void tmem_commit() {
    asm volatile("tcgen05.commit;" ::: "memory");
}

__device__ __forceinline__ void tmem_wait() {
    asm volatile("tcgen05.wait;" ::: "memory");
}

// Simplified descriptors: using tmem addresses directly.
__device__ __forceinline__ uint32_t make_tmem_desc(uint32_t taddr) {
    return taddr;
}

__global__ void mma_kernel(int iters, unsigned long long* out) {
    __shared__ uint32_t taddr_a, taddr_b, taddr_d;
    __shared__ uint32_t desc_a, desc_b, desc_d;

    if (threadIdx.x == 0) {
        tmem_alloc(taddr_a, 4);
        tmem_alloc(taddr_b, 4);
        tmem_alloc(taddr_d, 4);
        desc_a = make_tmem_desc(taddr_a);
        desc_b = make_tmem_desc(taddr_b);
        desc_d = make_tmem_desc(taddr_d);
    }
    __syncthreads();

    // Dummy fill via ld/st would be added if needed.
    tmem_commit();
    tmem_wait();

    unsigned long long start = clock64();
    for (int i = 0; i < iters; ++i) {
        // tcgen05.mma.ws.cta_group::1.kind::<MMA_KIND> [d-tmem], [a-tmem], b-desc, idesc, enable-input-d
        asm volatile("tcgen05.mma.ws.cta_group::1.kind::" STR(MMA_KIND) " [ %0 ], [ %1 ], %2, %3, enable-input-d;"
                     :: "r"(desc_d), "r"(desc_a), "r"(desc_b), "r"(0));
    }
    tmem_commit();
    tmem_wait();
    unsigned long long end = clock64();

    if (threadIdx.x == 0) {
        out[0] = end - start;
        tmem_dealloc(taddr_a, 4);
        tmem_dealloc(taddr_b, 4);
        tmem_dealloc(taddr_d, 4);
    }
}

int main(int argc, char** argv) {
    int iters = 1024;
    if (argc >= 2) iters = atoi(argv[1]);

    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess || prop.major < 10) {
        fprintf(stderr, "tcgen05 requires SM100+ GPU; detected compute capability %d.%d\n", prop.major, prop.minor);
        return 1;
    }

    unsigned long long* d_out;
    cudaMalloc(&d_out, sizeof(unsigned long long));
    cudaMemset(d_out, 0, sizeof(unsigned long long));

    mma_kernel<<<1, 32>>>(iters, d_out);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "mma_ws kernel failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    unsigned long long cycles = 0;
    cudaMemcpy(&cycles, d_out, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    double clock_ghz = prop.clockRate * 1e-6;
    double ns = double(cycles) / clock_ghz;
    double avg_ns = ns / iters;
    printf("tcgen05.mma.ws f16 iters=%d cycles=%llu avg=%.3f ns\\n", iters, (unsigned long long)cycles, avg_ns);

    cudaFree(d_out);
    return 0;
}

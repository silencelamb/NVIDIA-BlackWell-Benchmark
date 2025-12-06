#include <cstdio>
#include <cuda.h>

#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 0
#endif

#if __CUDA_ARCH__ < 1000
#error "tcgen05 requires SM100+"
#endif

// Measure tcgen05.cp and tcgen05.shift throughput/latency.
// Usage: ./tmem_cp_shift [iters]

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

// Simple descriptor in shared memory (mock)
__device__ __forceinline__ uint32_t make_desc(const void* ptr) {
    return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(ptr));
}

__global__ void cp_shift_kernel(const float* gmem, float* gdst, int iters, unsigned long long* out) {
    __shared__ uint32_t taddr_src;
    __shared__ uint32_t taddr_dst;
    __shared__ uint32_t desc_src;
    __shared__ uint32_t desc_dst;

    if (threadIdx.x == 0) {
        tmem_alloc(taddr_src, 4); // allocate a few columns
        tmem_alloc(taddr_dst, 4);
        desc_src = make_desc(gmem);
        desc_dst = make_desc(gdst);
    }
    __syncthreads();

    // Initial copy
    asm volatile("tcgen05.cp.cta_group.128x256b [ %0 ], %1;" :: "r"(taddr_src), "r"(desc_src));
    tmem_commit();
    tmem_wait();

    unsigned long long start = clock64();
    for (int i = 0; i < iters; ++i) {
        // shift destination tmem pointer (down)
        asm volatile("tcgen05.shift.cta_group.down [%0];" :: "r"(taddr_dst));
        // copy from src tmem to dst tmem (tmem->tmem via descriptor)
        asm volatile("tcgen05.cp.cta_group.128x256b [ %0 ], %1;" :: "r"(taddr_dst), "r"(taddr_src));
    }
    tmem_commit();
    tmem_wait();
    unsigned long long end = clock64();

    if (threadIdx.x == 0) {
        out[0] = end - start;
        tmem_dealloc(taddr_src, 4);
        tmem_dealloc(taddr_dst, 4);
    }
}

int main(int argc, char** argv) {
    int iters = 1024;
    if (argc >= 2) iters = atoi(argv[1]);

    size_t bytes = 256 * 1024; // dummy buffer
    float* gmem = nullptr;
    float* gdst = nullptr;
    cudaMalloc(&gmem, bytes);
    cudaMalloc(&gdst, bytes);
    cudaMemset(gmem, 1, bytes);
    cudaMemset(gdst, 0, bytes);

    unsigned long long* d_out;
    cudaMalloc(&d_out, sizeof(unsigned long long));
    cudaMemset(d_out, 0, sizeof(unsigned long long));

    cp_shift_kernel<<<1, 32>>>(gmem, gdst, iters, d_out);
    cudaDeviceSynchronize();

    unsigned long long cycles = 0;
    cudaMemcpy(&cycles, d_out, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    double clock_ghz = prop.clockRate * 1e-6;
    double ns = double(cycles) / clock_ghz;
    double avg_ns = ns / iters;
    printf("tcgen05.cp+shift iters=%d cycles=%llu avg=%.3f ns\\n", iters, (unsigned long long)cycles, avg_ns);

    cudaFree(d_out);
    cudaFree(gmem);
    cudaFree(gdst);
    return 0;
}

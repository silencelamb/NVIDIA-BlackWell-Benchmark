#include <cstdio>
#include <cuda.h>

#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 0
#endif

#if __CUDA_ARCH__ < 1000
#error "tcgen05 requires SM100+"
#endif

// Measure tcgen05.ld / tcgen05.st latency and throughput on tmem.
// Usage: ./tmem_ldst [iters] [shape]
// shape: 0 -> .16x64b, 1 -> .16x128b, 2 -> .16x256b, 3 -> .32x32b, 4 -> .16x32bx2

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

// Load/store helpers for shape selector
template<int SHAPE>
__device__ __forceinline__ void tmem_ld(uint32_t& r, uint32_t taddr) {
    if constexpr (SHAPE == 0) asm volatile("tcgen05.ld.sync.aligned.16x64b.x1.b32 %0, [%1];" : "=r"(r) : "r"(taddr));
    else if constexpr (SHAPE == 1) asm volatile("tcgen05.ld.sync.aligned.16x128b.x1.b32 %0, [%1];" : "=r"(r) : "r"(taddr));
    else if constexpr (SHAPE == 2) asm volatile("tcgen05.ld.sync.aligned.16x256b.x1.b32 %0, [%1];" : "=r"(r) : "r"(taddr));
    else if constexpr (SHAPE == 3) asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 %0, [%1];" : "=r"(r) : "r"(taddr));
    else if constexpr (SHAPE == 4) asm volatile("tcgen05.ld.sync.aligned.16x32bx2.x1.b32 %0, [%1];" : "=r"(r) : "r"(taddr));
}

template<int SHAPE>
__device__ __forceinline__ void tmem_st(uint32_t taddr, uint32_t r) {
    if constexpr (SHAPE == 0) asm volatile("tcgen05.st.sync.aligned.16x64b.x1.b32 [%0], %1;" :: "r"(taddr), "r"(r));
    else if constexpr (SHAPE == 1) asm volatile("tcgen05.st.sync.aligned.16x128b.x1.b32 [%0], %1;" :: "r"(taddr), "r"(r));
    else if constexpr (SHAPE == 2) asm volatile("tcgen05.st.sync.aligned.16x256b.x1.b32 [%0], %1;" :: "r"(taddr), "r"(r));
    else if constexpr (SHAPE == 3) asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], %1;" :: "r"(taddr), "r"(r));
    else if constexpr (SHAPE == 4) asm volatile("tcgen05.st.sync.aligned.16x32bx2.x1.b32 [%0], %1;" :: "r"(taddr), "r"(r));
}

template<int SHAPE>
__global__ void ldst_kernel(int iters, unsigned long long* out) {
    __shared__ uint32_t taddr;
    if (threadIdx.x == 0) {
        tmem_alloc(taddr, 1); // minimal columns; shape-specific payload
    }
    __syncthreads();

    // warmup store
    tmem_st<SHAPE>(taddr, 0xdeadbeef);
    tmem_commit();
    tmem_wait();

    unsigned long long start = clock64();
    uint32_t v = 0;
    #pragma unroll 1
    for (int i = 0; i < iters; ++i) {
        tmem_ld<SHAPE>(v, taddr);
        tmem_st<SHAPE>(taddr, v + 1);
    }
    tmem_commit();
    tmem_wait();
    unsigned long long end = clock64();

    if (threadIdx.x == 0) {
        out[0] = end - start;
        tmem_dealloc(taddr, 1);
    }
}

int main(int argc, char** argv) {
    int iters = 1024;
    int shape = 0;
    if (argc >= 2) iters = atoi(argv[1]);
    if (argc >= 3) shape = atoi(argv[2]);

    unsigned long long* d_out;
    cudaMalloc(&d_out, sizeof(unsigned long long));
    cudaMemset(d_out, 0, sizeof(unsigned long long));

    dim3 block(32);
    dim3 grid(1);

    switch (shape) {
        case 0: ldst_kernel<0><<<grid, block>>>(iters, d_out); break;
        case 1: ldst_kernel<1><<<grid, block>>>(iters, d_out); break;
        case 2: ldst_kernel<2><<<grid, block>>>(iters, d_out); break;
        case 3: ldst_kernel<3><<<grid, block>>>(iters, d_out); break;
        case 4: ldst_kernel<4><<<grid, block>>>(iters, d_out); break;
        default: ldst_kernel<0><<<grid, block>>>(iters, d_out); break;
    }
    cudaDeviceSynchronize();

    unsigned long long cycles = 0;
    cudaMemcpy(&cycles, d_out, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    double clock_ghz = prop.clockRate * 1e-6;
    double ns = double(cycles) / clock_ghz;
    double avg_ns = ns / iters;

    const char* shape_name = (shape==0) ? "16x64b" : (shape==1) ? "16x128b" : (shape==2) ? "16x256b" :
                             (shape==3) ? "32x32b" : "16x32bx2";
    printf("shape=%s iters=%d cycles=%llu avg=%.3f ns\\n", shape_name, iters, (unsigned long long)cycles, avg_ns);

    cudaFree(d_out);
    return 0;
}

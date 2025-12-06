#include <cstdio>
#include <cstdlib>
#include <cuda.h>

// Remote bandwidth benchmark: kernel runs on compute_dev and streams from memory allocated on mem_dev.

__global__ void bw_kernel(const float* __restrict__ src, float* __restrict__ dst, size_t elements) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    float sum = 0.f;
    for (size_t i = idx; i < elements; i += stride) {
        float4 v = reinterpret_cast<const float4*>(src)[i];
        sum += v.x + v.y + v.z + v.w;
        reinterpret_cast<float4*>(dst)[i] = v;
    }
    // prevent compiler from optimizing away
    if (idx == 0) dst[0] = sum;
}

static void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char** argv) {
    int compute_dev = 0;
    int mem_dev = 1;
    size_t bytes = size_t(1) << 28; // 256 MiB default
    int iters = 50;

    if (argc >= 3) {
        compute_dev = std::atoi(argv[1]);
        mem_dev = std::atoi(argv[2]);
    }
    if (argc >= 4) {
        bytes = size_t(atoll(argv[3]));
    }
    if (argc >= 5) {
        iters = std::atoi(argv[4]);
    }

    printf("compute_dev=%d mem_dev=%d bytes=%zu iters=%d\n", compute_dev, mem_dev, bytes, iters);

    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (compute_dev >= device_count || mem_dev >= device_count) {
        fprintf(stderr, "Invalid device id. Found %d devices.\n", device_count);
        return EXIT_FAILURE;
    }

    int access_ab = 0, access_ba = 0;
    check_cuda(cudaDeviceCanAccessPeer(&access_ab, compute_dev, mem_dev), "cudaDeviceCanAccessPeer ab");
    check_cuda(cudaDeviceCanAccessPeer(&access_ba, mem_dev, compute_dev), "cudaDeviceCanAccessPeer ba");
    if (!(access_ab && access_ba)) {
        fprintf(stderr, "Peer access not available between %d and %d\n", compute_dev, mem_dev);
        return EXIT_FAILURE;
    }

    check_cuda(cudaSetDevice(compute_dev), "set compute dev");
    check_cuda(cudaDeviceEnablePeerAccess(mem_dev, 0), "enable peer mem->compute");
    check_cuda(cudaSetDevice(mem_dev), "set mem dev");
    check_cuda(cudaDeviceEnablePeerAccess(compute_dev, 0), "enable peer compute->mem");

    // allocate on mem_dev
    check_cuda(cudaSetDevice(mem_dev), "set mem dev (alloc src)");
    float* d_src = nullptr;
    check_cuda(cudaMalloc(&d_src, bytes), "cudaMalloc src");
    check_cuda(cudaMemset(d_src, 1, bytes), "memset src");

    size_t elements = bytes / sizeof(float4);
    if (elements == 0) {
        fprintf(stderr, "Buffer too small\n");
        return EXIT_FAILURE;
    }

    // allocate destination on compute_dev
    check_cuda(cudaSetDevice(compute_dev), "set compute dev (alloc dst)");
    float* d_dst = nullptr;
    check_cuda(cudaMalloc(&d_dst, bytes), "cudaMalloc dst");
    check_cuda(cudaMemset(d_dst, 0, bytes), "memset dst");

    cudaEvent_t start, stop;
    check_cuda(cudaEventCreate(&start), "event create start");
    check_cuda(cudaEventCreate(&stop), "event create stop");

    dim3 block(256);
    dim3 grid((elements + block.x - 1) / block.x);
    grid.x = grid.x < 80 ? 80 : grid.x; // keep some occupancy

    double total_ms = 0.0;
    for (int i = 0; i < iters; ++i) {
        check_cuda(cudaEventRecord(start), "record start");
        bw_kernel<<<grid, block>>>(reinterpret_cast<const float*>(d_src), d_dst, elements);
        check_cuda(cudaPeekAtLastError(), "kernel launch");
        check_cuda(cudaEventRecord(stop), "record stop");
        check_cuda(cudaEventSynchronize(stop), "sync stop");
        float ms = 0.f;
        check_cuda(cudaEventElapsedTime(&ms, start, stop), "elapsed");
        total_ms += ms;
    }

    double avg_ms = total_ms / iters;
    double gb = double(bytes) * 2e-9; // read + write
    double bw = gb / (avg_ms / 1e3);
    printf("avg_time=%.3f ms avg_bw=%.2f GB/s (bytes=%zu iters=%d)\n", avg_ms, bw, bytes, iters);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_dst);
    cudaSetDevice(mem_dev);
    cudaFree(d_src);
    return 0;
}

#include <cstdio>
#include <cstdlib>
#include <cuda.h>

// Simple pointer-chasing latency on remote memory (compute on compute_dev, memory on mem_dev).
// Assumes peer access is available between the two devices (e.g., B200 dual-die package).

__global__ void latency_kernel(const int* __restrict__ ptr, int iterations, unsigned long long* out_cycles) {
    const int* p = ptr;
    unsigned long long start = clock64();
    #pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
        p = reinterpret_cast<const int*>(*p);
    }
    unsigned long long end = clock64();
    out_cycles[0] = end - start;
    // prevent compiler from optimizing away
    if (p == nullptr) out_cycles[0] = 0;
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
    int elements = 1 << 20; // size of pointer array
    int iterations = 1024;

    if (argc >= 3) {
        compute_dev = std::atoi(argv[1]);
        mem_dev = std::atoi(argv[2]);
    }
    if (argc >= 4) {
        iterations = std::atoi(argv[3]);
    }

    printf("compute_dev=%d mem_dev=%d iterations=%d\n", compute_dev, mem_dev, iterations);

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

    // allocate pointer list on mem_dev
    check_cuda(cudaSetDevice(mem_dev), "set mem dev (alloc)");
    int* d_ptr_list = nullptr;
    check_cuda(cudaMalloc(&d_ptr_list, elements * sizeof(int)), "cudaMalloc d_ptr_list");

    // build pointer-chasing ring on host
    int* h_ptr_list = (int*)malloc(elements * sizeof(int));
    if (!h_ptr_list) {
        fprintf(stderr, "Host malloc failed\n");
        return EXIT_FAILURE;
    }
    for (int i = 0; i < elements - 1; ++i) {
        h_ptr_list[i] = (int)((uintptr_t)(d_ptr_list + i + 1));
    }
    h_ptr_list[elements - 1] = (int)((uintptr_t)(d_ptr_list)); // wrap

    check_cuda(cudaMemcpy(d_ptr_list, h_ptr_list, elements * sizeof(int), cudaMemcpyHostToDevice),
               "cudaMemcpy ptr list");
    free(h_ptr_list);

    // output buffer on compute_dev
    check_cuda(cudaSetDevice(compute_dev), "set compute dev (alloc out)");
    unsigned long long* d_out = nullptr;
    check_cuda(cudaMalloc(&d_out, sizeof(unsigned long long)), "cudaMalloc d_out");

    // launch on compute_dev with pointer on mem_dev
    latency_kernel<<<1, 1>>>(reinterpret_cast<const int*>(d_ptr_list), iterations, d_out);
    check_cuda(cudaPeekAtLastError(), "kernel launch");
    check_cuda(cudaDeviceSynchronize(), "kernel sync");

    unsigned long long cycles = 0;
    check_cuda(cudaMemcpy(&cycles, d_out, sizeof(unsigned long long), cudaMemcpyDeviceToHost), "copy out");

    cudaDeviceProp prop{};
    check_cuda(cudaGetDeviceProperties(&prop, compute_dev), "get props");
    double clock_ghz = prop.clockRate * 1e-6; // clockRate in kHz
    double ns = (double)cycles / clock_ghz; // cycles / (GHz) = ns
    double avg_ns = ns / iterations;
    printf("total_cycles=%llu avg_latency=%.2f ns (iterations=%d)\n", cycles, avg_ns, iterations);

    cudaFree(d_out);
    cudaSetDevice(mem_dev);
    cudaFree(d_ptr_list);
    return 0;
}

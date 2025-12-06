#ifndef COMMON_DEVICE_INFO_H_
#define COMMON_DEVICE_INFO_H_

#include <cuda_runtime.h>
#include <cstdio>

// Fetch device properties and optionally print a concise summary.
inline void GetDeviceInfo(int dev, cudaDeviceProp* prop_out, bool verbose = true) {
    cudaError_t err = cudaGetDeviceProperties(prop_out, dev);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceProperties failed for device %d: %s\n", dev, cudaGetErrorString(err));
        return;
    }
    if (!verbose) return;

    int l2 = 0, smem_per_sm = 0, sm_count = 0, mem_bus_width = 0, mem_clock_khz = 0;
    cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, dev);
    cudaDeviceGetAttribute(&smem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev);
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
    cudaDeviceGetAttribute(&mem_bus_width, cudaDevAttrGlobalMemoryBusWidth, dev);
    cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, dev);

    double mem_clock_mhz = mem_clock_khz / 1000.0;
    double theo_bw_gbps = 2.0 * mem_clock_mhz * mem_bus_width / 8.0 / 1e3; // DDR, convert to GB/s

    printf("Device %d: %s\n", dev, prop_out->name);
    printf("  Compute Capability: sm_%d%d\n", prop_out->major, prop_out->minor);
    printf("  SMs: %d, Clock: %.2f MHz, Warp Size: %d\n",
           sm_count, prop_out->clockRate / 1000.0, prop_out->warpSize);
    printf("  L2: %d KB, Shared/SM: %d KB\n", l2 / 1024, smem_per_sm / 1024);
    printf("  Max Threads/Block: %d, Max Threads/SM: %d\n",
           prop_out->maxThreadsPerBlock, prop_out->maxThreadsPerMultiProcessor);
    printf("  Global Mem: %.2f GB, Bus: %d-bit @ %.2f MHz (theoretical BW: %.1f GB/s)\n",
           prop_out->totalGlobalMem / (1024.0 * 1024.0 * 1024.0),
           mem_bus_width, mem_clock_mhz, theo_bw_gbps);
}

#endif  // COMMON_DEVICE_INFO_H_

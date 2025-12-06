#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

// Pointer-chasing latency benchmark that separates compute device and memory
// device. Uses cudaMemMap to pin the allocation on mem_dev and grant access to
// compute_dev, which mirrors the on-package HSI path on B200.
//
// Usage:
//   ./hsi_latency [compute_dev] [mem_dev] [iterations] [elements] [mode] [cache]
//     mode  : malloc | memmap (default: memmap)
//     cache : cv | cg (default: cv; cv bypasses L2 to expose DRAM/HSI)

struct Options {
    int compute_dev = 0;
    int mem_dev = 0;
    int iterations = 2048;
    size_t elements = 1 << 20; // number of 64-bit nodes
    bool use_memmap = true;
    bool bypass_l2 = true; // use ld.global.cv
};

static void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

static void check_cu(CUresult res, const char* msg) {
    if (res != CUDA_SUCCESS) {
        const char* err_str = nullptr;
        cuGetErrorString(res, &err_str);
        fprintf(stderr, "%s: %s\n", msg, err_str ? err_str : "unknown");
        exit(EXIT_FAILURE);
    }
}

static Options parse(int argc, char** argv) {
    Options opt;
    if (argc >= 2) opt.compute_dev = std::atoi(argv[1]);
    if (argc >= 3) opt.mem_dev = std::atoi(argv[2]);
    else opt.mem_dev = opt.compute_dev;
    if (argc >= 4) opt.iterations = std::atoi(argv[3]);
    if (argc >= 5) opt.elements = static_cast<size_t>(atoll(argv[4]));
    if (argc >= 6) {
        opt.use_memmap = (std::string(argv[5]) != "malloc");
    }
    if (argc >= 7) {
        opt.bypass_l2 = (std::string(argv[6]) != "cg");
    }
    return opt;
}

struct MappedAllocation {
    std::uintptr_t* ptr = nullptr;
    size_t bytes = 0;           // bytes actively used by the pointer list
    size_t reserved_bytes = 0;  // bytes mapped (may be rounded to granularity)
    bool mapped = false;
};

static MappedAllocation allocate_ptr_list(const Options& opt) {
    MappedAllocation alloc{};
    alloc.bytes = opt.elements * sizeof(std::uintptr_t);
    if (opt.use_memmap) {
        CUdeviceptr addr = 0;
        check_cuda(cudaSetDevice(opt.compute_dev), "set compute device");

        CUmemAllocationProp prop{};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = opt.mem_dev;
        prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;

        size_t granularity = 0;
        check_cu(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM), "granularity");
        alloc.reserved_bytes = ((alloc.bytes + granularity - 1) / granularity) * granularity;

        check_cu(cuMemAddressReserve(&addr, alloc.reserved_bytes, 0, 0, 0), "reserve VA");

        CUmemGenericAllocationHandle handle;
        check_cu(cuMemCreate(&handle, alloc.reserved_bytes, &prop, 0), "mem create");
        check_cu(cuMemMap(addr, alloc.reserved_bytes, 0, handle, 0), "mem map");
        check_cu(cuMemRelease(handle), "mem release");

        CUmemAccessDesc access[2]{};
        access[0].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access[0].location.id = opt.mem_dev;
        access[0].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        access[1].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access[1].location.id = opt.compute_dev;
        access[1].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        check_cu(cuMemSetAccess(addr, alloc.reserved_bytes, access, 2), "set access");

        alloc.ptr = reinterpret_cast<std::uintptr_t*>(addr);
        alloc.mapped = true;
    } else {
        alloc.reserved_bytes = alloc.bytes;
        check_cuda(cudaSetDevice(opt.mem_dev), "set mem device");
        check_cuda(cudaMalloc(&alloc.ptr, alloc.bytes), "cudaMalloc ptr list");
    }
    return alloc;
}

static void free_allocation(MappedAllocation& alloc) {
    if (alloc.mapped) {
        check_cu(cuMemUnmap(reinterpret_cast<CUdeviceptr>(alloc.ptr), alloc.reserved_bytes), "mem unmap");
        check_cu(cuMemAddressFree(reinterpret_cast<CUdeviceptr>(alloc.ptr), alloc.reserved_bytes), "free VA");
    } else if (alloc.ptr) {
        check_cuda(cudaFree(alloc.ptr), "free ptr list");
    }
    alloc.ptr = nullptr;
    alloc.bytes = 0;
    alloc.reserved_bytes = 0;
    alloc.mapped = false;
}

template<bool BYPASS_L2>
__global__ void latency_kernel(const std::uintptr_t* __restrict__ ptr,
                               int iterations,
                               unsigned long long* out_cycles) {
    const std::uintptr_t* p = ptr;
    unsigned long long start = clock64();
    #pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
        std::uintptr_t tmp;
        if constexpr (BYPASS_L2) {
            asm volatile("ld.global.cv.u64 %0, [%1];" : "=l"(tmp) : "l"(p));
        } else {
            asm volatile("ld.global.cg.u64 %0, [%1];" : "=l"(tmp) : "l"(p));
        }
        p = reinterpret_cast<const std::uintptr_t*>(tmp);
    }
    unsigned long long end = clock64();
    out_cycles[0] = end - start;
    if (p == nullptr) out_cycles[0] = 0;
}

int main(int argc, char** argv) {
    Options opt = parse(argc, argv);
    printf("compute_dev=%d mem_dev=%d iterations=%d elements=%zu mode=%s cache=%s\n",
           opt.compute_dev, opt.mem_dev, opt.iterations, opt.elements,
           opt.use_memmap ? "memmap" : "malloc", opt.bypass_l2 ? "cv" : "cg");

    check_cu(cuInit(0), "cuInit");

    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count), "device count");
    if (opt.compute_dev >= device_count || opt.mem_dev >= device_count) {
        fprintf(stderr, "Invalid device id. Found %d devices.\n", device_count);
        return EXIT_FAILURE;
    }

    if (opt.compute_dev != opt.mem_dev) {
        int access_ab = 0, access_ba = 0;
        check_cuda(cudaDeviceCanAccessPeer(&access_ab, opt.compute_dev, opt.mem_dev), "peer ab");
        check_cuda(cudaDeviceCanAccessPeer(&access_ba, opt.mem_dev, opt.compute_dev), "peer ba");
        if (!(access_ab && access_ba)) {
            fprintf(stderr, "Peer access not available between %d and %d\n", opt.compute_dev, opt.mem_dev);
            return EXIT_FAILURE;
        }
        check_cuda(cudaSetDevice(opt.compute_dev), "set compute dev");
        cudaError_t s = cudaDeviceEnablePeerAccess(opt.mem_dev, 0);
        if (s != cudaSuccess && s != cudaErrorPeerAccessAlreadyEnabled) check_cuda(s, "enable peer mem->compute");
        check_cuda(cudaSetDevice(opt.mem_dev), "set mem dev");
        s = cudaDeviceEnablePeerAccess(opt.compute_dev, 0);
        if (s != cudaSuccess && s != cudaErrorPeerAccessAlreadyEnabled) check_cuda(s, "enable peer compute->mem");
    }

    MappedAllocation alloc = allocate_ptr_list(opt);
    std::vector<std::uintptr_t> host(opt.elements);
    for (size_t i = 0; i < opt.elements; ++i) {
        size_t next = (i + 32) % opt.elements; // 32 * 8B = 256B stride to avoid cache residency
        host[i] = reinterpret_cast<std::uintptr_t>(alloc.ptr + next);
    }
    check_cuda(cudaSetDevice(opt.compute_dev), "set compute dev (init copy)");
    check_cuda(cudaMemcpy(alloc.ptr, host.data(), alloc.bytes, cudaMemcpyHostToDevice), "copy ptr list");

    unsigned long long* d_out = nullptr;
    check_cuda(cudaMalloc(&d_out, sizeof(unsigned long long)), "cudaMalloc out");

    if (opt.bypass_l2) {
        latency_kernel<true><<<1, 1>>>(alloc.ptr, opt.iterations, d_out);
    } else {
        latency_kernel<false><<<1, 1>>>(alloc.ptr, opt.iterations, d_out);
    }
    check_cuda(cudaPeekAtLastError(), "launch");
    check_cuda(cudaDeviceSynchronize(), "sync");

    unsigned long long cycles = 0;
    check_cuda(cudaMemcpy(&cycles, d_out, sizeof(unsigned long long), cudaMemcpyDeviceToHost), "copy out");

    cudaDeviceProp prop{};
    check_cuda(cudaGetDeviceProperties(&prop, opt.compute_dev), "device props");
    double clock_ghz = prop.clockRate * 1e-6;
    double ns = double(cycles) / clock_ghz;
    double avg_ns = ns / opt.iterations;
    printf("total_cycles=%llu avg_latency=%.2f ns (iterations=%d)\n",
           (unsigned long long)cycles, avg_ns, opt.iterations);

    cudaFree(d_out);
    free_allocation(alloc);
    return 0;
}

#include "Device.h"
#include <stdexcept>
#include <iostream>
#include <cstring>

#if NNK_WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace nnk {

namespace {

/**
 * @brief CPU Backend Implementation
 * Provides CPU execution with TT-Metal simulation patterns
 */
class CPUBackend : public DeviceBackend {
public:
    CPUBackend(int device_id = 0) : device_id_(device_id) {}

    DeviceInfo get_device_info() const override {
        DeviceInfo info(DeviceType::CPU, device_id_, "CPU TT-Metal Simulator");
        info.total_memory = 8ULL * 1024 * 1024 * 1024; // 8GB simulated
        info.free_memory = info.total_memory;
        info.compute_capability_major = 1;
        info.compute_capability_minor = 0;
        info.capabilities = {DeviceCapability::COMPUTE, DeviceCapability::UNIFIED_MEMORY};
        return info;
    }

    void* allocate(size_t size) override {
        void* ptr = std::aligned_alloc(32, size); // 32-byte aligned for SIMD
        if (!ptr) {
            throw std::bad_alloc();
        }
        return ptr;
    }

    void deallocate(void* ptr) override {
        if (ptr) {
            std::free(ptr);
        }
    }

    bool copy_to_device(void* dst, const void* src, size_t size) override {
        // On CPU, device memory is host memory
        std::memcpy(dst, src, size);
        return true;
    }

    bool copy_to_host(void* dst, const void* src, size_t size) override {
        // On CPU, device memory is host memory
        std::memcpy(dst, src, size);
        return true;
    }

    void synchronize() override {
        // No-op for CPU backend - all operations are synchronous
    }

    bool has_capability(DeviceCapability cap) const override {
        switch (cap) {
            case DeviceCapability::COMPUTE:
            case DeviceCapability::UNIFIED_MEMORY:
                return true;
            case DeviceCapability::PEER_ACCESS:
            case DeviceCapability::TENSORCORES:
            case DeviceCapability::SPARSE_OPS:
                return false;
        }
        return false;
    }

private:
    int device_id_;
};

#if NNK_WITH_CUDA
/**
 * @brief CUDA Backend Implementation
 * Provides GPU execution with CUDA acceleration
 */
class CUDABackend : public DeviceBackend {
public:
    explicit CUDABackend(int device_id = 0) : device_id_(device_id) {
        cudaError_t err = cudaSetDevice(device_id_);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to set CUDA device: " +
                                    std::string(cudaGetErrorString(err)));
        }
    }

    DeviceInfo get_device_info() const override {
        cudaDeviceProp prop;
        cudaError_t err = cudaGetDeviceProperties(&prop, device_id_);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to get CUDA device properties");
        }

        DeviceInfo info(DeviceType::CUDA, device_id_, prop.name);
        info.total_memory = prop.totalGlobalMem;

        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        info.free_memory = free_mem;

        info.compute_capability_major = prop.major;
        info.compute_capability_minor = prop.minor;

        // Set capabilities based on compute capability
        info.capabilities = {DeviceCapability::COMPUTE};
        if (prop.unifiedAddressing) {
            info.capabilities.push_back(DeviceCapability::UNIFIED_MEMORY);
        }
        if (prop.canMapHostMemory) {
            info.capabilities.push_back(DeviceCapability::PEER_ACCESS);
        }
        if (prop.major >= 7) { // Tensor cores available from Volta onwards
            info.capabilities.push_back(DeviceCapability::TENSORCORES);
        }

        return info;
    }

    void* allocate(size_t size) override {
        void* ptr;
        cudaError_t err = cudaMalloc(&ptr, size);
        if (err != cudaSuccess) {
            return nullptr;
        }
        return ptr;
    }

    void deallocate(void* ptr) override {
        if (ptr) {
            cudaFree(ptr);
        }
    }

    bool copy_to_device(void* dst, const void* src, size_t size) override {
        cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
        return err == cudaSuccess;
    }

    bool copy_to_host(void* dst, const void* src, size_t size) override {
        cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
        return err == cudaSuccess;
    }

    void synchronize() override {
        cudaDeviceSynchronize();
    }

    bool has_capability(DeviceCapability cap) const override {
        auto info = get_device_info();
        for (auto& capability : info.capabilities) {
            if (capability == cap) return true;
        }
        return false;
    }

private:
    int device_id_;
};
#endif // NNK_WITH_CUDA

} // anonymous namespace

// DeviceManager implementation
DeviceManager& DeviceManager::instance() {
    static DeviceManager instance;
    return instance;
}

std::vector<DeviceInfo> DeviceManager::discover_devices() {
    initialize();

    std::vector<DeviceInfo> devices;

    // Always have CPU available
    devices.emplace_back(DeviceType::CPU, 0, "CPU TT-Metal Simulator");

#if NNK_WITH_CUDA
    int cuda_count = 0;
    cudaError_t err = cudaGetDeviceCount(&cuda_count);
    if (err == cudaSuccess) {
        for (int i = 0; i < cuda_count; ++i) {
            cudaDeviceProp prop;
            if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
                devices.emplace_back(DeviceType::CUDA, i, prop.name);
            }
        }
    }
#endif

    return devices;
}

std::unique_ptr<DeviceBackend> DeviceManager::create_backend(DeviceType type, int device_id) {
    initialize();

    switch (type) {
        case DeviceType::CPU:
            return std::make_unique<CPUBackend>(device_id);

        case DeviceType::CUDA:
#if NNK_WITH_CUDA
            if (is_device_available(DeviceType::CUDA)) {
                try {
                    return std::make_unique<CUDABackend>(device_id);
                } catch (const std::exception&) {
                    return nullptr;
                }
            }
#endif
            return nullptr;

        case DeviceType::AUTO:
            // Try CUDA first, fallback to CPU
            if (is_device_available(DeviceType::CUDA)) {
                auto cuda_backend = create_backend(DeviceType::CUDA, device_id);
                if (cuda_backend) return cuda_backend;
            }
            return create_backend(DeviceType::CPU, device_id);
    }

    return nullptr;
}

DeviceType DeviceManager::get_default_device_type() {
    initialize();

    if (is_device_available(DeviceType::CUDA)) {
        return DeviceType::CUDA;
    }
    return DeviceType::CPU;
}

bool DeviceManager::is_device_available(DeviceType type) const {
    return get_device_count(type) > 0;
}

int DeviceManager::get_device_count(DeviceType type) const {
    initialize();

    auto it = device_counts_.find(type);
    if (it != device_counts_.end()) {
        return it->second;
    }

    int count = 0;
    switch (type) {
        case DeviceType::CPU:
            count = 1; // Always have CPU
            break;

        case DeviceType::CUDA:
#if NNK_WITH_CUDA
            cudaError_t err = cudaGetDeviceCount(&count);
            if (err != cudaSuccess) {
                count = 0;
            }
#else
            count = 0;
#endif
            break;

        case DeviceType::AUTO:
            // AUTO is not a physical device type
            count = 0;
            break;
    }

    device_counts_[type] = count;
    return count;
}

void DeviceManager::initialize() const {
    if (initialized_) return;

    // Initialize CUDA runtime if available
#if NNK_WITH_CUDA
    int cuda_count = 0;
    cudaError_t err = cudaGetDeviceCount(&cuda_count);
    if (err == cudaSuccess && cuda_count > 0) {
        // Initialize CUDA runtime
        cudaFree(0);
    }
#endif

    initialized_ = true;
}

// Device implementation
Device::Device(DeviceType type, int device_id) {
    backend_ = DeviceManager::instance().create_backend(type, device_id);
    if (!backend_) {
        throw std::runtime_error("Failed to create device backend for type: " + to_string(type));
    }
}

Device::Device(Device&& other) noexcept : backend_(std::move(other.backend_)) {}

Device& Device::operator=(Device&& other) noexcept {
    if (this != &other) {
        backend_ = std::move(other.backend_);
    }
    return *this;
}

// Utility functions
std::string to_string(DeviceType type) {
    switch (type) {
        case DeviceType::CPU: return "CPU";
        case DeviceType::CUDA: return "CUDA";
        case DeviceType::AUTO: return "AUTO";
    }
    return "UNKNOWN";
}

std::string to_string(DeviceCapability cap) {
    switch (cap) {
        case DeviceCapability::COMPUTE: return "COMPUTE";
        case DeviceCapability::UNIFIED_MEMORY: return "UNIFIED_MEMORY";
        case DeviceCapability::PEER_ACCESS: return "PEER_ACCESS";
        case DeviceCapability::TENSORCORES: return "TENSORCORES";
        case DeviceCapability::SPARSE_OPS: return "SPARSE_OPS";
    }
    return "UNKNOWN";
}

} // namespace nnk
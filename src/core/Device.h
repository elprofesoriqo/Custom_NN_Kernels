#pragma once

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

namespace nnk {

/**
 * @brief Enumeration of supported device types
 */
enum class DeviceType {
    CPU = 0,    ///< CPU execution with TT-Metal simulation
    CUDA = 1,   ///< NVIDIA GPU execution with CUDA
    AUTO = 2    ///< Automatic device selection based on availability
};

/**
 * @brief Device capability flags
 */
enum class DeviceCapability {
    COMPUTE,        ///< Basic compute operations
    UNIFIED_MEMORY, ///< Unified memory address space
    PEER_ACCESS,    ///< Direct device-to-device memory access
    TENSORCORES,    ///< Tensor core acceleration units
    SPARSE_OPS      ///< Sparse matrix operations support
};

/**
 * @brief Device information and properties
 */
struct DeviceInfo {
    DeviceType type;
    int device_id;
    std::string name;
    size_t total_memory;
    size_t free_memory;
    int compute_capability_major;
    int compute_capability_minor;
    std::vector<DeviceCapability> capabilities;

    DeviceInfo() = default;
    DeviceInfo(DeviceType t, int id, const std::string& n)
        : type(t), device_id(id), name(n), total_memory(0), free_memory(0),
          compute_capability_major(0), compute_capability_minor(0) {}
};

/**
 * @brief Abstract base class for device-specific backends
 *
 * This class defines the interface for all backend implementations,
 * providing a unified API for memory management and kernel execution.
 */
class DeviceBackend {
public:
    virtual ~DeviceBackend() = default;

    /**
     * @brief Get device information
     * @return DeviceInfo structure containing device properties
     */
    virtual DeviceInfo get_device_info() const = 0;

    /**
     * @brief Allocate device memory
     * @param size Size in bytes to allocate
     * @return Pointer to allocated memory, nullptr on failure
     */
    virtual void* allocate(size_t size) = 0;

    /**
     * @brief Free device memory
     * @param ptr Pointer to memory to be freed
     */
    virtual void deallocate(void* ptr) = 0;

    /**
     * @brief Copy memory from host to device
     * @param dst Device destination pointer
     * @param src Host source pointer
     * @param size Number of bytes to copy
     * @return True on success, false on failure
     */
    virtual bool copy_to_device(void* dst, const void* src, size_t size) = 0;

    /**
     * @brief Copy memory from device to host
     * @param dst Host destination pointer
     * @param src Device source pointer
     * @param size Number of bytes to copy
     * @return True on success, false on failure
     */
    virtual bool copy_to_host(void* dst, const void* src, size_t size) = 0;

    /**
     * @brief Synchronize device execution
     * Blocks until all device operations are complete
     */
    virtual void synchronize() = 0;

    /**
     * @brief Check if device has specific capability
     * @param cap Capability to check
     * @return True if supported, false otherwise
     */
    virtual bool has_capability(DeviceCapability cap) const = 0;
};

/**
 * @brief Device management and backend factory
 *
 * Singleton class responsible for device discovery, backend creation,
 * and resource management across different hardware platforms.
 */
class DeviceManager {
public:
    /**
     * @brief Get singleton instance of DeviceManager
     * @return Reference to the global device manager
     */
    static DeviceManager& instance();

    /**
     * @brief Discover and enumerate available devices
     * @return Vector of available device information
     */
    std::vector<DeviceInfo> discover_devices();

    /**
     * @brief Create backend for specific device
     * @param type Device type to create backend for
     * @param device_id Specific device ID (default: 0)
     * @return Unique pointer to device backend, nullptr on failure
     */
    std::unique_ptr<DeviceBackend> create_backend(DeviceType type, int device_id = 0);

    /**
     * @brief Get default device type based on availability
     * @return Automatically selected device type
     */
    DeviceType get_default_device_type();

    /**
     * @brief Check if device type is available on system
     * @param type Device type to check
     * @return True if available, false otherwise
     */
    bool is_device_available(DeviceType type) const;

    /**
     * @brief Get number of devices of specific type
     * @param type Device type to count
     * @return Number of available devices
     */
    int get_device_count(DeviceType type) const;

private:
    DeviceManager() = default;
    ~DeviceManager() = default;
    DeviceManager(const DeviceManager&) = delete;
    DeviceManager& operator=(const DeviceManager&) = delete;

    mutable std::unordered_map<DeviceType, int> device_counts_;
    mutable bool initialized_ = false;

    void initialize() const;
};

/**
 * @brief Device handle for high-level operations
 *
 * RAII wrapper around device backends providing automatic resource
 * management and convenient operations for tensor computations.
 */
class Device {
public:
    /**
     * @brief Create device with automatic type selection
     */
    Device() : Device(DeviceType::AUTO) {}

    /**
     * @brief Create device with specific type
     * @param type Desired device type
     * @param device_id Specific device ID (default: 0)
     */
    explicit Device(DeviceType type, int device_id = 0);

    /**
     * @brief Move constructor
     */
    Device(Device&& other) noexcept;

    /**
     * @brief Move assignment
     */
    Device& operator=(Device&& other) noexcept;

    /**
     * @brief Destructor - automatically cleans up resources
     */
    ~Device() = default;

    // Disable copy operations for RAII safety
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    /**
     * @brief Get device information
     * @return Device information structure
     */
    DeviceInfo info() const { return backend_->get_device_info(); }

    /**
     * @brief Get device type
     * @return Current device type
     */
    DeviceType type() const { return backend_->get_device_info().type; }

    /**
     * @brief Check if device is valid and operational
     * @return True if device is ready for operations
     */
    bool is_valid() const { return backend_ != nullptr; }

    /**
     * @brief Get backend pointer for advanced operations
     * @return Raw pointer to device backend (non-owning)
     * @warning Use carefully - prefer high-level device operations
     */
    DeviceBackend* backend() { return backend_.get(); }
    const DeviceBackend* backend() const { return backend_.get(); }

private:
    std::unique_ptr<DeviceBackend> backend_;
};

/**
 * @brief Convert device type to string representation
 * @param type Device type to convert
 * @return String representation of device type
 */
std::string to_string(DeviceType type);

/**
 * @brief Convert device capability to string representation
 * @param cap Device capability to convert
 * @return String representation of capability
 */
std::string to_string(DeviceCapability cap);

} // namespace nnk



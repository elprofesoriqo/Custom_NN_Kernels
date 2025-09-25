/**
 * @file Tensor.h
 * @brief High-performance tensor computation library for neural network kernels
 *
 * This file contains the core Tensor class implementation providing efficient
 * multi-dimensional array operations with support for both CPU and GPU backends.
 * The design emphasizes RAII principles, type safety, and cross-device compatibility.
 *
 * @author nn_kernels_tt Development Team
 * @version 1.0
 * @date 2025
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "core/Device.h"

namespace nnk {

/**
 * @brief High-performance multi-dimensional tensor class with RAII semantics
 *
 * The Tensor class provides a unified interface for multi-dimensional array operations
 * across heterogeneous computing platforms (CPU/GPU). It implements strict RAII semantics
 * for automatic memory management and supports efficient cross-device data movement.
 *
 * Key features:
 * - RAII-compliant memory management with automatic cleanup
 * - Move semantics for efficient tensor transfers
 * - Cross-device operations (CPU ↔ GPU)
 * - Exception-safe design with strong type safety
 * - Template-based operations for compile-time optimization
 *
 * @par Thread Safety
 * Tensor operations are not thread-safe. External synchronization is required
 * for concurrent access from multiple threads.
 *
 * @par Memory Layout
 * Tensors use row-major (C-style) memory layout. Multi-dimensional indexing
 * follows standard mathematical conventions.
 *
 * @par Example Usage
 * @code
 * // Create a 3x4 tensor filled with zeros
 * auto tensor = nnk::Tensor::zeros({3, 4});
 *
 * // Create tensor from existing data
 * std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
 * auto tensor_from_data = nnk::Tensor::from_vector({2, 2}, data);
 *
 * // Move tensor to GPU (if CUDA support enabled)
 * auto gpu_tensor = tensor.to_device(DeviceType::CUDA);
 * @endcode
 */
class Tensor {
public:
    /**
     * @brief Default constructor creates an empty tensor
     *
     * Constructs an empty tensor with zero dimensions and no allocated memory.
     * The tensor is placed on CPU by default.
     *
     * @post numel() == 0, shape().empty() == true, device() == DeviceType::CPU
     */
    Tensor();

    /**
     * @brief Destructor with automatic resource cleanup
     *
     * Automatically releases all allocated memory including GPU memory if applicable.
     * Follows RAII principles ensuring no memory leaks.
     */
    ~Tensor();

    /**
     * @brief Copy constructor performing deep copy
     *
     * Creates a new tensor by copying data from another tensor. Handles both
     * CPU and GPU tensors appropriately, including cross-device copying.
     *
     * @param other The source tensor to copy from
     * @throw std::runtime_error If memory allocation fails
     */
    Tensor(const Tensor& other);

    /**
     * @brief Move constructor with efficient resource transfer
     *
     * Transfers ownership of resources from another tensor without copying data.
     * The source tensor is left in a valid but unspecified state.
     *
     * @param other The source tensor to move from (will be emptied)
     * @noexcept This operation never throws exceptions
     */
    Tensor(Tensor&& other) noexcept;

    /**
     * @brief Copy assignment operator
     *
     * Assigns the contents of another tensor to this tensor, performing
     * deep copy of all data and metadata.
     *
     * @param other The source tensor to copy from
     * @return Reference to this tensor after assignment
     * @throw std::runtime_error If memory allocation fails
     */
    Tensor& operator=(const Tensor& other);

    /**
     * @brief Move assignment operator
     *
     * Transfers ownership of resources from another tensor efficiently.
     * Existing resources are properly released before transfer.
     *
     * @param other The source tensor to move from (will be emptied)
     * @return Reference to this tensor after assignment
     * @noexcept This operation never throws exceptions
     */
    Tensor& operator=(Tensor&& other) noexcept;

    /**
     * @brief Factory method to create zero-initialized tensor
     *
     * Creates a new tensor with the specified shape filled with zeros.
     * Memory is allocated on the requested device.
     *
     * @param shape Vector specifying tensor dimensions (e.g., {3, 4} for 3x4 matrix)
     * @param device Target device for tensor allocation (CPU or CUDA)
     * @return New tensor filled with zeros
     * @throw std::invalid_argument If shape contains non-positive dimensions
     * @throw std::runtime_error If memory allocation fails
     *
     * @par Example
     * @code
     * auto tensor = Tensor::zeros({2, 3, 4});  // 2x3x4 tensor of zeros
     * @endcode
     */
    static Tensor zeros(const std::vector<int64_t>& shape, DeviceType device = DeviceType::CPU);

    /**
     * @brief Factory method to create tensor from existing data
     *
     * Creates a new tensor with specified shape initialized from vector data.
     * Data is copied to the target device.
     *
     * @param shape Vector specifying tensor dimensions
     * @param data Source data vector (must match total tensor size)
     * @param device Target device for tensor allocation
     * @return New tensor containing copied data
     * @throw std::invalid_argument If data size doesn't match computed tensor size
     * @throw std::runtime_error If memory allocation fails
     *
     * @par Example
     * @code
     * std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
     * auto tensor = Tensor::from_vector({2, 2}, data);
     * @endcode
     */
    static Tensor from_vector(const std::vector<int64_t>& shape, const std::vector<float>& data, DeviceType device = DeviceType::CPU);

    /**
     * @brief Get total number of elements in tensor
     *
     * @return Total element count (product of all dimensions)
     * @noexcept This operation never throws exceptions
     */
    size_t numel() const noexcept { return num_elements_; }

    /**
     * @brief Get tensor shape/dimensions
     *
     * @return Const reference to vector containing tensor dimensions
     * @noexcept This operation never throws exceptions
     */
    const std::vector<int64_t>& shape() const noexcept { return shape_; }

    /**
     * @brief Get tensor device type
     *
     * @return Device type where tensor data is currently stored
     * @noexcept This operation never throws exceptions
     */
    DeviceType device() const noexcept { return device_type_; }

    /**
     * @brief Get mutable pointer to CPU tensor data
     *
     * Provides direct access to tensor data for CPU computations.
     * Tensor must be located on CPU device.
     *
     * @return Pointer to first element of tensor data
     * @throw std::runtime_error If tensor is not on CPU
     *
     * @warning Direct data access bypasses bounds checking.
     * @warning Modifying data through this pointer may invalidate GPU copies.
     */
    float* data();

    /**
     * @brief Get immutable pointer to CPU tensor data
     *
     * Provides read-only access to tensor data for CPU computations.
     * Tensor must be located on CPU device.
     *
     * @return Const pointer to first element of tensor data
     * @throw std::runtime_error If tensor is not on CPU
     */
    const float* data() const;

    /**
     * @brief Copy tensor to target device
     *
     * Creates a new tensor on the specified device containing a copy of this
     * tensor's data. Original tensor remains unchanged.
     *
     * @param target Target device type for the new tensor
     * @return New tensor on target device with copied data
     * @throw std::runtime_error If device transfer fails
     * @throw std::runtime_error If target device is not available
     *
     * @par Example
     * @code
     * auto cpu_tensor = Tensor::zeros({100, 100});
     * auto gpu_tensor = cpu_tensor.to_device(DeviceType::CUDA);
     * @endcode
     */
    Tensor to_device(DeviceType target) const;

    /**
     * @brief Copy tensor to CPU (convenience method)
     *
     * Equivalent to to_device(DeviceType::CPU). Useful for bringing
     * GPU tensors back to CPU for inspection or CPU-only operations.
     *
     * @return New CPU tensor with copied data
     * @throw std::runtime_error If device transfer fails
     */
    Tensor to_host() const { return to_device(DeviceType::CPU); }

private:
    /**
     * @brief Compute total number of elements from shape vector
     * @param dims Shape dimensions vector
     * @return Product of all dimensions
     * @throw std::overflow_error If computed size exceeds size_t capacity
     */
    static size_t compute_numel(const std::vector<int64_t>& dims);

    /**
     * @brief Validate tensor shape for correctness
     * @param dims Shape dimensions vector to validate
     * @throw std::invalid_argument If any dimension is non-positive
     */
    static void validate_shape(const std::vector<int64_t>& dims);

    // Core tensor metadata
    std::vector<int64_t> shape_{};           ///< Tensor dimensions
    size_t num_elements_ = 0;               ///< Total element count (cached)
    DeviceType device_type_ = DeviceType::CPU; ///< Current device location

    // CPU memory storage
    std::vector<float> host_storage_{};     ///< Host (CPU) data storage

#if TT_WITH_CUDA
    // GPU memory storage (conditional compilation)
    float* device_storage_ = nullptr;       ///< Device (GPU) data pointer
#endif

#if TT_WITH_CUDA
    /**
     * @brief Free CUDA device memory
     * @note Internal method for CUDA memory management
     */
    void free_device();

    /**
     * @brief Allocate CUDA memory and copy from source
     * @param src_device_ptr Source device pointer to copy from
     * @param count Number of elements to copy
     * @throw std::runtime_error If CUDA operations fail
     */
    void allocate_and_copy_device_from(const float* src_device_ptr, size_t count);
#endif
};

/**
 * @brief Neural Network Kernels Tensor Library namespace
 *
 * The nnk namespace contains all classes and functions for the nn_kernels_tt
 * high-performance tensor computation library. This library provides efficient
 * cross-platform tensor operations for neural network applications with support
 * for both CPU and GPU backends.
 */
} // namespace nnk



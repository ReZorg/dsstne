/*
 *  Copyright 2016-2026  Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License").
 *  You may not use this file except in compliance with the License.
 *  A copy of the License is located at
 *
 *  http://aws.amazon.com/apache2.0/
 *
 *  or in the "license" file accompanying this file.
 *  This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
 *  either express or implied.
 *
 *  See the License for the specific language governing permissions and limitations under the License.
 */

#ifndef GPU_COMMON_H
#define GPU_COMMON_H

/**
 * @file GpuCommon.h
 * @brief Common GPU utilities shared between engine and KNN modules
 * 
 * This header consolidates GPU-related constants, macros, and utility functions
 * used across different DSSTNE modules to avoid code duplication.
 */

#include <stdint.h>
#include <stdio.h>

// Forward declare NNFloat if not already defined
#ifndef NNFloat
#define NNFloat float
#endif

namespace dsstne {
namespace gpu {

//=============================================================================
// GPU Architecture Constants
//=============================================================================

// Thread block sizes for different GPU architectures
static const int SM_3X_THREADS_PER_BLOCK = 128;
static const int SM_5X_THREADS_PER_BLOCK = 128;
static const int SM_6X_THREADS_PER_BLOCK = 128;
static const int SM_7X_THREADS_PER_BLOCK = 128;
static const int SM_8X_THREADS_PER_BLOCK = 128;

// Maximum values for GPU operations
static const float MAX_FLOAT_VALUE = 999999999999999.0f;
static const float MIN_FLOAT_VALUE = -999999999999999.0f;

// Numerical stability constants
static const float EPSILON = 1e-8f;
static const float CENTERED_DIFF_DELTA = 1e-4f;

//=============================================================================
// Launch Bounds Macros
//=============================================================================

#if defined(__CUDA_ARCH__)
    #if (__CUDA_ARCH__ >= 800)
        #define GPU_LAUNCH_BOUNDS() __launch_bounds__(SM_8X_THREADS_PER_BLOCK, 8)
        #define GPU_LAUNCH_BOUNDS256() __launch_bounds__(256, 5)
    #elif (__CUDA_ARCH__ >= 700)
        #define GPU_LAUNCH_BOUNDS() __launch_bounds__(SM_7X_THREADS_PER_BLOCK, 8)
        #define GPU_LAUNCH_BOUNDS256() __launch_bounds__(256, 5)
    #elif (__CUDA_ARCH__ >= 600)
        #define GPU_LAUNCH_BOUNDS() __launch_bounds__(SM_6X_THREADS_PER_BLOCK, 8)
        #define GPU_LAUNCH_BOUNDS256() __launch_bounds__(256, 5)
    #elif (__CUDA_ARCH__ >= 500)
        #define GPU_LAUNCH_BOUNDS() __launch_bounds__(SM_5X_THREADS_PER_BLOCK, 8)
        #define GPU_LAUNCH_BOUNDS256() __launch_bounds__(256, 5)
    #else
        #define GPU_LAUNCH_BOUNDS() __launch_bounds__(SM_3X_THREADS_PER_BLOCK, 10)
        #define GPU_LAUNCH_BOUNDS256() __launch_bounds__(256, 4)
    #endif
#else
    // Non-CUDA compilation (for header parsing)
    #define GPU_LAUNCH_BOUNDS()
    #define GPU_LAUNCH_BOUNDS256()
#endif

#define GPU_LAUNCH_BOUNDS512() __launch_bounds__(512, 2)
#define GPU_LAUNCH_BOUNDS1024() __launch_bounds__(1024, 1)

//=============================================================================
// Error Checking Macros
//=============================================================================

#ifdef SYNCHRONOUS
    #define GPU_CHECK_ERROR(s) \
        { \
            cudaError_t status = cudaGetLastError(); \
            if (status != cudaSuccess) { \
                printf("Error: %s launching kernel %s\n", cudaGetErrorString(status), s); \
                exit(-1); \
            } \
            cudaDeviceSynchronize(); \
        }
#else
    #define GPU_CHECK_ERROR(s) \
        { \
            cudaError_t status = cudaGetLastError(); \
            if (status != cudaSuccess) { \
                printf("Error: %s launching kernel %s\n", cudaGetErrorString(status), s); \
                exit(-1); \
            } \
        }
#endif

// Safe CUDA call wrapper
#define CUDA_SAFE_CALL(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

//=============================================================================
// Utility Functions
//=============================================================================

/**
 * @brief Calculate grid dimensions for a 1D kernel launch
 * @param n Total number of elements
 * @param blockSize Number of threads per block
 * @return Number of blocks needed
 */
inline uint32_t calculateGridSize1D(uint64_t n, uint32_t blockSize) {
    return static_cast<uint32_t>((n + blockSize - 1) / blockSize);
}

/**
 * @brief Calculate grid dimensions for a 2D kernel launch
 * @param width Width dimension
 * @param height Height dimension
 * @param blockDimX Block dimension in X
 * @param blockDimY Block dimension in Y
 * @return dim3 grid size
 */
inline void calculateGridSize2D(uint64_t width, uint64_t height,
                                uint32_t blockDimX, uint32_t blockDimY,
                                uint32_t& gridDimX, uint32_t& gridDimY) {
    gridDimX = static_cast<uint32_t>((width + blockDimX - 1) / blockDimX);
    gridDimY = static_cast<uint32_t>((height + blockDimY - 1) / blockDimY);
}

/**
 * @brief Round up to the nearest multiple
 * @param value Value to round
 * @param multiple Multiple to round to
 * @return Rounded value
 */
inline uint64_t roundUpToMultiple(uint64_t value, uint64_t multiple) {
    return ((value + multiple - 1) / multiple) * multiple;
}

/**
 * @brief Check if a value is a power of two
 */
inline bool isPowerOfTwo(uint64_t x) {
    return (x != 0) && ((x & (x - 1)) == 0);
}

/**
 * @brief Get next power of two
 */
inline uint64_t nextPowerOfTwo(uint64_t x) {
    if (x == 0) return 1;
    x--;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x |= x >> 32;
    return x + 1;
}

//=============================================================================
// Gradient Computation Utilities
//=============================================================================

/**
 * @brief Numerical stability epsilon for gradient computations
 */
static const NNFloat GRADIENT_EPSILON = 1e-8f;

/**
 * @brief Compute centered finite difference approximation of derivative
 * 
 * Uses the formula: f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
 * This is more accurate than the non-centered formula: f'(x) ≈ (f(x+h) - f(x)) / h
 * 
 * @param f_plus Value of f(x+h)
 * @param f_minus Value of f(x-h)
 * @param h Step size
 * @return Approximate derivative
 */
inline NNFloat centeredFiniteDifference(NNFloat f_plus, NNFloat f_minus, NNFloat h) {
    return (f_plus - f_minus) / (2.0f * h);
}

/**
 * @brief Compute second-order centered finite difference
 * 
 * Uses the formula: f''(x) ≈ (f(x+h) - 2*f(x) + f(x-h)) / h²
 * 
 * @param f_plus Value of f(x+h)
 * @param f_center Value of f(x)
 * @param f_minus Value of f(x-h)
 * @param h Step size
 * @return Approximate second derivative
 */
inline NNFloat centeredSecondDerivative(NNFloat f_plus, NNFloat f_center, NNFloat f_minus, NNFloat h) {
    return (f_plus - 2.0f * f_center + f_minus) / (h * h);
}

//=============================================================================
// Memory Alignment Utilities
//=============================================================================

/**
 * @brief Minimum alignment for GPU memory operations
 */
static const size_t GPU_MEMORY_ALIGNMENT = 256;

/**
 * @brief Align size to GPU memory requirements
 */
inline size_t alignToGpuMemory(size_t size) {
    return ((size + GPU_MEMORY_ALIGNMENT - 1) / GPU_MEMORY_ALIGNMENT) * GPU_MEMORY_ALIGNMENT;
}

/**
 * @brief Calculate padding needed for tensor core operations
 * Tensor cores require dimensions to be multiples of 8 for FP16
 */
inline uint32_t calculateTensorCorePadding(uint32_t dimension) {
    const uint32_t TENSOR_CORE_ALIGNMENT = 8;
    uint32_t remainder = dimension % TENSOR_CORE_ALIGNMENT;
    return (remainder == 0) ? 0 : (TENSOR_CORE_ALIGNMENT - remainder);
}

} // namespace gpu
} // namespace dsstne

#endif // GPU_COMMON_H

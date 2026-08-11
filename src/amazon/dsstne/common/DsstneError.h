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

#ifndef DSSTNE_ERROR_H
#define DSSTNE_ERROR_H

#include <stdexcept>
#include <string>
#include <sstream>

namespace dsstne {

/**
 * @brief Error codes for DSSTNE operations
 */
enum class ErrorCode {
    // General errors
    OK = 0,
    UNKNOWN_ERROR = 1,
    INVALID_ARGUMENT = 2,
    OUT_OF_MEMORY = 3,
    NOT_IMPLEMENTED = 4,

    // GPU errors
    GPU_NOT_AVAILABLE = 100,
    GPU_MEMORY_ALLOCATION_FAILED = 101,
    GPU_KERNEL_LAUNCH_FAILED = 102,
    GPU_SYNCHRONIZATION_FAILED = 103,
    CUDA_ERROR = 104,

    // Network errors
    NETWORK_LOAD_FAILED = 200,
    NETWORK_SAVE_FAILED = 201,
    NETWORK_INVALID_CONFIG = 202,
    NETWORK_LAYER_NOT_FOUND = 203,
    NETWORK_WEIGHT_NOT_FOUND = 204,
    NETWORK_DATASET_MISSING = 205,

    // Data errors
    DATA_FORMAT_ERROR = 300,
    DATA_DIMENSION_MISMATCH = 301,
    DATA_TYPE_MISMATCH = 302,
    NETCDF_ERROR = 303,

    // Configuration errors
    CONFIG_PARSE_ERROR = 400,
    CONFIG_VALIDATION_ERROR = 401,
    CONFIG_VERSION_MISMATCH = 402,

    // KNN errors
    KNN_INDEX_ERROR = 500,
    KNN_SEARCH_FAILED = 501,

    // Multi-GPU errors
    MPI_ERROR = 600,
    MULTI_GPU_SYNC_ERROR = 601
};

/**
 * @brief Convert error code to human-readable string
 */
inline std::string ErrorCodeToString(ErrorCode code) {
    switch (code) {
        case ErrorCode::OK: return "OK";
        case ErrorCode::UNKNOWN_ERROR: return "Unknown error";
        case ErrorCode::INVALID_ARGUMENT: return "Invalid argument";
        case ErrorCode::OUT_OF_MEMORY: return "Out of memory";
        case ErrorCode::NOT_IMPLEMENTED: return "Not implemented";
        case ErrorCode::GPU_NOT_AVAILABLE: return "GPU not available";
        case ErrorCode::GPU_MEMORY_ALLOCATION_FAILED: return "GPU memory allocation failed";
        case ErrorCode::GPU_KERNEL_LAUNCH_FAILED: return "GPU kernel launch failed";
        case ErrorCode::GPU_SYNCHRONIZATION_FAILED: return "GPU synchronization failed";
        case ErrorCode::CUDA_ERROR: return "CUDA error";
        case ErrorCode::NETWORK_LOAD_FAILED: return "Network load failed";
        case ErrorCode::NETWORK_SAVE_FAILED: return "Network save failed";
        case ErrorCode::NETWORK_INVALID_CONFIG: return "Invalid network configuration";
        case ErrorCode::NETWORK_LAYER_NOT_FOUND: return "Network layer not found";
        case ErrorCode::NETWORK_WEIGHT_NOT_FOUND: return "Network weight not found";
        case ErrorCode::NETWORK_DATASET_MISSING: return "Network dataset missing";
        case ErrorCode::DATA_FORMAT_ERROR: return "Data format error";
        case ErrorCode::DATA_DIMENSION_MISMATCH: return "Data dimension mismatch";
        case ErrorCode::DATA_TYPE_MISMATCH: return "Data type mismatch";
        case ErrorCode::NETCDF_ERROR: return "NetCDF error";
        case ErrorCode::CONFIG_PARSE_ERROR: return "Configuration parse error";
        case ErrorCode::CONFIG_VALIDATION_ERROR: return "Configuration validation error";
        case ErrorCode::CONFIG_VERSION_MISMATCH: return "Configuration version mismatch";
        case ErrorCode::KNN_INDEX_ERROR: return "KNN index error";
        case ErrorCode::KNN_SEARCH_FAILED: return "KNN search failed";
        case ErrorCode::MPI_ERROR: return "MPI error";
        case ErrorCode::MULTI_GPU_SYNC_ERROR: return "Multi-GPU synchronization error";
        default: return "Unknown error code";
    }
}

/**
 * @brief Base exception class for all DSSTNE errors
 */
class DsstneException : public std::runtime_error {
protected:
    ErrorCode _code;
    std::string _details;
    std::string _file;
    int _line;

public:
    DsstneException(ErrorCode code, const std::string& message,
                    const std::string& file = "", int line = 0)
        : std::runtime_error(buildMessage(code, message, file, line)),
          _code(code), _details(message), _file(file), _line(line) {}

    ErrorCode code() const { return _code; }
    const std::string& details() const { return _details; }
    const std::string& file() const { return _file; }
    int line() const { return _line; }

private:
    static std::string buildMessage(ErrorCode code, const std::string& message,
                                    const std::string& file, int line) {
        std::ostringstream oss;
        oss << "[DSSTNE Error " << static_cast<int>(code) << "] "
            << ErrorCodeToString(code) << ": " << message;
        if (!file.empty()) {
            oss << " (at " << file << ":" << line << ")";
        }
        return oss.str();
    }
};

/**
 * @brief GPU-related exception
 */
class GpuException : public DsstneException {
public:
    GpuException(ErrorCode code, const std::string& message,
                 const std::string& file = "", int line = 0)
        : DsstneException(code, message, file, line) {}
};

/**
 * @brief Network-related exception
 */
class NetworkException : public DsstneException {
public:
    NetworkException(ErrorCode code, const std::string& message,
                     const std::string& file = "", int line = 0)
        : DsstneException(code, message, file, line) {}
};

/**
 * @brief Data-related exception
 */
class DataException : public DsstneException {
public:
    DataException(ErrorCode code, const std::string& message,
                  const std::string& file = "", int line = 0)
        : DsstneException(code, message, file, line) {}
};

/**
 * @brief Configuration-related exception
 */
class ConfigException : public DsstneException {
public:
    ConfigException(ErrorCode code, const std::string& message,
                    const std::string& file = "", int line = 0)
        : DsstneException(code, message, file, line) {}
};

/**
 * @brief KNN-related exception
 */
class KnnException : public DsstneException {
public:
    KnnException(ErrorCode code, const std::string& message,
                 const std::string& file = "", int line = 0)
        : DsstneException(code, message, file, line) {}
};

// Convenience macros for throwing exceptions with file/line info
#define DSSTNE_THROW(code, msg) \
    throw dsstne::DsstneException(code, msg, __FILE__, __LINE__)

#define DSSTNE_THROW_GPU(code, msg) \
    throw dsstne::GpuException(code, msg, __FILE__, __LINE__)

#define DSSTNE_THROW_NETWORK(code, msg) \
    throw dsstne::NetworkException(code, msg, __FILE__, __LINE__)

#define DSSTNE_THROW_DATA(code, msg) \
    throw dsstne::DataException(code, msg, __FILE__, __LINE__)

#define DSSTNE_THROW_CONFIG(code, msg) \
    throw dsstne::ConfigException(code, msg, __FILE__, __LINE__)

#define DSSTNE_THROW_KNN(code, msg) \
    throw dsstne::KnnException(code, msg, __FILE__, __LINE__)

// Assertion-style macros
#define DSSTNE_ASSERT(condition, code, msg) \
    do { \
        if (!(condition)) { \
            DSSTNE_THROW(code, msg); \
        } \
    } while (0)

#define DSSTNE_ASSERT_GPU(condition, code, msg) \
    do { \
        if (!(condition)) { \
            DSSTNE_THROW_GPU(code, msg); \
        } \
    } while (0)

} // namespace dsstne

#endif // DSSTNE_ERROR_H

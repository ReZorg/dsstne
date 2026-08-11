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

#ifndef DSSTNE_CONFIG_H
#define DSSTNE_CONFIG_H

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <stdexcept>
#include <cstdlib>

namespace dsstne {

/**
 * @brief Configuration version for backward compatibility
 */
struct ConfigVersion {
    uint32_t major;
    uint32_t minor;
    uint32_t patch;
    
    ConfigVersion() : major(1), minor(0), patch(0) {}
    ConfigVersion(uint32_t maj, uint32_t min, uint32_t pat)
        : major(maj), minor(min), patch(pat) {}
    
    std::string toString() const {
        return std::to_string(major) + "." + 
               std::to_string(minor) + "." + 
               std::to_string(patch);
    }
    
    bool isCompatibleWith(const ConfigVersion& other) const {
        // Major version must match for compatibility
        return major == other.major;
    }
};

/**
 * @brief GPU configuration settings
 */
struct GpuConfig {
    int deviceId;                   // CUDA device ID (-1 for auto)
    float memoryFraction;           // Fraction of GPU memory to use (0.0-1.0)
    bool allowGrowth;               // Allow dynamic memory growth
    bool vgpuMode;                  // Virtual GPU mode optimizations
    uint32_t maxConcurrentKernels;  // Max concurrent kernel launches
    
    GpuConfig() 
        : deviceId(-1), 
          memoryFraction(0.9f),
          allowGrowth(true),
          vgpuMode(false),
          maxConcurrentKernels(16) {}
};

/**
 * @brief Training configuration settings
 */
struct TrainingConfig {
    std::string optimizer;          // sgd, momentum, adam, etc.
    float learningRate;             // Initial learning rate
    float momentum;                 // Momentum factor
    float weightDecay;              // L2 regularization
    float l1Regularization;         // L1 regularization
    float gradientClip;             // Gradient clipping threshold
    bool shuffleIndices;            // Shuffle training data
    uint32_t checkpointInterval;    // Epochs between checkpoints
    std::string checkpointPath;     // Path for checkpoint files
    
    TrainingConfig()
        : optimizer("sgd"),
          learningRate(0.01f),
          momentum(0.9f),
          weightDecay(0.0001f),
          l1Regularization(0.0f),
          gradientClip(0.0f),
          shuffleIndices(true),
          checkpointInterval(1),
          checkpointPath("checkpoints") {}
};

/**
 * @brief Network configuration settings
 */
struct NetworkConfig {
    std::string name;               // Network name
    std::string kind;               // FeedForward, AutoEncoder, etc.
    std::string errorFunction;      // Loss function
    uint32_t batchSize;             // Training/inference batch size
    std::string dataPath;           // Path to data files
    std::string modelPath;          // Path to model files
    
    NetworkConfig()
        : kind("FeedForward"),
          errorFunction("CrossEntropy"),
          batchSize(32) {}
};

/**
 * @brief Inference configuration settings
 */
struct InferenceConfig {
    uint32_t topK;                  // Top-K predictions to return
    float threshold;                // Score threshold for predictions
    bool returnScores;              // Include prediction scores
    bool returnEmbeddings;          // Return intermediate embeddings
    std::string embeddingLayer;     // Layer for embedding extraction
    
    InferenceConfig()
        : topK(10),
          threshold(0.0f),
          returnScores(true),
          returnEmbeddings(false) {}
};

/**
 * @brief KNN configuration settings
 */
struct KnnConfig {
    uint32_t k;                     // Number of neighbors
    bool useGpu;                    // Use GPU-accelerated KNN
    uint32_t batchSize;             // Query batch size
    std::string metric;             // Distance metric (euclidean, cosine)
    
    KnnConfig()
        : k(10),
          useGpu(true),
          batchSize(1024),
          metric("euclidean") {}
};

/**
 * @brief Logging configuration settings
 */
struct LoggingConfig {
    std::string level;              // TRACE, DEBUG, INFO, WARN, ERROR, FATAL
    std::string file;               // Log file path (empty for stdout)
    bool console;                   // Enable console output
    bool timestamps;                // Include timestamps
    
    LoggingConfig()
        : level("INFO"),
          console(true),
          timestamps(true) {}
};

/**
 * @brief Unified configuration class for DSSTNE
 * 
 * Consolidates all configuration settings with support for:
 * - JSON configuration files
 * - YAML configuration files
 * - Environment variable overrides
 * - Command-line argument overrides
 * - Configuration validation
 */
class DsstneConfig {
public:
    // Sub-configurations
    ConfigVersion version;
    GpuConfig gpu;
    TrainingConfig training;
    NetworkConfig network;
    InferenceConfig inference;
    KnnConfig knn;
    LoggingConfig logging;
    
    /**
     * @brief Default constructor
     */
    DsstneConfig() = default;
    
    /**
     * @brief Load configuration from JSON file
     */
    static DsstneConfig loadJson(const std::string& path);
    
    /**
     * @brief Load configuration from YAML file
     */
    static DsstneConfig loadYaml(const std::string& path);
    
    /**
     * @brief Save configuration to JSON file
     */
    void saveJson(const std::string& path) const;
    
    /**
     * @brief Save configuration to YAML file
     */
    void saveYaml(const std::string& path) const;
    
    /**
     * @brief Apply environment variable overrides
     * 
     * Looks for environment variables with DSSTNE_ prefix:
     * - DSSTNE_GPU_DEVICE_ID
     * - DSSTNE_LEARNING_RATE
     * - DSSTNE_BATCH_SIZE
     * - DSSTNE_LOG_LEVEL
     * etc.
     */
    void applyEnvironmentOverrides();
    
    /**
     * @brief Parse command-line arguments
     * @param argc Argument count
     * @param argv Argument values
     */
    void parseCommandLine(int argc, char** argv);
    
    /**
     * @brief Validate configuration
     * @return List of validation errors (empty if valid)
     */
    std::vector<std::string> validate() const;
    
    /**
     * @brief Check if configuration is valid
     */
    bool isValid() const { return validate().empty(); }
    
    /**
     * @brief Get configuration value by path (e.g., "gpu.deviceId")
     */
    template<typename T>
    T get(const std::string& path) const;
    
    /**
     * @brief Set configuration value by path
     */
    template<typename T>
    void set(const std::string& path, const T& value);
    
    /**
     * @brief Merge another configuration (other values take precedence)
     */
    void merge(const DsstneConfig& other);
    
    /**
     * @brief Get singleton instance (for global configuration)
     */
    static DsstneConfig& instance() {
        static DsstneConfig config;
        return config;
    }

private:
    /**
     * @brief Get environment variable with optional default
     */
    static std::string getEnv(const std::string& name, const std::string& defaultVal = "");
    
    /**
     * @brief Parse integer from string with validation
     */
    static int parseIntEnv(const std::string& name, int defaultVal);
    
    /**
     * @brief Parse float from string with validation
     */
    static float parseFloatEnv(const std::string& name, float defaultVal);
    
    /**
     * @brief Parse boolean from string
     */
    static bool parseBoolEnv(const std::string& name, bool defaultVal);
};

//=============================================================================
// Inline implementations
//=============================================================================

inline std::string DsstneConfig::getEnv(const std::string& name, const std::string& defaultVal) {
    const char* val = std::getenv(name.c_str());
    return val ? std::string(val) : defaultVal;
}

inline int DsstneConfig::parseIntEnv(const std::string& name, int defaultVal) {
    const char* val = std::getenv(name.c_str());
    if (!val) return defaultVal;
    try {
        return std::stoi(val);
    } catch (...) {
        return defaultVal;
    }
}

inline float DsstneConfig::parseFloatEnv(const std::string& name, float defaultVal) {
    const char* val = std::getenv(name.c_str());
    if (!val) return defaultVal;
    try {
        return std::stof(val);
    } catch (...) {
        return defaultVal;
    }
}

inline bool DsstneConfig::parseBoolEnv(const std::string& name, bool defaultVal) {
    const char* val = std::getenv(name.c_str());
    if (!val) return defaultVal;
    std::string s(val);
    return s == "1" || s == "true" || s == "TRUE" || s == "yes" || s == "YES";
}

inline void DsstneConfig::applyEnvironmentOverrides() {
    // GPU configuration
    gpu.deviceId = parseIntEnv("DSSTNE_GPU_DEVICE_ID", gpu.deviceId);
    gpu.memoryFraction = parseFloatEnv("DSSTNE_GPU_MEMORY_FRACTION", gpu.memoryFraction);
    gpu.vgpuMode = parseBoolEnv("DSSTNE_VGPU_MODE", gpu.vgpuMode);
    
    // Training configuration
    training.learningRate = parseFloatEnv("DSSTNE_LEARNING_RATE", training.learningRate);
    training.momentum = parseFloatEnv("DSSTNE_MOMENTUM", training.momentum);
    training.weightDecay = parseFloatEnv("DSSTNE_WEIGHT_DECAY", training.weightDecay);
    
    // Network configuration
    network.batchSize = static_cast<uint32_t>(parseIntEnv("DSSTNE_BATCH_SIZE", network.batchSize));
    
    // Logging configuration
    std::string logLevel = getEnv("DSSTNE_LOG_LEVEL", "");
    if (!logLevel.empty()) {
        logging.level = logLevel;
    }
    std::string logFile = getEnv("DSSTNE_LOG_FILE", "");
    if (!logFile.empty()) {
        logging.file = logFile;
    }
}

inline std::vector<std::string> DsstneConfig::validate() const {
    std::vector<std::string> errors;
    
    // GPU validation
    if (gpu.memoryFraction <= 0.0f || gpu.memoryFraction > 1.0f) {
        errors.push_back("gpu.memoryFraction must be between 0.0 and 1.0");
    }
    
    // Training validation
    if (training.learningRate <= 0.0f) {
        errors.push_back("training.learningRate must be positive");
    }
    if (training.momentum < 0.0f || training.momentum >= 1.0f) {
        errors.push_back("training.momentum must be in [0.0, 1.0)");
    }
    
    // Network validation
    if (network.batchSize == 0) {
        errors.push_back("network.batchSize must be positive");
    }
    
    // Inference validation
    if (inference.topK == 0) {
        errors.push_back("inference.topK must be positive");
    }
    
    // KNN validation
    if (knn.k == 0) {
        errors.push_back("knn.k must be positive");
    }
    
    return errors;
}

} // namespace dsstne

#endif // DSSTNE_CONFIG_H

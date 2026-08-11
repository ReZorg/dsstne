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

#ifndef PARALLELISM_DETECTOR_H
#define PARALLELISM_DETECTOR_H

#include <vector>
#include <string>
#include <cstdint>
#include <map>

// Forward declarations
class NNNetwork;
class NNLayer;
class NNWeight;

namespace dsstne {

/**
 * @brief Parallelism strategy for neural network execution
 */
enum class ParallelismStrategy {
    MODEL_PARALLEL,     // Distribute layers across GPUs (default DSSTNE behavior)
    DATA_PARALLEL,      // Distribute data batches across GPUs
    HYBRID,             // Combination of model and data parallel
    AUTO                // Automatically detect best strategy
};

/**
 * @brief Layer-level parallelism recommendation
 */
struct LayerParallelismInfo {
    std::string layerName;
    uint64_t parameterCount;
    uint64_t activationMemory;
    ParallelismStrategy recommendedStrategy;
    float computeIntensity;  // FLOPS per byte
    bool isBottleneck;
};

/**
 * @brief Network-level parallelism analysis result
 */
struct NetworkParallelismAnalysis {
    ParallelismStrategy overallRecommendation;
    std::vector<LayerParallelismInfo> layerInfo;
    uint64_t totalParameters;
    uint64_t totalActivationMemory;
    float estimatedDataParallelEfficiency;
    float estimatedModelParallelEfficiency;
    std::string analysisNotes;
};

/**
 * @brief Utility class to detect optimal parallelism strategy
 * 
 * This class analyzes neural network topology and recommends the best
 * parallelism strategy based on:
 * - Layer sizes and parameter counts
 * - Memory requirements
 * - Available GPU memory
 * - Communication overhead estimates
 */
class ParallelismDetector {
public:
    /**
     * @brief Constructor
     * @param numGpus Number of available GPUs
     * @param gpuMemoryMB Available GPU memory per device in MB
     */
    ParallelismDetector(uint32_t numGpus = 1, uint64_t gpuMemoryMB = 8192);

    /**
     * @brief Analyze a network and recommend parallelism strategy
     * @param network The neural network to analyze
     * @return Analysis result with recommendations
     */
    NetworkParallelismAnalysis analyze(const NNNetwork* network) const;

    /**
     * @brief Check if data-parallel is beneficial for a weight matrix
     * 
     * Data-parallel is beneficial when:
     * - Weight matrix fits in single GPU memory
     * - Batch size is large enough to utilize multiple GPUs
     * - Communication overhead is smaller than computation time
     * 
     * @param weight The weight matrix to analyze
     * @param batchSize Current batch size
     * @return true if data-parallel is recommended
     */
    bool shouldUseDataParallel(const NNWeight* weight, uint32_t batchSize) const;

    /**
     * @brief Detect if the network is better suited for model or data parallel
     * @param network The neural network
     * @param batchSize Batch size for inference/training
     * @return Recommended strategy
     */
    ParallelismStrategy detectBestStrategy(const NNNetwork* network, uint32_t batchSize) const;

    /**
     * @brief Get optimal batch size for data-parallel execution
     * @param network The neural network
     * @return Recommended batch size per GPU
     */
    uint32_t getOptimalBatchSizePerGpu(const NNNetwork* network) const;

    /**
     * @brief Set weight synchronization threshold
     * 
     * Weights smaller than this threshold will be replicated (data-parallel).
     * Weights larger will be distributed (model-parallel).
     * 
     * @param thresholdMB Threshold in megabytes
     */
    void setWeightThreshold(float thresholdMB);

    /**
     * @brief Set communication overhead factor
     * 
     * Higher values favor model-parallel (less synchronization).
     * Lower values favor data-parallel.
     * 
     * @param factor Communication overhead factor (default 1.0)
     */
    void setCommunicationOverheadFactor(float factor);

private:
    uint32_t _numGpus;
    uint64_t _gpuMemoryBytes;
    float _weightThresholdMB;
    float _communicationOverheadFactor;

    // Constants for analysis
    static const uint64_t BYTES_PER_FLOAT = 4;
    static const float DEFAULT_WEIGHT_THRESHOLD_MB;
    static const float DEFAULT_COMMUNICATION_FACTOR;

    /**
     * @brief Calculate memory footprint of a layer
     */
    uint64_t calculateLayerMemory(const NNLayer* layer) const;

    /**
     * @brief Calculate parameter count of a weight matrix
     */
    uint64_t calculateWeightParameters(const NNWeight* weight) const;

    /**
     * @brief Estimate communication overhead for data-parallel sync
     */
    float estimateSyncOverhead(uint64_t parameterCount) const;

    /**
     * @brief Estimate computation time for a layer
     */
    float estimateComputeTime(const NNLayer* layer, uint32_t batchSize) const;
};

//=============================================================================
// Inline implementations
//=============================================================================

inline ParallelismDetector::ParallelismDetector(uint32_t numGpus, uint64_t gpuMemoryMB)
    : _numGpus(numGpus),
      _gpuMemoryBytes(gpuMemoryMB * 1024 * 1024),
      _weightThresholdMB(DEFAULT_WEIGHT_THRESHOLD_MB),
      _communicationOverheadFactor(DEFAULT_COMMUNICATION_FACTOR) {
}

inline void ParallelismDetector::setWeightThreshold(float thresholdMB) {
    _weightThresholdMB = thresholdMB;
}

inline void ParallelismDetector::setCommunicationOverheadFactor(float factor) {
    _communicationOverheadFactor = factor;
}

} // namespace dsstne

#endif // PARALLELISM_DETECTOR_H

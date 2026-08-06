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

#include "ParallelismDetector.h"
#include "NNNetwork.h"
#include "NNLayer.h"
#include "NNWeight.h"
#include "GpuTypes.h"
#include <sstream>
#include <algorithm>

namespace dsstne {

// Static constant definitions
const float ParallelismDetector::DEFAULT_WEIGHT_THRESHOLD_MB = 256.0f;  // 256 MB threshold
const float ParallelismDetector::DEFAULT_COMMUNICATION_FACTOR = 1.0f;

NetworkParallelismAnalysis ParallelismDetector::analyze(const NNNetwork* network) const {
    NetworkParallelismAnalysis result;
    result.totalParameters = 0;
    result.totalActivationMemory = 0;
    
    if (network == nullptr) {
        result.overallRecommendation = ParallelismStrategy::MODEL_PARALLEL;
        result.analysisNotes = "Null network provided";
        return result;
    }

    // Get all layers from the network
    const std::vector<NNLayer*>& layers = network->GetLayers();
    
    for (const NNLayer* layer : layers) {
        LayerParallelismInfo info;
        info.layerName = layer->GetName();
        info.activationMemory = calculateLayerMemory(layer);
        info.parameterCount = 0;
        info.isBottleneck = false;
        
        // Calculate parameter count from incoming weights
        const std::vector<NNWeight*>& incomingWeights = layer->GetIncomingWeights();
        for (const NNWeight* weight : incomingWeights) {
            info.parameterCount += calculateWeightParameters(weight);
        }
        
        result.totalParameters += info.parameterCount;
        result.totalActivationMemory += info.activationMemory;
        
        // Calculate compute intensity (FLOPS per byte)
        // For fully connected: ~2 FLOPS per parameter per sample
        uint64_t memoryBytes = info.parameterCount * BYTES_PER_FLOAT;
        info.computeIntensity = (memoryBytes > 0) ? 
            (2.0f * info.parameterCount) / memoryBytes : 0.0f;
        
        // Determine layer-level recommendation
        float weightMemoryMB = (info.parameterCount * BYTES_PER_FLOAT) / (1024.0f * 1024.0f);
        if (weightMemoryMB < _weightThresholdMB && _numGpus > 1) {
            info.recommendedStrategy = ParallelismStrategy::DATA_PARALLEL;
        } else {
            info.recommendedStrategy = ParallelismStrategy::MODEL_PARALLEL;
        }
        
        // Mark bottleneck layers (largest parameter count)
        if (info.parameterCount > result.totalParameters / layers.size() * 2) {
            info.isBottleneck = true;
        }
        
        result.layerInfo.push_back(info);
    }
    
    // Calculate overall efficiency estimates
    float totalWeightMemoryMB = (result.totalParameters * BYTES_PER_FLOAT) / (1024.0f * 1024.0f);
    
    // Data parallel efficiency: depends on communication overhead
    float syncOverhead = estimateSyncOverhead(result.totalParameters);
    result.estimatedDataParallelEfficiency = 1.0f / (1.0f + syncOverhead * _communicationOverheadFactor);
    
    // Model parallel efficiency: depends on load balancing
    float maxLayerParams = 0;
    for (const auto& info : result.layerInfo) {
        maxLayerParams = std::max(maxLayerParams, static_cast<float>(info.parameterCount));
    }
    float avgLayerParams = static_cast<float>(result.totalParameters) / std::max(1ul, result.layerInfo.size());
    result.estimatedModelParallelEfficiency = avgLayerParams / std::max(1.0f, maxLayerParams);
    
    // Determine overall recommendation
    if (_numGpus == 1) {
        result.overallRecommendation = ParallelismStrategy::MODEL_PARALLEL;
        result.analysisNotes = "Single GPU detected, model-parallel by default";
    } else if (totalWeightMemoryMB < _gpuMemoryBytes / (1024 * 1024) / 2) {
        // Weights fit in half of GPU memory - data parallel may be beneficial
        if (result.estimatedDataParallelEfficiency > result.estimatedModelParallelEfficiency) {
            result.overallRecommendation = ParallelismStrategy::DATA_PARALLEL;
            result.analysisNotes = "Weights fit in GPU memory, data-parallel recommended for throughput";
        } else {
            result.overallRecommendation = ParallelismStrategy::MODEL_PARALLEL;
            result.analysisNotes = "Model-parallel preferred due to layer imbalance";
        }
    } else {
        // Large model - model parallel required
        result.overallRecommendation = ParallelismStrategy::MODEL_PARALLEL;
        result.analysisNotes = "Large model detected, model-parallel required to fit in GPU memory";
    }
    
    // Check for hybrid potential
    bool hasSmallLayers = false;
    bool hasLargeLayers = false;
    for (const auto& info : result.layerInfo) {
        float layerMemMB = (info.parameterCount * BYTES_PER_FLOAT) / (1024.0f * 1024.0f);
        if (layerMemMB < _weightThresholdMB / 4) hasSmallLayers = true;
        if (layerMemMB > _weightThresholdMB) hasLargeLayers = true;
    }
    
    if (hasSmallLayers && hasLargeLayers && _numGpus > 2) {
        result.overallRecommendation = ParallelismStrategy::HYBRID;
        result.analysisNotes += "; Mixed layer sizes suggest hybrid strategy could be beneficial";
    }
    
    return result;
}

bool ParallelismDetector::shouldUseDataParallel(const NNWeight* weight, uint32_t batchSize) const {
    if (weight == nullptr || _numGpus <= 1) {
        return false;
    }
    
    uint64_t params = calculateWeightParameters(weight);
    float weightMemoryMB = (params * BYTES_PER_FLOAT) / (1024.0f * 1024.0f);
    
    // Check if weight fits in GPU memory
    if (weightMemoryMB > _gpuMemoryBytes / (1024 * 1024)) {
        return false;  // Too large, must use model-parallel
    }
    
    // Check if batch size is large enough to benefit from data-parallel
    if (batchSize < _numGpus * 8) {
        return false;  // Batch too small for efficient data-parallel
    }
    
    // Estimate computation vs communication
    float computeTime = params * batchSize * 2.0f;  // 2 FLOPS per param per sample
    float syncOverhead = estimateSyncOverhead(params);
    
    // Data-parallel beneficial if compute time >> sync time
    return computeTime > syncOverhead * _communicationOverheadFactor * 10.0f;
}

ParallelismStrategy ParallelismDetector::detectBestStrategy(const NNNetwork* network, uint32_t batchSize) const {
    NetworkParallelismAnalysis analysis = analyze(network);
    
    // Override based on batch size
    if (batchSize < _numGpus * 8) {
        return ParallelismStrategy::MODEL_PARALLEL;
    }
    
    return analysis.overallRecommendation;
}

uint32_t ParallelismDetector::getOptimalBatchSizePerGpu(const NNNetwork* network) const {
    if (network == nullptr || _numGpus == 0) {
        return 32;  // Default batch size
    }
    
    NetworkParallelismAnalysis analysis = analyze(network);
    
    // Calculate based on available GPU memory and model size
    uint64_t modelMemory = analysis.totalParameters * BYTES_PER_FLOAT;
    uint64_t activationMemoryPerSample = analysis.totalActivationMemory;
    
    // Reserve 20% of GPU memory for workspace
    uint64_t availableMemory = static_cast<uint64_t>(_gpuMemoryBytes * 0.8);
    
    // Available memory for batch = total - model weights
    if (availableMemory <= modelMemory) {
        return 1;  // Minimum batch size
    }
    
    uint64_t batchMemory = availableMemory - modelMemory;
    uint32_t maxBatchSize = static_cast<uint32_t>(batchMemory / std::max(1UL, activationMemoryPerSample));
    
    // Round down to power of 2 for efficiency
    uint32_t optimalBatch = 1;
    while (optimalBatch * 2 <= maxBatchSize && optimalBatch < 1024) {
        optimalBatch *= 2;
    }
    
    return std::max(1u, optimalBatch);
}

uint64_t ParallelismDetector::calculateLayerMemory(const NNLayer* layer) const {
    if (layer == nullptr) return 0;
    
    uint64_t numUnits = layer->GetNumUnits();
    
    // Memory for activations and deltas (both stored for backprop)
    return numUnits * BYTES_PER_FLOAT * 2;
}

uint64_t ParallelismDetector::calculateWeightParameters(const NNWeight* weight) const {
    if (weight == nullptr) return 0;
    
    // Get dimensions from the weight
    std::vector<uint64_t> dimensions;
    const_cast<NNWeight*>(weight)->GetDimensions(dimensions);
    
    uint64_t totalParams = 1;
    for (uint64_t dim : dimensions) {
        totalParams *= dim;
    }
    
    return totalParams;
}

float ParallelismDetector::estimateSyncOverhead(uint64_t parameterCount) const {
    // Estimate based on bandwidth (assume 10 GB/s per GPU link)
    const float BANDWIDTH_GBPS = 10.0f;
    float dataGB = (parameterCount * BYTES_PER_FLOAT) / (1024.0f * 1024.0f * 1024.0f);
    
    // AllReduce requires 2*(N-1)/N data transfers
    float allReduceFactor = 2.0f * (_numGpus - 1) / _numGpus;
    
    return (dataGB * allReduceFactor) / BANDWIDTH_GBPS;
}

float ParallelismDetector::estimateComputeTime(const NNLayer* layer, uint32_t batchSize) const {
    if (layer == nullptr) return 0.0f;
    
    uint64_t numUnits = layer->GetNumUnits();
    
    // Estimate based on FLOPS (assume 10 TFLOPS per GPU)
    const float TFLOPS = 10.0f;
    float flops = numUnits * batchSize * 2.0f;  // Matrix multiply FLOPS
    
    return flops / (TFLOPS * 1e12f);
}

} // namespace dsstne

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

#include "EmbeddingExtractor.h"

#include <stdexcept>
#include <cstring>

namespace dsstne {
namespace pipeline {

EmbeddingExtractor::EmbeddingExtractor(NNNetwork* network)
    : _network(network)
    , _copyToHost(true)
    , _normalize(false)
    , _poolingMode(PoolingMode::Mean) {
    
    if (!_network) {
        throw std::invalid_argument("Network cannot be null");
    }
}

void EmbeddingExtractor::setCopyToHost(bool copyToHost) {
    _copyToHost = copyToHost;
}

void EmbeddingExtractor::setNormalize(bool normalize) {
    _normalize = normalize;
}

void EmbeddingExtractor::setPoolingMode(PoolingMode mode) {
    _poolingMode = mode;
}

void EmbeddingExtractor::setTargetLayer(const std::string& layerName) {
    _targetLayerName = layerName;
}

std::vector<float> EmbeddingExtractor::extractEmbedding(
    const std::string& layerName,
    uint32_t batchIndex) const {
    
    NNLayer* layer = _network->GetLayer(layerName);
    if (!layer) {
        throw std::runtime_error("Layer not found: " + layerName);
    }
    
    uint32_t stride = layer->GetStride();
    uint32_t batchSize = _network->GetBatch();
    
    if (batchIndex >= batchSize) {
        throw std::out_of_range("Batch index out of range");
    }
    
    std::vector<float> embedding(stride);
    
    // Copy from GPU
    float* deviceData = layer->GetUnitBuffer();
    cudaMemcpy(embedding.data(), 
               deviceData + batchIndex * stride,
               stride * sizeof(float),
               cudaMemcpyDeviceToHost);
    
    // Normalize if requested
    if (_normalize) {
        normalizeEmbedding(embedding);
    }
    
    return embedding;
}

std::vector<std::vector<float>> EmbeddingExtractor::extractBatchEmbeddings(
    const std::string& layerName) const {
    
    NNLayer* layer = _network->GetLayer(layerName);
    if (!layer) {
        throw std::runtime_error("Layer not found: " + layerName);
    }
    
    uint32_t stride = layer->GetStride();
    uint32_t batchSize = _network->GetBatch();
    
    std::vector<std::vector<float>> embeddings(batchSize);
    
    // Get all batch data at once
    std::vector<float> allData(stride * batchSize);
    float* deviceData = layer->GetUnitBuffer();
    cudaMemcpy(allData.data(),
               deviceData,
               stride * batchSize * sizeof(float),
               cudaMemcpyDeviceToHost);
    
    // Split into individual embeddings
    for (uint32_t i = 0; i < batchSize; ++i) {
        embeddings[i].resize(stride);
        std::memcpy(embeddings[i].data(),
                    allData.data() + i * stride,
                    stride * sizeof(float));
        
        if (_normalize) {
            normalizeEmbedding(embeddings[i]);
        }
    }
    
    return embeddings;
}

GpuBuffer<float> EmbeddingExtractor::extractEmbeddingGpu(
    const std::string& layerName,
    uint32_t batchIndex) const {
    
    NNLayer* layer = _network->GetLayer(layerName);
    if (!layer) {
        throw std::runtime_error("Layer not found: " + layerName);
    }
    
    uint32_t stride = layer->GetStride();
    
    // Create GPU buffer and copy the specific batch
    GpuBuffer<float> buffer(stride);
    float* deviceData = layer->GetUnitBuffer();
    
    cudaMemcpy(buffer.data(),
               deviceData + batchIndex * stride,
               stride * sizeof(float),
               cudaMemcpyDeviceToDevice);
    
    return buffer;
}

std::vector<std::string> EmbeddingExtractor::getAvailableLayers() const {
    std::vector<std::string> layerNames;
    
    auto& layers = _network->GetLayers();
    for (auto* layer : layers) {
        if (layer && !layer->GetName().empty()) {
            layerNames.push_back(layer->GetName());
        }
    }
    
    return layerNames;
}

std::pair<uint32_t, uint32_t> EmbeddingExtractor::getEmbeddingDimensions(
    const std::string& layerName) const {
    
    NNLayer* layer = _network->GetLayer(layerName);
    if (!layer) {
        throw std::runtime_error("Layer not found: " + layerName);
    }
    
    return {layer->GetStride(), _network->GetBatch()};
}

void EmbeddingExtractor::normalizeEmbedding(std::vector<float>& embedding) const {
    float norm = 0.0f;
    for (float v : embedding) {
        norm += v * v;
    }
    
    norm = std::sqrt(norm);
    if (norm > 1e-8f) {
        for (float& v : embedding) {
            v /= norm;
        }
    }
}

} // namespace pipeline
} // namespace dsstne

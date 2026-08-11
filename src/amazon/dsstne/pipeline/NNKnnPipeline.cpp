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

#include "NNKnnPipeline.h"

#include <stdexcept>
#include <algorithm>

namespace dsstne {
namespace pipeline {

NNKnnPipeline::NNKnnPipeline(
    NNNetwork* network,
    const std::string& embeddingLayerName)
    : _network(network)
    , _embeddingLayerName(embeddingLayerName)
    , _embeddingExtractor(network)
    , _defaultK(100)
    , _distanceMetric(DistanceMetric::Cosine)
    , _useGpuKnn(true)
    , _normalizeEmbeddings(true) {
    
    if (!_network) {
        throw std::invalid_argument("Network cannot be null");
    }
    
    _embeddingExtractor.setNormalize(_normalizeEmbeddings);
}

void NNKnnPipeline::setDistanceMetric(DistanceMetric metric) {
    _distanceMetric = metric;
}

void NNKnnPipeline::setUseGpuKnn(bool useGpu) {
    _useGpuKnn = useGpu;
}

void NNKnnPipeline::setNormalizeEmbeddings(bool normalize) {
    _normalizeEmbeddings = normalize;
    _embeddingExtractor.setNormalize(normalize);
}

void NNKnnPipeline::setDefaultK(uint32_t k) {
    _defaultK = k;
}

void NNKnnPipeline::buildIndex(
    const std::vector<std::vector<float>>& embeddings,
    const std::vector<uint32_t>& ids) {
    
    if (embeddings.empty()) {
        throw std::invalid_argument("Embeddings cannot be empty");
    }
    
    if (!ids.empty() && ids.size() != embeddings.size()) {
        throw std::invalid_argument("IDs size must match embeddings size");
    }
    
    _indexEmbeddings = embeddings;
    _indexIds = ids.empty() ? generateSequentialIds(embeddings.size()) : ids;
    
    // Copy to GPU if using GPU KNN
    if (_useGpuKnn) {
        buildGpuIndex();
    }
}

void NNKnnPipeline::buildIndexFromNetwork(
    const std::vector<std::vector<NNDataSet*>>& inputBatches,
    const std::vector<uint32_t>& ids) {
    
    std::vector<std::vector<float>> allEmbeddings;
    
    for (const auto& batch : inputBatches) {
        // Run forward pass
        _network->SetPosition(0);
        _network->PredictBatch();
        
        // Extract embeddings
        auto batchEmbeddings = _embeddingExtractor.extractBatchEmbeddings(_embeddingLayerName);
        
        for (auto& emb : batchEmbeddings) {
            allEmbeddings.push_back(std::move(emb));
        }
    }
    
    buildIndex(allEmbeddings, ids);
}

std::vector<KnnResult> NNKnnPipeline::queryWithInputData(
    NNDataSet* inputData,
    uint32_t k) {
    
    if (k == 0) {
        k = _defaultK;
    }
    
    // Run forward pass
    _network->SetPosition(0);
    _network->PredictBatch();
    
    // Extract query embedding
    auto queryEmbeddings = _embeddingExtractor.extractBatchEmbeddings(_embeddingLayerName);
    
    // Run KNN query
    return query(queryEmbeddings, k);
}

std::vector<KnnResult> NNKnnPipeline::query(
    const std::vector<std::vector<float>>& queryEmbeddings,
    uint32_t k) {
    
    if (_indexEmbeddings.empty()) {
        throw std::runtime_error("Index not built. Call buildIndex first.");
    }
    
    if (k == 0) {
        k = _defaultK;
    }
    
    std::vector<KnnResult> results;
    results.reserve(queryEmbeddings.size());
    
    for (const auto& queryEmb : queryEmbeddings) {
        results.push_back(querySingle(queryEmb, k));
    }
    
    return results;
}

KnnResult NNKnnPipeline::querySingle(
    const std::vector<float>& queryEmbedding,
    uint32_t k) const {
    
    KnnResult result;
    
    // Calculate distances to all index embeddings
    std::vector<std::pair<float, uint32_t>> distances;
    distances.reserve(_indexEmbeddings.size());
    
    for (size_t i = 0; i < _indexEmbeddings.size(); ++i) {
        float dist = calculateDistance(queryEmbedding, _indexEmbeddings[i]);
        distances.emplace_back(dist, i);
    }
    
    // Sort by distance
    std::partial_sort(
        distances.begin(),
        distances.begin() + std::min(static_cast<size_t>(k), distances.size()),
        distances.end()
    );
    
    // Extract top-K
    size_t numResults = std::min(static_cast<size_t>(k), distances.size());
    result.ids.reserve(numResults);
    result.distances.reserve(numResults);
    
    for (size_t i = 0; i < numResults; ++i) {
        result.ids.push_back(_indexIds[distances[i].second]);
        result.distances.push_back(distances[i].first);
    }
    
    return result;
}

float NNKnnPipeline::calculateDistance(
    const std::vector<float>& a,
    const std::vector<float>& b) const {
    
    if (a.size() != b.size()) {
        throw std::invalid_argument("Embedding dimensions must match");
    }
    
    switch (_distanceMetric) {
        case DistanceMetric::Cosine:
            return cosineDistance(a, b);
        case DistanceMetric::Euclidean:
            return euclideanDistance(a, b);
        case DistanceMetric::DotProduct:
            return -dotProduct(a, b);  // Negative for min-sort
        default:
            return cosineDistance(a, b);
    }
}

float NNKnnPipeline::cosineDistance(
    const std::vector<float>& a,
    const std::vector<float>& b) const {
    
    float dotProd = 0.0f;
    float normA = 0.0f;
    float normB = 0.0f;
    
    for (size_t i = 0; i < a.size(); ++i) {
        dotProd += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }
    
    float denom = std::sqrt(normA) * std::sqrt(normB);
    if (denom < 1e-8f) {
        return 1.0f;
    }
    
    return 1.0f - (dotProd / denom);
}

float NNKnnPipeline::euclideanDistance(
    const std::vector<float>& a,
    const std::vector<float>& b) const {
    
    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

float NNKnnPipeline::dotProduct(
    const std::vector<float>& a,
    const std::vector<float>& b) const {
    
    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

std::vector<uint32_t> NNKnnPipeline::generateSequentialIds(size_t count) const {
    std::vector<uint32_t> ids(count);
    for (size_t i = 0; i < count; ++i) {
        ids[i] = static_cast<uint32_t>(i);
    }
    return ids;
}

void NNKnnPipeline::buildGpuIndex() {
    if (_indexEmbeddings.empty()) {
        return;
    }
    
    // Calculate total size
    size_t embeddingDim = _indexEmbeddings[0].size();
    size_t totalSize = _indexEmbeddings.size() * embeddingDim;
    
    // Flatten embeddings for GPU
    std::vector<float> flatEmbeddings(totalSize);
    for (size_t i = 0; i < _indexEmbeddings.size(); ++i) {
        std::copy(
            _indexEmbeddings[i].begin(),
            _indexEmbeddings[i].end(),
            flatEmbeddings.begin() + i * embeddingDim
        );
    }
    
    // Allocate GPU memory
    _gpuIndexBuffer = GpuBuffer<float>(totalSize);
    
    // Copy to GPU
    cudaMemcpy(
        _gpuIndexBuffer.data(),
        flatEmbeddings.data(),
        totalSize * sizeof(float),
        cudaMemcpyHostToDevice
    );
}

size_t NNKnnPipeline::getIndexSize() const {
    return _indexEmbeddings.size();
}

size_t NNKnnPipeline::getEmbeddingDimension() const {
    if (_indexEmbeddings.empty()) {
        return 0;
    }
    return _indexEmbeddings[0].size();
}

void NNKnnPipeline::clearIndex() {
    _indexEmbeddings.clear();
    _indexIds.clear();
    _gpuIndexBuffer = GpuBuffer<float>();
}

} // namespace pipeline
} // namespace dsstne

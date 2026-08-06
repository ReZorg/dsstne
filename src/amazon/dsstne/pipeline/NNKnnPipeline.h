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

#ifndef NN_KNN_PIPELINE_H
#define NN_KNN_PIPELINE_H

#include <string>
#include <vector>
#include <memory>
#include <functional>

#include "EmbeddingExtractor.h"

// Forward declarations
class NNNetwork;
class NNDataSetBase;

namespace dsstne {

/**
 * @brief Configuration for the NN+KNN pipeline
 */
struct NNKnnPipelineConfig {
    // Embedding extraction settings
    std::string embeddingLayerName;     // Layer to extract embeddings from (empty = last hidden)
    bool normalizeEmbeddings;            // L2-normalize embeddings before KNN
    
    // KNN settings
    uint32_t k;                          // Number of nearest neighbors
    uint32_t numVectors;                 // Number of reference vectors in KNN index
    uint32_t vectorDim;                  // Dimension of vectors
    
    // Performance settings
    bool useGpuKnn;                      // Use GPU-accelerated KNN
    bool zeroCopyTransfer;               // Zero-copy embedding to KNN
    
    NNKnnPipelineConfig() 
        : normalizeEmbeddings(true),
          k(10),
          numVectors(0),
          vectorDim(0),
          useGpuKnn(true),
          zeroCopyTransfer(true) {}
};

/**
 * @brief KNN search result for a single query
 */
struct KnnQueryResult {
    std::vector<uint32_t> neighborIndices;  // Indices of K nearest neighbors
    std::vector<float> distances;            // Distances to K nearest neighbors
};

/**
 * @brief Combined result of NN inference + KNN search
 */
struct NNKnnResult {
    std::vector<KnnQueryResult> results;     // Results for each sample in batch
    EmbeddingResult embeddings;              // Extracted embeddings
    uint32_t batchSize;
    float inferenceTimeMs;
    float knnTimeMs;
};

/**
 * @brief Pipeline combining neural network inference with KNN lookup
 * 
 * This class provides an integrated pipeline for:
 * 1. Running neural network inference
 * 2. Extracting embeddings from a hidden layer
 * 3. Performing KNN search on the embeddings
 * 
 * Key features:
 * - Zero-copy data transfer between NN and KNN on GPU
 * - Batch processing for efficiency
 * - Support for both GPU and CPU KNN backends
 */
class NNKnnPipeline {
public:
    /**
     * @brief Constructor
     * @param network Pre-loaded neural network
     * @param config Pipeline configuration
     */
    NNKnnPipeline(NNNetwork* network, const NNKnnPipelineConfig& config);
    
    /**
     * @brief Destructor
     */
    ~NNKnnPipeline();
    
    /**
     * @brief Initialize KNN index with reference vectors
     * @param vectors Pointer to reference vector data (vectorDim * numVectors floats)
     * @param numVectors Number of reference vectors
     * @param vectorDim Dimension of each vector
     */
    void initializeKnnIndex(const float* vectors, uint32_t numVectors, uint32_t vectorDim);
    
    /**
     * @brief Initialize KNN index from a file
     * @param indexPath Path to pre-built index file
     */
    void loadKnnIndex(const std::string& indexPath);
    
    /**
     * @brief Save KNN index to a file
     * @param indexPath Path to save index
     */
    void saveKnnIndex(const std::string& indexPath);
    
    /**
     * @brief Run the full pipeline: inference -> embedding -> KNN
     * @param result Combined result
     * @return true on success
     */
    bool run(NNKnnResult& result);
    
    /**
     * @brief Run inference only (populate embeddings without KNN)
     * @param result Embedding result
     * @return true on success
     */
    bool runInferenceOnly(EmbeddingResult& result);
    
    /**
     * @brief Run KNN only on pre-extracted embeddings
     * @param embeddings Query embeddings
     * @param result KNN results
     * @return true on success
     */
    bool runKnnOnly(const EmbeddingResult& embeddings, NNKnnResult& result);
    
    /**
     * @brief Get the embedding extractor for direct access
     */
    EmbeddingExtractor& getEmbeddingExtractor() { return *_embeddingExtractor; }
    
    /**
     * @brief Get configuration
     */
    const NNKnnPipelineConfig& getConfig() const { return _config; }
    
    /**
     * @brief Update K value for KNN search
     */
    void setK(uint32_t k) { _config.k = k; }
    
    /**
     * @brief Set callback for progress reporting
     */
    void setProgressCallback(std::function<void(float)> callback) {
        _progressCallback = callback;
    }

private:
    NNNetwork* _network;
    NNKnnPipelineConfig _config;
    std::unique_ptr<EmbeddingExtractor> _embeddingExtractor;
    std::function<void(float)> _progressCallback;
    
    // KNN index data (managed externally or loaded from file)
    float* _knnIndex;
    uint32_t _knnIndexSize;
    bool _knnIndexOwned;
    
    /**
     * @brief Perform GPU-based KNN search
     */
    void gpuKnnSearch(const float* queries, uint32_t numQueries,
                      uint32_t* outIndices, float* outDistances);
    
    /**
     * @brief Perform CPU-based KNN search (fallback)
     */
    void cpuKnnSearch(const float* queries, uint32_t numQueries,
                      uint32_t* outIndices, float* outDistances);
};

} // namespace dsstne

#endif // NN_KNN_PIPELINE_H

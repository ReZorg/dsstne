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

#ifndef EMBEDDING_EXTRACTOR_H
#define EMBEDDING_EXTRACTOR_H

#include <string>
#include <vector>
#include <memory>

// Forward declarations
class NNNetwork;
class NNLayer;
template<typename T> class GpuBuffer;

namespace dsstne {

/**
 * @brief Configuration for embedding extraction
 */
struct EmbeddingConfig {
    std::string layerName;          // Name of layer to extract embeddings from
    bool normalize;                  // Whether to L2-normalize embeddings
    bool copyToHost;                 // Whether to copy embeddings to CPU memory
    
    EmbeddingConfig() : normalize(false), copyToHost(true) {}
};

/**
 * @brief Result of embedding extraction
 */
struct EmbeddingResult {
    std::vector<float> embeddings;  // CPU-side embeddings (if copyToHost=true)
    float* gpuEmbeddings;           // GPU pointer (always valid after extraction)
    uint32_t batchSize;             // Number of embeddings
    uint32_t embeddingDim;          // Dimension of each embedding
    
    EmbeddingResult() : gpuEmbeddings(nullptr), batchSize(0), embeddingDim(0) {}
    
    size_t totalSize() const { return static_cast<size_t>(batchSize) * static_cast<size_t>(embeddingDim); }
};

/**
 * @brief Extracts embeddings from hidden layers of neural networks
 * 
 * This class provides a clean interface for extracting intermediate
 * layer activations as embeddings, suitable for:
 * - KNN similarity search
 * - Downstream ML models
 * - Visualization
 * - Analysis
 */
class EmbeddingExtractor {
public:
    /**
     * @brief Constructor
     * @param network Neural network to extract embeddings from
     */
    explicit EmbeddingExtractor(NNNetwork* network);
    
    /**
     * @brief Destructor
     */
    ~EmbeddingExtractor();
    
    /**
     * @brief Extract embeddings from a specified layer
     * @param config Extraction configuration
     * @param result Output embeddings
     * @return true on success
     */
    bool extract(const EmbeddingConfig& config, EmbeddingResult& result);
    
    /**
     * @brief Extract embeddings from the last hidden layer before output
     * @param result Output embeddings
     * @return true on success
     */
    bool extractLastHiddenLayer(EmbeddingResult& result);
    
    /**
     * @brief Get list of available layers for embedding extraction
     */
    std::vector<std::string> getAvailableLayers() const;
    
    /**
     * @brief Get embedding dimension for a specific layer
     * @param layerName Name of the layer
     * @return Dimension of embeddings from that layer
     */
    uint32_t getEmbeddingDimension(const std::string& layerName) const;
    
    /**
     * @brief Check if a layer is valid for embedding extraction
     */
    bool isValidLayer(const std::string& layerName) const;
    
    /**
     * @brief Run inference to populate layer activations
     * 
     * Must be called before extract() if network state has changed
     */
    void runInference();
    
    /**
     * @brief Set whether to use zero-copy access for GPU embeddings
     */
    void setZeroCopy(bool enabled) { _useZeroCopy = enabled; }

private:
    NNNetwork* _network;
    bool _useZeroCopy;
    
    /**
     * @brief Get layer by name
     */
    NNLayer* getLayer(const std::string& layerName) const;
    
    /**
     * @brief Normalize embeddings in-place on GPU
     */
    void normalizeEmbeddings(float* embeddings, uint32_t batchSize, uint32_t dim);
};

} // namespace dsstne

#endif // EMBEDDING_EXTRACTOR_H

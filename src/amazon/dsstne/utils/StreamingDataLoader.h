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

#ifndef STREAMING_DATA_LOADER_H
#define STREAMING_DATA_LOADER_H

#include <string>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <memory>
#include <functional>

// Forward declarations
class NNDataSetBase;
struct NNDataSetDescriptor;

namespace dsstne {

/**
 * @brief Data batch for streaming inference
 */
struct DataBatch {
    std::vector<float> data;
    std::vector<uint64_t> sparseStart;
    std::vector<uint64_t> sparseEnd;
    std::vector<uint32_t> sparseIndex;
    std::vector<float> sparseValue;
    uint32_t batchSize;
    uint32_t width;
    bool isSparse;
    
    DataBatch() : batchSize(0), width(0), isSparse(false) {}
    
    void clear() {
        data.clear();
        sparseStart.clear();
        sparseEnd.clear();
        sparseIndex.clear();
        sparseValue.clear();
        batchSize = 0;
        width = 0;
        isSparse = false;
    }
    
    size_t memoryUsage() const {
        size_t bytes = data.size() * sizeof(float);
        bytes += sparseStart.size() * sizeof(uint64_t);
        bytes += sparseEnd.size() * sizeof(uint64_t);
        bytes += sparseIndex.size() * sizeof(uint32_t);
        bytes += sparseValue.size() * sizeof(float);
        return bytes;
    }
};

/**
 * @brief Streaming data loader for efficient inference
 * 
 * This class provides a streaming interface for loading data directly
 * into GPU memory without intermediate NetCDF file generation.
 * 
 * Features:
 * - In-memory data conversion (skip NetCDF write/read)
 * - Prefetch support with configurable buffer depth
 * - Async data loading pipeline
 * - Support for both dense and sparse data
 */
class StreamingDataLoader {
public:
    /**
     * @brief Constructor
     * @param batchSize Number of samples per batch
     * @param prefetchDepth Number of batches to prefetch (0 = sync mode)
     */
    StreamingDataLoader(uint32_t batchSize, uint32_t prefetchDepth = 2);
    
    /**
     * @brief Destructor - stops background threads
     */
    ~StreamingDataLoader();
    
    /**
     * @brief Initialize the loader with dataset descriptor
     * @param descriptor Dataset configuration
     */
    void initialize(const NNDataSetDescriptor& descriptor);
    
    /**
     * @brief Start the background prefetch thread
     */
    void startPrefetch();
    
    /**
     * @brief Stop the background prefetch thread
     */
    void stopPrefetch();
    
    /**
     * @brief Add dense data to the streaming queue
     * @param data Pointer to dense data array
     * @param numSamples Number of samples
     * @param width Width per sample
     */
    void addDenseData(const float* data, uint32_t numSamples, uint32_t width);
    
    /**
     * @brief Add sparse data to the streaming queue
     * @param sparseStart Start indices for each sample
     * @param sparseEnd End indices for each sample
     * @param sparseIndex Feature indices
     * @param sparseValue Feature values
     * @param numSamples Number of samples
     * @param width Maximum width
     */
    void addSparseData(const uint64_t* sparseStart, const uint64_t* sparseEnd,
                       const uint32_t* sparseIndex, const float* sparseValue,
                       uint32_t numSamples, uint32_t width);
    
    /**
     * @brief Get next batch (blocking if prefetch is enabled)
     * @param batch Output batch data
     * @return true if batch was retrieved, false if no more data
     */
    bool getNextBatch(DataBatch& batch);
    
    /**
     * @brief Check if more data is available
     */
    bool hasMoreData() const;
    
    /**
     * @brief Get number of batches ready for processing
     */
    size_t getReadyBatchCount() const;
    
    /**
     * @brief Reset the loader for new data
     */
    void reset();
    
    /**
     * @brief Set callback for batch ready notification
     */
    void setOnBatchReady(std::function<void(const DataBatch&)> callback);
    
    /**
     * @brief Get statistics about loader performance
     */
    struct Statistics {
        uint64_t batchesLoaded;
        uint64_t bytesProcessed;
        double avgLoadTimeMs;
        uint32_t currentQueueDepth;
    };
    Statistics getStatistics() const;

private:
    // Configuration
    uint32_t _batchSize;
    uint32_t _prefetchDepth;
    uint32_t _width;
    bool _isSparse;
    
    // Input data queue
    std::queue<DataBatch> _inputQueue;
    std::mutex _inputMutex;
    std::condition_variable _inputCondition;
    
    // Ready batches queue
    std::queue<DataBatch> _readyQueue;
    std::mutex _readyMutex;
    std::condition_variable _readyCondition;
    
    // Prefetch thread
    std::unique_ptr<std::thread> _prefetchThread;
    std::atomic<bool> _running;
    std::atomic<bool> _stopped;
    
    // Statistics
    std::atomic<uint64_t> _batchesLoaded;
    std::atomic<uint64_t> _bytesProcessed;
    std::atomic<double> _totalLoadTimeMs;
    
    // Callback
    std::function<void(const DataBatch&)> _onBatchReady;
    
    /**
     * @brief Background prefetch worker
     */
    void prefetchWorker();
    
    /**
     * @brief Process raw input data into a batch
     */
    void processBatch(DataBatch& batch);
};

//=============================================================================
// Inline implementations
//=============================================================================

inline StreamingDataLoader::StreamingDataLoader(uint32_t batchSize, uint32_t prefetchDepth)
    : _batchSize(batchSize),
      _prefetchDepth(prefetchDepth),
      _width(0),
      _isSparse(false),
      _running(false),
      _stopped(true),
      _batchesLoaded(0),
      _bytesProcessed(0),
      _totalLoadTimeMs(0.0) {
}

inline StreamingDataLoader::~StreamingDataLoader() {
    stopPrefetch();
}

inline bool StreamingDataLoader::hasMoreData() const {
    return !_inputQueue.empty() || !_readyQueue.empty();
}

inline size_t StreamingDataLoader::getReadyBatchCount() const {
    return _readyQueue.size();
}

inline void StreamingDataLoader::setOnBatchReady(std::function<void(const DataBatch&)> callback) {
    _onBatchReady = callback;
}

inline StreamingDataLoader::Statistics StreamingDataLoader::getStatistics() const {
    Statistics stats;
    stats.batchesLoaded = _batchesLoaded.load();
    stats.bytesProcessed = _bytesProcessed.load();
    stats.avgLoadTimeMs = (stats.batchesLoaded > 0) ? 
        _totalLoadTimeMs.load() / stats.batchesLoaded : 0.0;
    stats.currentQueueDepth = static_cast<uint32_t>(_readyQueue.size());
    return stats;
}

} // namespace dsstne

#endif // STREAMING_DATA_LOADER_H

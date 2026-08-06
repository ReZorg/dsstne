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

#include "StreamingDataLoader.h"
#include <chrono>
#include <algorithm>
#include <cstring>

namespace dsstne {

void StreamingDataLoader::initialize(const NNDataSetDescriptor& descriptor) {
    std::lock_guard<std::mutex> lock(_inputMutex);
    reset();
    // Configuration from descriptor would be applied here
}

void StreamingDataLoader::startPrefetch() {
    if (_prefetchDepth == 0) {
        return;  // Sync mode, no prefetch thread
    }
    
    if (_running.exchange(true)) {
        return;  // Already running
    }
    
    _stopped = false;
    _prefetchThread = std::make_unique<std::thread>(&StreamingDataLoader::prefetchWorker, this);
}

void StreamingDataLoader::stopPrefetch() {
    _running = false;
    
    // Wake up the prefetch thread if it's waiting
    _inputCondition.notify_all();
    _readyCondition.notify_all();
    
    if (_prefetchThread && _prefetchThread->joinable()) {
        _prefetchThread->join();
    }
    _prefetchThread.reset();
    _stopped = true;
}

void StreamingDataLoader::addDenseData(const float* data, uint32_t numSamples, uint32_t width) {
    _width = width;
    _isSparse = false;
    
    // Process data into batches
    uint32_t samplesProcessed = 0;
    while (samplesProcessed < numSamples) {
        DataBatch batch;
        batch.isSparse = false;
        batch.width = width;
        batch.batchSize = std::min(_batchSize, numSamples - samplesProcessed);
        
        // Copy data for this batch - use size_t to avoid overflow
        size_t batchDataSize = static_cast<size_t>(batch.batchSize) * static_cast<size_t>(width);
        batch.data.resize(batchDataSize);
        std::memcpy(batch.data.data(), data + (static_cast<size_t>(samplesProcessed) * static_cast<size_t>(width)), 
                    batchDataSize * sizeof(float));
        
        // Add to input queue
        {
            std::lock_guard<std::mutex> lock(_inputMutex);
            _inputQueue.push(std::move(batch));
        }
        _inputCondition.notify_one();
        
        samplesProcessed += batch.batchSize;
    }
}

void StreamingDataLoader::addSparseData(const uint64_t* sparseStart, const uint64_t* sparseEnd,
                                        const uint32_t* sparseIndex, const float* sparseValue,
                                        uint32_t numSamples, uint32_t width) {
    _width = width;
    _isSparse = true;
    
    // Process sparse data into batches
    uint32_t samplesProcessed = 0;
    while (samplesProcessed < numSamples) {
        DataBatch batch;
        batch.isSparse = true;
        batch.width = width;
        batch.batchSize = std::min(_batchSize, numSamples - samplesProcessed);
        
        // Calculate total non-zero elements in this batch
        uint64_t totalNonZeros = 0;
        for (uint32_t i = 0; i < batch.batchSize; i++) {
            uint32_t sampleIdx = samplesProcessed + i;
            totalNonZeros += sparseEnd[sampleIdx] - sparseStart[sampleIdx];
        }
        
        // Allocate space
        batch.sparseStart.resize(batch.batchSize + 1);
        batch.sparseEnd.resize(batch.batchSize);
        batch.sparseIndex.resize(totalNonZeros);
        batch.sparseValue.resize(totalNonZeros);
        
        // Copy sparse data
        uint64_t currentOffset = 0;
        for (uint32_t i = 0; i < batch.batchSize; i++) {
            uint32_t sampleIdx = samplesProcessed + i;
            uint64_t start = sparseStart[sampleIdx];
            uint64_t end = sparseEnd[sampleIdx];
            uint64_t nnz = end - start;
            
            batch.sparseStart[i] = currentOffset;
            batch.sparseEnd[i] = currentOffset + nnz;
            
            if (nnz > 0) {
                std::memcpy(batch.sparseIndex.data() + currentOffset, 
                           sparseIndex + start, nnz * sizeof(uint32_t));
                std::memcpy(batch.sparseValue.data() + currentOffset, 
                           sparseValue + start, nnz * sizeof(float));
            }
            
            currentOffset += nnz;
        }
        batch.sparseStart[batch.batchSize] = currentOffset;
        
        // Add to input queue
        {
            std::lock_guard<std::mutex> lock(_inputMutex);
            _inputQueue.push(std::move(batch));
        }
        _inputCondition.notify_one();
        
        samplesProcessed += batch.batchSize;
    }
}

bool StreamingDataLoader::getNextBatch(DataBatch& batch) {
    if (_prefetchDepth > 0) {
        // Prefetch mode - get from ready queue
        std::unique_lock<std::mutex> lock(_readyMutex);
        
        // Wait for a batch to be ready
        _readyCondition.wait(lock, [this]() {
            return !_readyQueue.empty() || !_running;
        });
        
        if (_readyQueue.empty()) {
            return false;
        }
        
        batch = std::move(_readyQueue.front());
        _readyQueue.pop();
        
        // Signal that there's room in the queue
        _inputCondition.notify_one();
        
        return true;
    } else {
        // Sync mode - process directly from input queue
        std::lock_guard<std::mutex> lock(_inputMutex);
        
        if (_inputQueue.empty()) {
            return false;
        }
        
        batch = std::move(_inputQueue.front());
        _inputQueue.pop();
        
        // Process the batch
        auto startTime = std::chrono::high_resolution_clock::now();
        processBatch(batch);
        auto endTime = std::chrono::high_resolution_clock::now();
        
        // Update statistics
        _batchesLoaded++;
        _bytesProcessed += batch.memoryUsage();
        double loadTimeMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
        _totalLoadTimeMs = _totalLoadTimeMs.load() + loadTimeMs;
        
        return true;
    }
}

void StreamingDataLoader::reset() {
    stopPrefetch();
    
    {
        std::lock_guard<std::mutex> inputLock(_inputMutex);
        while (!_inputQueue.empty()) _inputQueue.pop();
    }
    
    {
        std::lock_guard<std::mutex> readyLock(_readyMutex);
        while (!_readyQueue.empty()) _readyQueue.pop();
    }
    
    _batchesLoaded = 0;
    _bytesProcessed = 0;
    _totalLoadTimeMs = 0.0;
}

void StreamingDataLoader::prefetchWorker() {
    while (_running) {
        DataBatch batch;
        
        // Get batch from input queue
        {
            std::unique_lock<std::mutex> lock(_inputMutex);
            _inputCondition.wait(lock, [this]() {
                return !_inputQueue.empty() || !_running;
            });
            
            if (!_running && _inputQueue.empty()) {
                break;
            }
            
            if (_inputQueue.empty()) {
                continue;
            }
            
            batch = std::move(_inputQueue.front());
            _inputQueue.pop();
        }
        
        // Process the batch
        auto startTime = std::chrono::high_resolution_clock::now();
        processBatch(batch);
        auto endTime = std::chrono::high_resolution_clock::now();
        
        // Update statistics
        _batchesLoaded++;
        _bytesProcessed += batch.memoryUsage();
        double loadTimeMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
        _totalLoadTimeMs = _totalLoadTimeMs.load() + loadTimeMs;
        
        // Wait if ready queue is full
        {
            std::unique_lock<std::mutex> lock(_readyMutex);
            _readyCondition.wait(lock, [this]() {
                return _readyQueue.size() < _prefetchDepth || !_running;
            });
            
            if (!_running) {
                break;
            }
            
            _readyQueue.push(std::move(batch));
        }
        
        // Notify that batch is ready
        _readyCondition.notify_one();
        
        // Call callback if set
        if (_onBatchReady) {
            _onBatchReady(_readyQueue.back());
        }
    }
}

void StreamingDataLoader::processBatch(DataBatch& batch) {
    // This is where data transformation would happen
    // For now, data is already in the correct format
    // 
    // Future enhancements:
    // - Data normalization
    // - Feature encoding
    // - Data augmentation
    // - Type conversion
}

} // namespace dsstne

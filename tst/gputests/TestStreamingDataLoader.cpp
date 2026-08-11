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

/**
 * @file TestStreamingDataLoader.cpp
 * @brief Tests for StreamingDataLoader functionality.
 */

#include <cppunit/extensions/HelperMacros.h>
#include <cuda_runtime.h>
#include <vector>
#include <random>
#include <thread>
#include <chrono>

#include "StreamingDataLoader.h"

class TestStreamingDataLoader : public CppUnit::TestFixture {
    CPPUNIT_TEST_SUITE(TestStreamingDataLoader);
    
    // Basic functionality tests
    CPPUNIT_TEST(testDenseDataLoading);
    CPPUNIT_TEST(testSparseDataLoading);
    CPPUNIT_TEST(testMixedDataLoading);
    
    // Batch tests
    CPPUNIT_TEST(testBatchSizeMatching);
    CPPUNIT_TEST(testMultipleBatches);
    CPPUNIT_TEST(testPartialLastBatch);
    
    // Prefetch tests
    CPPUNIT_TEST(testPrefetchEnabled);
    CPPUNIT_TEST(testPrefetchDepth);
    
    // Edge cases
    CPPUNIT_TEST(testEmptyData);
    CPPUNIT_TEST(testSingleSample);
    CPPUNIT_TEST(testLargeBatch);
    
    // Error handling
    CPPUNIT_TEST(testInvalidBatchSize);
    CPPUNIT_TEST(testMismatchedDimensions);
    
    CPPUNIT_TEST_SUITE_END();

public:
    void setUp() override {
        _gen = std::mt19937(42);
        _batchSize = 32;
        _inputDim = 100;
        _numSamples = 100;
    }
    
    void tearDown() override {
    }
    
    /**
     * Test loading dense data.
     */
    void testDenseDataLoading() {
        using namespace dsstne::utils;
        
        StreamingDataLoader loader(_batchSize);
        loader.setBatchSize(_batchSize);
        
        // Generate test data
        std::vector<float> denseData = generateRandomData(_numSamples * _inputDim);
        
        // Add data to loader
        loader.addDenseData("input", denseData.data(), _numSamples, _inputDim);
        
        // Verify we can get batches
        CPPUNIT_ASSERT(loader.hasNextBatch());
        
        auto batch = loader.getNextBatch();
        CPPUNIT_ASSERT(!batch.denseData.empty());
        CPPUNIT_ASSERT_EQUAL(static_cast<size_t>(_batchSize), batch.batchSize);
    }
    
    /**
     * Test loading sparse data.
     */
    void testSparseDataLoading() {
        using namespace dsstne::utils;
        
        StreamingDataLoader loader(_batchSize);
        
        // Generate sparse data
        std::vector<uint64_t> sparseStart;
        std::vector<uint64_t> sparseEnd;
        std::vector<uint32_t> sparseIndex;
        std::vector<float> sparseData;
        
        generateSparseData(_numSamples, _inputDim, 0.1f,
                          sparseStart, sparseEnd, sparseIndex, sparseData);
        
        loader.addSparseData("sparse_input", 
                            sparseStart.data(), sparseEnd.data(),
                            sparseIndex.data(), sparseData.data(),
                            _numSamples, _inputDim);
        
        CPPUNIT_ASSERT(loader.hasNextBatch());
        
        auto batch = loader.getNextBatch();
        CPPUNIT_ASSERT(!batch.sparseData.empty());
    }
    
    /**
     * Test loading mixed dense and sparse data.
     */
    void testMixedDataLoading() {
        using namespace dsstne::utils;
        
        StreamingDataLoader loader(_batchSize);
        
        // Add dense data
        std::vector<float> denseData = generateRandomData(_numSamples * _inputDim);
        loader.addDenseData("dense", denseData.data(), _numSamples, _inputDim);
        
        // Add sparse data
        std::vector<uint64_t> sparseStart, sparseEnd;
        std::vector<uint32_t> sparseIndex;
        std::vector<float> sparseData;
        
        generateSparseData(_numSamples, _inputDim, 0.1f,
                          sparseStart, sparseEnd, sparseIndex, sparseData);
        
        loader.addSparseData("sparse", 
                            sparseStart.data(), sparseEnd.data(),
                            sparseIndex.data(), sparseData.data(),
                            _numSamples, _inputDim);
        
        auto batch = loader.getNextBatch();
        CPPUNIT_ASSERT(!batch.denseData.empty());
        CPPUNIT_ASSERT(!batch.sparseData.empty());
    }
    
    /**
     * Test batch size is correct.
     */
    void testBatchSizeMatching() {
        using namespace dsstne::utils;
        
        StreamingDataLoader loader(_batchSize);
        
        std::vector<float> data = generateRandomData(_numSamples * _inputDim);
        loader.addDenseData("input", data.data(), _numSamples, _inputDim);
        
        auto batch = loader.getNextBatch();
        CPPUNIT_ASSERT_EQUAL(static_cast<size_t>(_batchSize), batch.batchSize);
    }
    
    /**
     * Test iterating through multiple batches.
     */
    void testMultipleBatches() {
        using namespace dsstne::utils;
        
        StreamingDataLoader loader(_batchSize);
        
        std::vector<float> data = generateRandomData(_numSamples * _inputDim);
        loader.addDenseData("input", data.data(), _numSamples, _inputDim);
        
        int batchCount = 0;
        int totalSamples = 0;
        
        while (loader.hasNextBatch()) {
            auto batch = loader.getNextBatch();
            batchCount++;
            totalSamples += batch.batchSize;
        }
        
        // Should have processed all samples
        CPPUNIT_ASSERT_EQUAL(_numSamples, totalSamples);
    }
    
    /**
     * Test partial last batch handling.
     */
    void testPartialLastBatch() {
        using namespace dsstne::utils;
        
        int unevenSamples = 45;  // Not divisible by _batchSize
        StreamingDataLoader loader(_batchSize);
        
        std::vector<float> data = generateRandomData(unevenSamples * _inputDim);
        loader.addDenseData("input", data.data(), unevenSamples, _inputDim);
        
        int totalSamples = 0;
        while (loader.hasNextBatch()) {
            auto batch = loader.getNextBatch();
            totalSamples += batch.batchSize;
        }
        
        CPPUNIT_ASSERT_EQUAL(unevenSamples, totalSamples);
    }
    
    /**
     * Test prefetch mode is working.
     */
    void testPrefetchEnabled() {
        using namespace dsstne::utils;
        
        StreamingDataLoader loader(_batchSize);
        loader.setPrefetchDepth(2);
        
        std::vector<float> data = generateRandomData(_numSamples * _inputDim);
        loader.addDenseData("input", data.data(), _numSamples, _inputDim);
        
        // Start prefetching
        loader.startPrefetch();
        
        // Allow some time for prefetch
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        
        // Should still work correctly
        CPPUNIT_ASSERT(loader.hasNextBatch());
        
        loader.stopPrefetch();
    }
    
    /**
     * Test different prefetch depths.
     */
    void testPrefetchDepth() {
        using namespace dsstne::utils;
        
        std::vector<int> depths = {1, 2, 4, 8};
        
        for (int depth : depths) {
            StreamingDataLoader loader(_batchSize);
            loader.setPrefetchDepth(depth);
            
            std::vector<float> data = generateRandomData(_numSamples * _inputDim);
            loader.addDenseData("input", data.data(), _numSamples, _inputDim);
            
            int batchCount = 0;
            while (loader.hasNextBatch()) {
                auto batch = loader.getNextBatch();
                batchCount++;
            }
            
            // Should process same number regardless of prefetch depth
            int expectedBatches = (_numSamples + _batchSize - 1) / _batchSize;
            CPPUNIT_ASSERT_EQUAL(expectedBatches, batchCount);
        }
    }
    
    /**
     * Test with empty data.
     */
    void testEmptyData() {
        using namespace dsstne::utils;
        
        StreamingDataLoader loader(_batchSize);
        
        // Don't add any data
        CPPUNIT_ASSERT(!loader.hasNextBatch());
    }
    
    /**
     * Test with single sample.
     */
    void testSingleSample() {
        using namespace dsstne::utils;
        
        StreamingDataLoader loader(_batchSize);
        
        std::vector<float> data = generateRandomData(_inputDim);
        loader.addDenseData("input", data.data(), 1, _inputDim);
        
        CPPUNIT_ASSERT(loader.hasNextBatch());
        
        auto batch = loader.getNextBatch();
        CPPUNIT_ASSERT_EQUAL(static_cast<size_t>(1), batch.batchSize);
        
        CPPUNIT_ASSERT(!loader.hasNextBatch());
    }
    
    /**
     * Test with large batch size.
     */
    void testLargeBatch() {
        using namespace dsstne::utils;
        
        int largeBatch = 256;
        StreamingDataLoader loader(largeBatch);
        
        int samples = 1000;
        std::vector<float> data = generateRandomData(samples * _inputDim);
        loader.addDenseData("input", data.data(), samples, _inputDim);
        
        int totalProcessed = 0;
        while (loader.hasNextBatch()) {
            auto batch = loader.getNextBatch();
            totalProcessed += batch.batchSize;
        }
        
        CPPUNIT_ASSERT_EQUAL(samples, totalProcessed);
    }
    
    /**
     * Test invalid batch size.
     */
    void testInvalidBatchSize() {
        using namespace dsstne::utils;
        
        // Should throw or handle gracefully
        try {
            StreamingDataLoader loader(0);
            CPPUNIT_FAIL("Should have thrown for zero batch size");
        } catch (const std::invalid_argument&) {
            // Expected
        }
        
        try {
            StreamingDataLoader loader(-1);
            CPPUNIT_FAIL("Should have thrown for negative batch size");
        } catch (const std::invalid_argument&) {
            // Expected
        }
    }
    
    /**
     * Test mismatched dimensions across inputs.
     */
    void testMismatchedDimensions() {
        using namespace dsstne::utils;
        
        StreamingDataLoader loader(_batchSize);
        
        // Add data with different sample counts
        std::vector<float> data1 = generateRandomData(100 * _inputDim);
        std::vector<float> data2 = generateRandomData(50 * _inputDim);  // Different count
        
        loader.addDenseData("input1", data1.data(), 100, _inputDim);
        
        try {
            loader.addDenseData("input2", data2.data(), 50, _inputDim);
            CPPUNIT_FAIL("Should have thrown for mismatched sample counts");
        } catch (const std::invalid_argument&) {
            // Expected
        }
    }

private:
    std::mt19937 _gen;
    int _batchSize;
    int _inputDim;
    int _numSamples;
    
    /**
     * Generate random float data.
     */
    std::vector<float> generateRandomData(size_t size) {
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<float> data(size);
        for (size_t i = 0; i < size; ++i) {
            data[i] = dist(_gen);
        }
        return data;
    }
    
    /**
     * Generate sparse data for testing.
     */
    void generateSparseData(
        int numSamples, 
        int dim, 
        float sparsity,
        std::vector<uint64_t>& start,
        std::vector<uint64_t>& end,
        std::vector<uint32_t>& index,
        std::vector<float>& data) {
        
        std::uniform_real_distribution<float> valueDist(-1.0f, 1.0f);
        std::uniform_real_distribution<float> sparseDist(0.0f, 1.0f);
        
        start.resize(numSamples);
        end.resize(numSamples);
        
        uint64_t currentPos = 0;
        
        for (int sample = 0; sample < numSamples; ++sample) {
            start[sample] = currentPos;
            
            for (int d = 0; d < dim; ++d) {
                if (sparseDist(_gen) < sparsity) {
                    index.push_back(d);
                    data.push_back(valueDist(_gen));
                    currentPos++;
                }
            }
            
            end[sample] = currentPos;
        }
    }
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestStreamingDataLoader);

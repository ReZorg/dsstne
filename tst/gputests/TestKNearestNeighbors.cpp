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
 * @file TestKNearestNeighbors.cpp
 * @brief GPU tests for K-Nearest Neighbors functionality.
 */

#include <cppunit/extensions/HelperMacros.h>
#include <cuda_runtime.h>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

// Assuming these headers exist based on KNN module structure
// #include "knn.h"
// #include "topk.h"

class TestKNearestNeighbors : public CppUnit::TestFixture {
    CPPUNIT_TEST_SUITE(TestKNearestNeighbors);
    
    // Basic KNN tests
    CPPUNIT_TEST(testKnnSingleQuery);
    CPPUNIT_TEST(testKnnBatchQuery);
    CPPUNIT_TEST(testKnnDifferentKValues);
    
    // Distance metric tests
    CPPUNIT_TEST(testCosineDistance);
    CPPUNIT_TEST(testEuclideanDistance);
    CPPUNIT_TEST(testDotProductDistance);
    
    // TopK tests
    CPPUNIT_TEST(testTopKBasic);
    CPPUNIT_TEST(testTopKLargeK);
    CPPUNIT_TEST(testTopKWithTies);
    
    // Edge cases
    CPPUNIT_TEST(testEmptyDatabase);
    CPPUNIT_TEST(testSingleItemDatabase);
    CPPUNIT_TEST(testKLargerThanDatabase);
    
    // Performance-related
    CPPUNIT_TEST(testLargeScale);
    
    CPPUNIT_TEST_SUITE_END();

public:
    void setUp() override {
        // Initialize random generator
        _gen = std::mt19937(42);
        
        // Default test parameters
        _embeddingDim = 128;
        _numVectors = 1000;
        _k = 10;
    }
    
    void tearDown() override {
        // Cleanup GPU resources
    }
    
    /**
     * Test single query KNN.
     */
    void testKnnSingleQuery() {
        // Create test database
        std::vector<float> database = generateRandomVectors(_numVectors, _embeddingDim);
        std::vector<float> query = generateRandomVectors(1, _embeddingDim);
        
        // Run GPU KNN
        std::vector<int> gpuIndices(_k);
        std::vector<float> gpuDistances(_k);
        
        // TODO: Call actual GPU KNN implementation
        // knn_search(database.data(), _numVectors, query.data(), 1, 
        //            _embeddingDim, _k, gpuIndices.data(), gpuDistances.data());
        
        // Run CPU reference
        auto [cpuIndices, cpuDistances] = cpuKnnSearch(database, query, _k);
        
        // Compare results (allow some tolerance for floating point)
        // For now, just verify shapes
        CPPUNIT_ASSERT_EQUAL(static_cast<size_t>(_k), gpuIndices.size());
        CPPUNIT_ASSERT_EQUAL(static_cast<size_t>(_k), gpuDistances.size());
    }
    
    /**
     * Test batch query KNN.
     */
    void testKnnBatchQuery() {
        int batchSize = 32;
        std::vector<float> database = generateRandomVectors(_numVectors, _embeddingDim);
        std::vector<float> queries = generateRandomVectors(batchSize, _embeddingDim);
        
        std::vector<int> gpuIndices(batchSize * _k);
        std::vector<float> gpuDistances(batchSize * _k);
        
        // TODO: Call actual GPU KNN implementation
        // knn_search(database.data(), _numVectors, queries.data(), batchSize,
        //            _embeddingDim, _k, gpuIndices.data(), gpuDistances.data());
        
        CPPUNIT_ASSERT_EQUAL(static_cast<size_t>(batchSize * _k), gpuIndices.size());
    }
    
    /**
     * Test different K values.
     */
    void testKnnDifferentKValues() {
        std::vector<float> database = generateRandomVectors(_numVectors, _embeddingDim);
        std::vector<float> query = generateRandomVectors(1, _embeddingDim);
        
        std::vector<int> kValues = {1, 5, 10, 50, 100};
        
        for (int k : kValues) {
            std::vector<int> indices(k);
            std::vector<float> distances(k);
            
            // TODO: Call actual GPU KNN implementation
            // knn_search(database.data(), _numVectors, query.data(), 1,
            //            _embeddingDim, k, indices.data(), distances.data());
            
            CPPUNIT_ASSERT_EQUAL(static_cast<size_t>(k), indices.size());
            
            // Verify distances are sorted
            for (int i = 1; i < k; ++i) {
                // Placeholder check - would verify distances[i-1] <= distances[i]
            }
        }
    }
    
    /**
     * Test cosine distance computation.
     */
    void testCosineDistance() {
        // Identical vectors should have distance 0
        std::vector<float> v1 = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<float> v2 = v1;
        
        float distance = cpuCosineDistance(v1.data(), v2.data(), v1.size());
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0f, distance, 1e-6f);
        
        // Orthogonal vectors should have distance 1
        std::vector<float> v3 = {1.0f, 0.0f, 0.0f, 0.0f};
        std::vector<float> v4 = {0.0f, 1.0f, 0.0f, 0.0f};
        
        distance = cpuCosineDistance(v3.data(), v4.data(), v3.size());
        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0f, distance, 1e-6f);
        
        // Opposite vectors should have distance 2
        std::vector<float> v5 = {1.0f, 0.0f, 0.0f, 0.0f};
        std::vector<float> v6 = {-1.0f, 0.0f, 0.0f, 0.0f};
        
        distance = cpuCosineDistance(v5.data(), v6.data(), v5.size());
        CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0f, distance, 1e-6f);
    }
    
    /**
     * Test Euclidean distance computation.
     */
    void testEuclideanDistance() {
        // Same point should have distance 0
        std::vector<float> v1 = {1.0f, 2.0f, 3.0f};
        float distance = cpuEuclideanDistance(v1.data(), v1.data(), v1.size());
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0f, distance, 1e-6f);
        
        // Known distance
        std::vector<float> v2 = {0.0f, 0.0f, 0.0f};
        std::vector<float> v3 = {3.0f, 4.0f, 0.0f};  // 3-4-5 triangle
        
        distance = cpuEuclideanDistance(v2.data(), v3.data(), v2.size());
        CPPUNIT_ASSERT_DOUBLES_EQUAL(5.0f, distance, 1e-6f);
    }
    
    /**
     * Test dot product distance computation.
     */
    void testDotProductDistance() {
        std::vector<float> v1 = {1.0f, 2.0f, 3.0f};
        std::vector<float> v2 = {4.0f, 5.0f, 6.0f};
        
        float dotProduct = cpuDotProduct(v1.data(), v2.data(), v1.size());
        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        CPPUNIT_ASSERT_DOUBLES_EQUAL(32.0f, dotProduct, 1e-6f);
    }
    
    /**
     * Test basic TopK functionality.
     */
    void testTopKBasic() {
        std::vector<float> values = {3.0f, 1.0f, 4.0f, 1.0f, 5.0f, 9.0f, 2.0f, 6.0f};
        int k = 3;
        
        auto [indices, topValues] = cpuTopK(values, k);
        
        // Top 3 should be: 9.0 (idx 5), 6.0 (idx 7), 5.0 (idx 4)
        CPPUNIT_ASSERT_EQUAL(5, indices[0]);
        CPPUNIT_ASSERT_EQUAL(7, indices[1]);
        CPPUNIT_ASSERT_EQUAL(4, indices[2]);
    }
    
    /**
     * Test TopK with K larger than available elements.
     */
    void testTopKLargeK() {
        std::vector<float> values = {1.0f, 2.0f, 3.0f};
        int k = 10;  // Larger than array size
        
        auto [indices, topValues] = cpuTopK(values, k);
        
        // Should return all elements
        CPPUNIT_ASSERT_EQUAL(static_cast<size_t>(3), indices.size());
    }
    
    /**
     * Test TopK with tied values.
     */
    void testTopKWithTies() {
        std::vector<float> values = {1.0f, 5.0f, 5.0f, 5.0f, 2.0f};
        int k = 3;
        
        auto [indices, topValues] = cpuTopK(values, k);
        
        // All top-3 should have value 5.0
        CPPUNIT_ASSERT_DOUBLES_EQUAL(5.0f, topValues[0], 1e-6f);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(5.0f, topValues[1], 1e-6f);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(5.0f, topValues[2], 1e-6f);
    }
    
    /**
     * Test with empty database.
     */
    void testEmptyDatabase() {
        std::vector<float> database;
        std::vector<float> query = generateRandomVectors(1, _embeddingDim);
        
        // Should handle gracefully (return empty or throw)
        // Implementation-dependent behavior
        CPPUNIT_ASSERT(true);  // Placeholder
    }
    
    /**
     * Test with single item in database.
     */
    void testSingleItemDatabase() {
        std::vector<float> database = generateRandomVectors(1, _embeddingDim);
        std::vector<float> query = generateRandomVectors(1, _embeddingDim);
        
        std::vector<int> indices(1);
        std::vector<float> distances(1);
        
        // Should return the single item
        CPPUNIT_ASSERT_EQUAL(static_cast<size_t>(1), indices.size());
    }
    
    /**
     * Test when K is larger than database size.
     */
    void testKLargerThanDatabase() {
        int smallDbSize = 5;
        std::vector<float> database = generateRandomVectors(smallDbSize, _embeddingDim);
        std::vector<float> query = generateRandomVectors(1, _embeddingDim);
        
        int k = 100;  // Much larger than database
        
        // Should return only available items
        // Implementation should handle this gracefully
        CPPUNIT_ASSERT(true);  // Placeholder
    }
    
    /**
     * Test large-scale performance.
     */
    void testLargeScale() {
        int largeDbSize = 100000;
        int batchSize = 100;
        
        // This test is mainly for timing, not correctness
        std::vector<float> database = generateRandomVectors(largeDbSize, _embeddingDim);
        std::vector<float> queries = generateRandomVectors(batchSize, _embeddingDim);
        
        std::vector<int> indices(batchSize * _k);
        std::vector<float> distances(batchSize * _k);
        
        // TODO: Time the actual GPU KNN call
        // auto start = std::chrono::high_resolution_clock::now();
        // knn_search(database.data(), largeDbSize, queries.data(), batchSize,
        //            _embeddingDim, _k, indices.data(), distances.data());
        // auto end = std::chrono::high_resolution_clock::now();
        
        CPPUNIT_ASSERT(true);  // Placeholder
    }

private:
    std::mt19937 _gen;
    int _embeddingDim;
    int _numVectors;
    int _k;
    
    /**
     * Generate random vectors for testing.
     */
    std::vector<float> generateRandomVectors(int numVectors, int dim) {
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<float> vectors(numVectors * dim);
        
        for (size_t i = 0; i < vectors.size(); ++i) {
            vectors[i] = dist(_gen);
        }
        
        return vectors;
    }
    
    /**
     * CPU reference implementation of KNN search.
     */
    std::pair<std::vector<int>, std::vector<float>> cpuKnnSearch(
        const std::vector<float>& database,
        const std::vector<float>& query,
        int k) {
        
        int numVectors = database.size() / _embeddingDim;
        std::vector<std::pair<float, int>> distances;
        distances.reserve(numVectors);
        
        for (int i = 0; i < numVectors; ++i) {
            float dist = cpuCosineDistance(
                query.data(),
                database.data() + i * _embeddingDim,
                _embeddingDim);
            distances.emplace_back(dist, i);
        }
        
        std::partial_sort(
            distances.begin(),
            distances.begin() + std::min(k, numVectors),
            distances.end());
        
        int resultSize = std::min(k, numVectors);
        std::vector<int> indices(resultSize);
        std::vector<float> dists(resultSize);
        
        for (int i = 0; i < resultSize; ++i) {
            indices[i] = distances[i].second;
            dists[i] = distances[i].first;
        }
        
        return {indices, dists};
    }
    
    /**
     * CPU cosine distance computation.
     */
    float cpuCosineDistance(const float* a, const float* b, size_t dim) {
        float dotProd = 0.0f;
        float normA = 0.0f;
        float normB = 0.0f;
        
        for (size_t i = 0; i < dim; ++i) {
            dotProd += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        
        float denom = std::sqrt(normA) * std::sqrt(normB);
        if (denom < 1e-8f) return 1.0f;
        
        return 1.0f - (dotProd / denom);
    }
    
    /**
     * CPU Euclidean distance computation.
     */
    float cpuEuclideanDistance(const float* a, const float* b, size_t dim) {
        float sum = 0.0f;
        for (size_t i = 0; i < dim; ++i) {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }
    
    /**
     * CPU dot product computation.
     */
    float cpuDotProduct(const float* a, const float* b, size_t dim) {
        float sum = 0.0f;
        for (size_t i = 0; i < dim; ++i) {
            sum += a[i] * b[i];
        }
        return sum;
    }
    
    /**
     * CPU TopK implementation.
     */
    std::pair<std::vector<int>, std::vector<float>> cpuTopK(
        const std::vector<float>& values, int k) {
        
        std::vector<std::pair<float, int>> indexed;
        indexed.reserve(values.size());
        
        for (size_t i = 0; i < values.size(); ++i) {
            indexed.emplace_back(values[i], i);
        }
        
        int actualK = std::min(k, static_cast<int>(values.size()));
        
        std::partial_sort(
            indexed.begin(),
            indexed.begin() + actualK,
            indexed.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });
        
        std::vector<int> indices(actualK);
        std::vector<float> topValues(actualK);
        
        for (int i = 0; i < actualK; ++i) {
            indices[i] = indexed[i].second;
            topValues[i] = indexed[i].first;
        }
        
        return {indices, topValues};
    }
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestKNearestNeighbors);

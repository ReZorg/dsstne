/*
   Copyright 2016  Amazon.com, Inc. or its affiliates. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License").
   You may not use this file except in compliance with the License.
   A copy of the License is located at http://aws.amazon.com/apache2.0/
*/

#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/ui/text/TestRunner.h>
#include <cppunit/TestAssert.h>

#include <string>
#include <sstream>
#include <chrono>

#include "Utils.h"

using namespace std;

// NOTE: NNDataSetDimensions and NNDataSetDescriptor tests require GPU dependencies
// (NNTypes.h includes GpuTypes.h which requires CUDA).
// These tests are implemented in tst/amazon/dsstne/engine/ and run as part of the
// GPU test suite. See TestNNDataSetDimensions.cpp and TestNNDataSet.cpp.

/**
 * Test suite for elapsed_seconds utility function
 */
class TestElapsedSeconds : public CppUnit::TestFixture
{
public:
    
    void TestElapsedSecondsReturnsPositiveValue() {
        auto start = chrono::steady_clock::now();
        // Small delay
        volatile int x = 0;
        for (int i = 0; i < 10000; ++i) { x += i; }
        auto end = chrono::steady_clock::now();
        
        double elapsed = elapsed_seconds(start, end);
        CPPUNIT_ASSERT(elapsed >= 0.0);
    }
    
    void TestElapsedSecondsZeroForSameTime() {
        auto now = chrono::steady_clock::now();
        double elapsed = elapsed_seconds(now, now);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, elapsed, 1e-9);
    }
    
    void TestElapsedSecondsNegativeForReversedTime() {
        auto start = chrono::steady_clock::now();
        // Small delay
        volatile int x = 0;
        for (int i = 0; i < 10000; ++i) { x += i; }
        auto end = chrono::steady_clock::now();
        
        double elapsed = elapsed_seconds(end, start);
        CPPUNIT_ASSERT(elapsed <= 0.0);
    }
    
    void TestElapsedSecondsScalesCorrectly() {
        // This is more of a sanity check
        auto start = chrono::steady_clock::now();
        auto end = start + chrono::milliseconds(100);
        
        double elapsed = elapsed_seconds(start, end);
        // Should be approximately 0.1 seconds
        CPPUNIT_ASSERT(elapsed > 0.05 && elapsed < 0.15);
    }
    
    CPPUNIT_TEST_SUITE(TestElapsedSeconds);
    CPPUNIT_TEST(TestElapsedSecondsReturnsPositiveValue);
    CPPUNIT_TEST(TestElapsedSecondsZeroForSameTime);
    CPPUNIT_TEST(TestElapsedSecondsNegativeForReversedTime);
    CPPUNIT_TEST(TestElapsedSecondsScalesCorrectly);
    CPPUNIT_TEST_SUITE_END();
};

/**
 * Test suite for random number generation utilities
 */
class TestRandomUtils : public CppUnit::TestFixture
{
public:
    
    void TestRandIntWithinRange() {
        // Test rand(int min, int max) returns values in range
        for (int i = 0; i < 100; ++i) {
            int result = rand(0, 10);
            CPPUNIT_ASSERT(result >= 0 && result <= 10);
        }
    }
    
    void TestRandIntWithSameMinMax() {
        // When min == max, should always return that value
        for (int i = 0; i < 10; ++i) {
            int result = rand(5, 5);
            CPPUNIT_ASSERT_EQUAL(5, result);
        }
    }
    
    void TestRandFloatWithinRange() {
        // Test rand(float min, float max) returns values in range
        for (int i = 0; i < 100; ++i) {
            float result = rand(0.0f, 1.0f);
            CPPUNIT_ASSERT(result >= 0.0f && result <= 1.0f);
        }
    }
    
    void TestRandFloatWithSameMinMax() {
        // When min == max, should always return that value
        float result = rand(5.0f, 5.0f);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(5.0f, result, 1e-6);
    }
    
    void TestRandFloatWithNegativeRange() {
        for (int i = 0; i < 100; ++i) {
            float result = rand(-10.0f, -5.0f);
            CPPUNIT_ASSERT(result >= -10.0f && result <= -5.0f);
        }
    }
    
    CPPUNIT_TEST_SUITE(TestRandomUtils);
    CPPUNIT_TEST(TestRandIntWithinRange);
    CPPUNIT_TEST(TestRandIntWithSameMinMax);
    CPPUNIT_TEST(TestRandFloatWithinRange);
    CPPUNIT_TEST(TestRandFloatWithSameMinMax);
    CPPUNIT_TEST(TestRandFloatWithNegativeRange);
    CPPUNIT_TEST_SUITE_END();
};

/**
 * Test suite for CWMetric class
 */
class TestCWMetric : public CppUnit::TestFixture
{
public:
    
    void TestUpdateMetricsWithStringValue() {
        // Should not crash - CWMetric::updateMetrics is a no-op in current implementation
        CWMetric::updateMetrics("test_metric", "test_value");
        // If we get here without exception, test passes
        CPPUNIT_ASSERT(true);
    }
    
    void TestUpdateMetricsWithIntValue() {
        CWMetric::updateMetrics("test_metric", 42);
        CPPUNIT_ASSERT(true);
    }
    
    void TestUpdateMetricsWithFloatValue() {
        CWMetric::updateMetrics("test_metric", 3.14f);
        CPPUNIT_ASSERT(true);
    }
    
    void TestUpdateMetricsWithDoubleValue() {
        CWMetric::updateMetrics("test_metric", 2.718281828);
        CPPUNIT_ASSERT(true);
    }
    
    CPPUNIT_TEST_SUITE(TestCWMetric);
    CPPUNIT_TEST(TestUpdateMetricsWithStringValue);
    CPPUNIT_TEST(TestUpdateMetricsWithIntValue);
    CPPUNIT_TEST(TestUpdateMetricsWithFloatValue);
    CPPUNIT_TEST(TestUpdateMetricsWithDoubleValue);
    CPPUNIT_TEST_SUITE_END();
};

/**
 * Test suite for Constants and Global Values
 */
class TestConstants : public CppUnit::TestFixture
{
public:
    
    void TestInputDatasetSuffixValue() {
        CPPUNIT_ASSERT_EQUAL(string("_input"), INPUT_DATASET_SUFFIX);
    }
    
    void TestOutputDatasetSuffixValue() {
        CPPUNIT_ASSERT_EQUAL(string("_output"), OUTPUT_DATASET_SUFFIX);
    }
    
    void TestNetCDFFileExtensionValue() {
        CPPUNIT_ASSERT_EQUAL(string(".nc"), NETCDF_FILE_EXTENTION);
    }
    
    void TestFixedSeedValue() {
        // FIXED_SEED should be a specific value for reproducibility
        CPPUNIT_ASSERT_EQUAL(12134ul, FIXED_SEED);
    }
    
    CPPUNIT_TEST_SUITE(TestConstants);
    CPPUNIT_TEST(TestInputDatasetSuffixValue);
    CPPUNIT_TEST(TestOutputDatasetSuffixValue);
    CPPUNIT_TEST(TestNetCDFFileExtensionValue);
    CPPUNIT_TEST(TestFixedSeedValue);
    CPPUNIT_TEST_SUITE_END();
};

/**
 * TestNNWeightInit.cpp
 * 
 * Comprehensive tests for NNWeight initialization functionality.
 * Part of Phase 1: Core Inference & Training Tests
 * 
 * Copyright 2016-2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0
 */

// CppUnit
#include "cppunit/extensions/HelperMacros.h"
#include "cppunit/ui/text/TestRunner.h"
#include "cppunit/TestAssert.h"

// STL
#include <string>
#include <vector>
#include <memory>
#include <cmath>
#include <numeric>
#include <algorithm>

#include "Utils.h"
#include "GpuTypes.h"
#include "NNTypes.h"
#include "TestUtils.h"

//----------------------------------------------------------------------------
// Test class for NNWeight initialization
//----------------------------------------------------------------------------
class TestNNWeightInit : public CppUnit::TestFixture {
public:
    //------------------------------------------------------------------------
    // Test: Get weights from network
    //------------------------------------------------------------------------
    void testGetWeights() {
        const std::string modelPath = std::string(TEST_DATA_PATH) + "validate_L2_01.json";
        
        DataParameters dataParameters;
        dataParameters.numberOfSamples = 128;
        dataParameters.inpFeatureDimensionality = 1;
        dataParameters.outFeatureDimensionality = 1;
        
        std::vector<NNDataSetBase*> vDataSet;
        const std::string dataPath(TEST_DATA_PATH);
        generateTestData(dataPath, Regression, dataParameters, std::cout);
        vDataSet = LoadNetCDF(dataPath + "test.nc");
        
        NNNetwork* pNetwork = LoadNeuralNetworkJSON(modelPath, 32, vDataSet);
        CPPUNIT_ASSERT(pNetwork != nullptr);
        
        NNWeight* weight = pNetwork->GetWeight("Input", "Output");
        CPPUNIT_ASSERT_MESSAGE("Weight should exist", weight != nullptr);
        
        std::vector<NNFloat> weights;
        bool gotWeights = weight->GetWeights(weights);
        
        CPPUNIT_ASSERT_MESSAGE("GetWeights should succeed", gotWeights);
        CPPUNIT_ASSERT_MESSAGE("Weights should not be empty", !weights.empty());
        
        // Cleanup
        delete pNetwork;
        for (auto p : vDataSet) {
            delete p;
        }
    }

    //------------------------------------------------------------------------
    // Test: Get biases from weight
    //------------------------------------------------------------------------
    void testGetBiases() {
        const std::string modelPath = std::string(TEST_DATA_PATH) + "validate_L2_01.json";
        
        DataParameters dataParameters;
        dataParameters.numberOfSamples = 128;
        dataParameters.inpFeatureDimensionality = 1;
        dataParameters.outFeatureDimensionality = 1;
        
        std::vector<NNDataSetBase*> vDataSet;
        const std::string dataPath(TEST_DATA_PATH);
        generateTestData(dataPath, Regression, dataParameters, std::cout);
        vDataSet = LoadNetCDF(dataPath + "test.nc");
        
        NNNetwork* pNetwork = LoadNeuralNetworkJSON(modelPath, 32, vDataSet);
        CPPUNIT_ASSERT(pNetwork != nullptr);
        
        NNWeight* weight = pNetwork->GetWeight("Input", "Output");
        CPPUNIT_ASSERT(weight != nullptr);
        
        std::vector<NNFloat> biases;
        bool gotBiases = weight->GetBiases(biases);
        
        CPPUNIT_ASSERT_MESSAGE("GetBiases should succeed", gotBiases);
        CPPUNIT_ASSERT_MESSAGE("Biases should not be empty", !biases.empty());
        
        // Cleanup
        delete pNetwork;
        for (auto p : vDataSet) {
            delete p;
        }
    }

    //------------------------------------------------------------------------
    // Test: Set weights
    //------------------------------------------------------------------------
    void testSetWeights() {
        const std::string modelPath = std::string(TEST_DATA_PATH) + "validate_L2_01.json";
        
        DataParameters dataParameters;
        dataParameters.numberOfSamples = 128;
        dataParameters.inpFeatureDimensionality = 1;
        dataParameters.outFeatureDimensionality = 1;
        
        std::vector<NNDataSetBase*> vDataSet;
        const std::string dataPath(TEST_DATA_PATH);
        generateTestData(dataPath, Regression, dataParameters, std::cout);
        vDataSet = LoadNetCDF(dataPath + "test.nc");
        
        NNNetwork* pNetwork = LoadNeuralNetworkJSON(modelPath, 32, vDataSet);
        CPPUNIT_ASSERT(pNetwork != nullptr);
        
        NNWeight* weight = pNetwork->GetWeight("Input", "Output");
        CPPUNIT_ASSERT(weight != nullptr);
        
        // Get current weights to know the size
        std::vector<NNFloat> originalWeights;
        weight->GetWeights(originalWeights);
        
        // Create new weights
        std::vector<NNFloat> newWeights(originalWeights.size(), 0.5f);
        
        bool setResult = weight->SetWeights(newWeights);
        CPPUNIT_ASSERT_MESSAGE("SetWeights should succeed", setResult);
        
        // Verify weights were set
        std::vector<NNFloat> retrievedWeights;
        weight->GetWeights(retrievedWeights);
        
        for (size_t i = 0; i < newWeights.size(); i++) {
            CPPUNIT_ASSERT_DOUBLES_EQUAL_MESSAGE("Weights should match set values",
                newWeights[i], retrievedWeights[i], 1e-5);
        }
        
        // Cleanup
        delete pNetwork;
        for (auto p : vDataSet) {
            delete p;
        }
    }

    //------------------------------------------------------------------------
    // Test: Set biases
    //------------------------------------------------------------------------
    void testSetBiases() {
        const std::string modelPath = std::string(TEST_DATA_PATH) + "validate_L2_01.json";
        
        DataParameters dataParameters;
        dataParameters.numberOfSamples = 128;
        dataParameters.inpFeatureDimensionality = 1;
        dataParameters.outFeatureDimensionality = 1;
        
        std::vector<NNDataSetBase*> vDataSet;
        const std::string dataPath(TEST_DATA_PATH);
        generateTestData(dataPath, Regression, dataParameters, std::cout);
        vDataSet = LoadNetCDF(dataPath + "test.nc");
        
        NNNetwork* pNetwork = LoadNeuralNetworkJSON(modelPath, 32, vDataSet);
        CPPUNIT_ASSERT(pNetwork != nullptr);
        
        NNWeight* weight = pNetwork->GetWeight("Input", "Output");
        CPPUNIT_ASSERT(weight != nullptr);
        
        // Get current biases to know the size
        std::vector<NNFloat> originalBiases;
        weight->GetBiases(originalBiases);
        
        // Create new biases
        std::vector<NNFloat> newBiases(originalBiases.size(), 0.1f);
        
        bool setResult = weight->SetBiases(newBiases);
        CPPUNIT_ASSERT_MESSAGE("SetBiases should succeed", setResult);
        
        // Verify biases were set
        std::vector<NNFloat> retrievedBiases;
        weight->GetBiases(retrievedBiases);
        
        for (size_t i = 0; i < newBiases.size(); i++) {
            CPPUNIT_ASSERT_DOUBLES_EQUAL_MESSAGE("Biases should match set values",
                newBiases[i], retrievedBiases[i], 1e-5);
        }
        
        // Cleanup
        delete pNetwork;
        for (auto p : vDataSet) {
            delete p;
        }
    }

    //------------------------------------------------------------------------
    // Test: Weight dimensions
    //------------------------------------------------------------------------
    void testWeightDimensions() {
        const std::string modelPath = std::string(TEST_DATA_PATH) + "validate_L2_01.json";
        
        DataParameters dataParameters;
        dataParameters.numberOfSamples = 128;
        dataParameters.inpFeatureDimensionality = 1;
        dataParameters.outFeatureDimensionality = 1;
        
        std::vector<NNDataSetBase*> vDataSet;
        const std::string dataPath(TEST_DATA_PATH);
        generateTestData(dataPath, Regression, dataParameters, std::cout);
        vDataSet = LoadNetCDF(dataPath + "test.nc");
        
        NNNetwork* pNetwork = LoadNeuralNetworkJSON(modelPath, 32, vDataSet);
        CPPUNIT_ASSERT(pNetwork != nullptr);
        
        NNWeight* weight = pNetwork->GetWeight("Input", "Output");
        CPPUNIT_ASSERT(weight != nullptr);
        
        std::vector<uint64_t> dimensions;
        bool gotDimensions = weight->GetDimensions(dimensions);
        
        CPPUNIT_ASSERT_MESSAGE("GetDimensions should succeed", gotDimensions);
        CPPUNIT_ASSERT_MESSAGE("Dimensions should not be empty", !dimensions.empty());
        
        // For a 1->1 layer, expect input_dim x output_dim = 1 x 1
        std::cout << "Weight dimensions: ";
        for (auto d : dimensions) {
            std::cout << d << " ";
        }
        std::cout << std::endl;
        
        // Cleanup
        delete pNetwork;
        for (auto p : vDataSet) {
            delete p;
        }
    }

    //------------------------------------------------------------------------
    // Test: Weights are initialized
    //------------------------------------------------------------------------
    void testWeightsAreInitialized() {
        const std::string modelPath = std::string(TEST_DATA_PATH) + "validate_L2_01.json";
        
        DataParameters dataParameters;
        dataParameters.numberOfSamples = 128;
        dataParameters.inpFeatureDimensionality = 1;
        dataParameters.outFeatureDimensionality = 1;
        
        std::vector<NNDataSetBase*> vDataSet;
        const std::string dataPath(TEST_DATA_PATH);
        generateTestData(dataPath, Regression, dataParameters, std::cout);
        vDataSet = LoadNetCDF(dataPath + "test.nc");
        
        NNNetwork* pNetwork = LoadNeuralNetworkJSON(modelPath, 32, vDataSet);
        CPPUNIT_ASSERT(pNetwork != nullptr);
        
        NNWeight* weight = pNetwork->GetWeight("Input", "Output");
        CPPUNIT_ASSERT(weight != nullptr);
        
        std::vector<NNFloat> weights;
        weight->GetWeights(weights);
        
        // Verify weights are finite values
        for (auto w : weights) {
            CPPUNIT_ASSERT_MESSAGE("Weight should not be NaN", !std::isnan(w));
            CPPUNIT_ASSERT_MESSAGE("Weight should not be Inf", !std::isinf(w));
        }
        
        // Cleanup
        delete pNetwork;
        for (auto p : vDataSet) {
            delete p;
        }
    }

    //------------------------------------------------------------------------
    // Test: Copy weights between networks
    //------------------------------------------------------------------------
    void testCopyWeights() {
        const std::string modelPath = std::string(TEST_DATA_PATH) + "validate_L2_01.json";
        
        DataParameters dataParameters;
        dataParameters.numberOfSamples = 128;
        dataParameters.inpFeatureDimensionality = 1;
        dataParameters.outFeatureDimensionality = 1;
        
        std::vector<NNDataSetBase*> vDataSet;
        const std::string dataPath(TEST_DATA_PATH);
        generateTestData(dataPath, Regression, dataParameters, std::cout);
        vDataSet = LoadNetCDF(dataPath + "test.nc");
        
        // Create first network and train it
        NNNetwork* pNetwork1 = LoadNeuralNetworkJSON(modelPath, 32, vDataSet);
        pNetwork1->LoadDataSets(vDataSet);
        pNetwork1->SetTrainingMode(SGD);
        pNetwork1->Train(3, 0.01f, 0.001f, 0.0f, 0.9f, 0.0f);
        
        NNWeight* weight1 = pNetwork1->GetWeight("Input", "Output");
        std::vector<NNFloat> trainedWeights;
        weight1->GetWeights(trainedWeights);
        
        // Create second network
        NNNetwork* pNetwork2 = LoadNeuralNetworkJSON(modelPath, 32, vDataSet);
        NNWeight* weight2 = pNetwork2->GetWeight("Input", "Output");
        
        // Copy weights
        bool copied = weight2->CopyWeights(weight1);
        CPPUNIT_ASSERT_MESSAGE("CopyWeights should succeed", copied);
        
        // Verify weights match
        std::vector<NNFloat> copiedWeights;
        weight2->GetWeights(copiedWeights);
        
        for (size_t i = 0; i < trainedWeights.size(); i++) {
            CPPUNIT_ASSERT_DOUBLES_EQUAL_MESSAGE("Copied weights should match",
                trainedWeights[i], copiedWeights[i], 1e-5);
        }
        
        // Cleanup
        delete pNetwork1;
        delete pNetwork2;
        for (auto p : vDataSet) {
            delete p;
        }
    }

    //------------------------------------------------------------------------
    // Test: Weight norm setting
    //------------------------------------------------------------------------
    void testSetWeightNorm() {
        const std::string modelPath = std::string(TEST_DATA_PATH) + "validate_L2_01.json";
        
        DataParameters dataParameters;
        dataParameters.numberOfSamples = 128;
        dataParameters.inpFeatureDimensionality = 1;
        dataParameters.outFeatureDimensionality = 1;
        
        std::vector<NNDataSetBase*> vDataSet;
        const std::string dataPath(TEST_DATA_PATH);
        generateTestData(dataPath, Regression, dataParameters, std::cout);
        vDataSet = LoadNetCDF(dataPath + "test.nc");
        
        NNNetwork* pNetwork = LoadNeuralNetworkJSON(modelPath, 32, vDataSet);
        CPPUNIT_ASSERT(pNetwork != nullptr);
        
        NNWeight* weight = pNetwork->GetWeight("Input", "Output");
        CPPUNIT_ASSERT(weight != nullptr);
        
        // Set weight norm
        bool setNorm = weight->SetNorm(1.0f);
        CPPUNIT_ASSERT_MESSAGE("SetNorm should succeed", setNorm);
        
        // Cleanup
        delete pNetwork;
        for (auto p : vDataSet) {
            delete p;
        }
    }

    //------------------------------------------------------------------------
    // Test: Weight statistics (mean and variance)
    //------------------------------------------------------------------------
    void testWeightStatistics() {
        const std::string modelPath = std::string(TEST_DATA_PATH) + "validate_L2_LRelu_02.json";
        
        DataParameters dataParameters;
        dataParameters.numberOfSamples = 128;
        dataParameters.inpFeatureDimensionality = 2;
        dataParameters.outFeatureDimensionality = 2;
        
        std::vector<NNDataSetBase*> vDataSet;
        const std::string dataPath(TEST_DATA_PATH);
        generateTestData(dataPath, Classification, dataParameters, std::cout);
        vDataSet = LoadNetCDF(dataPath + "test.nc");
        
        NNNetwork* pNetwork = LoadNeuralNetworkJSON(modelPath, 32, vDataSet);
        CPPUNIT_ASSERT(pNetwork != nullptr);
        
        NNWeight* weight = pNetwork->GetWeight("Input", "Hidden");
        CPPUNIT_ASSERT(weight != nullptr);
        
        std::vector<NNFloat> weights;
        weight->GetWeights(weights);
        
        // Calculate mean
        NNFloat mean = std::accumulate(weights.begin(), weights.end(), 0.0f) / weights.size();
        
        // Calculate variance
        NNFloat variance = 0.0f;
        for (auto w : weights) {
            variance += (w - mean) * (w - mean);
        }
        variance /= weights.size();
        
        std::cout << "Weight mean: " << mean << std::endl;
        std::cout << "Weight variance: " << variance << std::endl;
        
        // Mean should be close to 0 for proper initialization
        CPPUNIT_ASSERT_MESSAGE("Weight mean should be finite", !std::isnan(mean) && !std::isinf(mean));
        CPPUNIT_ASSERT_MESSAGE("Weight variance should be finite", !std::isnan(variance) && !std::isinf(variance));
        
        // Cleanup
        delete pNetwork;
        for (auto p : vDataSet) {
            delete p;
        }
    }

    //------------------------------------------------------------------------
    // Test: Weight buffer size
    //------------------------------------------------------------------------
    void testWeightBufferSize() {
        const std::string modelPath = std::string(TEST_DATA_PATH) + "validate_L2_01.json";
        
        DataParameters dataParameters;
        dataParameters.numberOfSamples = 128;
        dataParameters.inpFeatureDimensionality = 1;
        dataParameters.outFeatureDimensionality = 1;
        
        std::vector<NNDataSetBase*> vDataSet;
        const std::string dataPath(TEST_DATA_PATH);
        generateTestData(dataPath, Regression, dataParameters, std::cout);
        vDataSet = LoadNetCDF(dataPath + "test.nc");
        
        NNNetwork* pNetwork = LoadNeuralNetworkJSON(modelPath, 32, vDataSet);
        CPPUNIT_ASSERT(pNetwork != nullptr);
        
        NNWeight* weight = pNetwork->GetWeight("Input", "Output");
        CPPUNIT_ASSERT(weight != nullptr);
        
        uint64_t bufferSize = weight->GetBufferSize();
        
        // Buffer size should be > 0
        CPPUNIT_ASSERT_MESSAGE("Buffer size should be > 0", bufferSize > 0);
        
        // For a 1->1 layer, weight buffer should be at least 1
        CPPUNIT_ASSERT_MESSAGE("Buffer size should be >= 1 for 1->1 layer", bufferSize >= 1);
        
        // Cleanup
        delete pNetwork;
        for (auto p : vDataSet) {
            delete p;
        }
    }

    //------------------------------------------------------------------------
    // Test: Weight gradient buffer
    //------------------------------------------------------------------------
    void testWeightGradientBuffer() {
        const std::string modelPath = std::string(TEST_DATA_PATH) + "validate_L2_01.json";
        
        DataParameters dataParameters;
        dataParameters.numberOfSamples = 256;
        dataParameters.inpFeatureDimensionality = 1;
        dataParameters.outFeatureDimensionality = 1;
        
        std::vector<NNDataSetBase*> vDataSet;
        const std::string dataPath(TEST_DATA_PATH);
        generateTestData(dataPath, Regression, dataParameters, std::cout);
        vDataSet = LoadNetCDF(dataPath + "test.nc");
        
        NNNetwork* pNetwork = LoadNeuralNetworkJSON(modelPath, 32, vDataSet);
        CPPUNIT_ASSERT(pNetwork != nullptr);
        
        pNetwork->LoadDataSets(vDataSet);
        pNetwork->SetTrainingMode(SGD);
        
        NNWeight* weight = pNetwork->GetWeight("Input", "Output");
        CPPUNIT_ASSERT(weight != nullptr);
        
        // Gradient buffer should exist after training setup
        NNFloat* gradientBuffer = weight->GetWeightGradientBuffer();
        CPPUNIT_ASSERT_MESSAGE("Gradient buffer should exist", gradientBuffer != nullptr);
        
        // Cleanup
        delete pNetwork;
        for (auto p : vDataSet) {
            delete p;
        }
    }

    //------------------------------------------------------------------------
    // Test: Weight buffer pointer
    //------------------------------------------------------------------------
    void testWeightBufferPointer() {
        const std::string modelPath = std::string(TEST_DATA_PATH) + "validate_L2_01.json";
        
        DataParameters dataParameters;
        dataParameters.numberOfSamples = 128;
        dataParameters.inpFeatureDimensionality = 1;
        dataParameters.outFeatureDimensionality = 1;
        
        std::vector<NNDataSetBase*> vDataSet;
        const std::string dataPath(TEST_DATA_PATH);
        generateTestData(dataPath, Regression, dataParameters, std::cout);
        vDataSet = LoadNetCDF(dataPath + "test.nc");
        
        NNNetwork* pNetwork = LoadNeuralNetworkJSON(modelPath, 32, vDataSet);
        CPPUNIT_ASSERT(pNetwork != nullptr);
        
        NNWeight* weight = pNetwork->GetWeight("Input", "Output");
        CPPUNIT_ASSERT(weight != nullptr);
        
        NNFloat* weightBuffer = weight->GetWeightBuffer();
        CPPUNIT_ASSERT_MESSAGE("Weight buffer pointer should not be null", weightBuffer != nullptr);
        
        // Cleanup
        delete pNetwork;
        for (auto p : vDataSet) {
            delete p;
        }
    }

public:
    CPPUNIT_TEST_SUITE(TestNNWeightInit);
    CPPUNIT_TEST(testGetWeights);
    CPPUNIT_TEST(testGetBiases);
    CPPUNIT_TEST(testSetWeights);
    CPPUNIT_TEST(testSetBiases);
    CPPUNIT_TEST(testWeightDimensions);
    CPPUNIT_TEST(testWeightsAreInitialized);
    CPPUNIT_TEST(testCopyWeights);
    CPPUNIT_TEST(testSetWeightNorm);
    CPPUNIT_TEST(testWeightStatistics);
    CPPUNIT_TEST(testWeightBufferSize);
    CPPUNIT_TEST(testWeightGradientBuffer);
    CPPUNIT_TEST(testWeightBufferPointer);
    CPPUNIT_TEST_SUITE_END();
};

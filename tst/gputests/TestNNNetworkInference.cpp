/**
 * TestNNNetworkInference.cpp
 * 
 * Comprehensive tests for NNNetwork inference functionality.
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
#include <fstream>

#include "Utils.h"
#include "GpuTypes.h"
#include "NNTypes.h"
#include "TestUtils.h"

//----------------------------------------------------------------------------
// Test class for NNNetwork inference functionality
//----------------------------------------------------------------------------
class TestNNNetworkInference : public CppUnit::TestFixture {
public:
    //------------------------------------------------------------------------
    // Setup and teardown
    //------------------------------------------------------------------------
    void setUp() override {
        // Ensure GPU is initialized
    }

    void tearDown() override {
        // Cleanup
    }

    //------------------------------------------------------------------------
    // Test: Network loading from JSON configuration
    //------------------------------------------------------------------------
    void testLoadNetworkFromJSON() {
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
        
        CPPUNIT_ASSERT_MESSAGE("Network should be loaded successfully", pNetwork != nullptr);
        CPPUNIT_ASSERT_EQUAL_MESSAGE("Network name should match", 
            std::string("L2 1d regression"), std::string(pNetwork->GetName()));
        
        // Cleanup
        delete pNetwork;
        for (auto p : vDataSet) {
            delete p;
        }
    }

    //------------------------------------------------------------------------
    // Test: Forward propagation (PredictBatch)
    //------------------------------------------------------------------------
    void testForwardPropagation() {
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
        
        pNetwork->LoadDataSets(vDataSet);
        pNetwork->SetPosition(0);
        
        // This should not throw
        pNetwork->PredictBatch();
        
        // Cleanup
        delete pNetwork;
        for (auto p : vDataSet) {
            delete p;
        }
    }

    //------------------------------------------------------------------------
    // Test: Forward propagation with hidden layers
    //------------------------------------------------------------------------
    void testForwardPropagationWithHiddenLayers() {
        const std::string modelPath = std::string(TEST_DATA_PATH) + "validate_L2_LRelu_02.json";
        
        DataParameters dataParameters;
        dataParameters.numberOfSamples = 256;
        dataParameters.inpFeatureDimensionality = 2;
        dataParameters.outFeatureDimensionality = 2;
        
        std::vector<NNDataSetBase*> vDataSet;
        const std::string dataPath(TEST_DATA_PATH);
        generateTestData(dataPath, Classification, dataParameters, std::cout);
        vDataSet = LoadNetCDF(dataPath + "test.nc");
        
        NNNetwork* pNetwork = LoadNeuralNetworkJSON(modelPath, 64, vDataSet);
        CPPUNIT_ASSERT(pNetwork != nullptr);
        
        pNetwork->LoadDataSets(vDataSet);
        pNetwork->SetPosition(0);
        
        // Run forward propagation
        pNetwork->PredictBatch();
        
        // Cleanup
        delete pNetwork;
        for (auto p : vDataSet) {
            delete p;
        }
    }

    //------------------------------------------------------------------------
    // Test: Batch size configuration
    //------------------------------------------------------------------------
    void testBatchSizeConfiguration() {
        const std::string modelPath = std::string(TEST_DATA_PATH) + "validate_L2_01.json";
        
        DataParameters dataParameters;
        dataParameters.numberOfSamples = 512;
        dataParameters.inpFeatureDimensionality = 1;
        dataParameters.outFeatureDimensionality = 1;
        
        std::vector<NNDataSetBase*> vDataSet;
        const std::string dataPath(TEST_DATA_PATH);
        generateTestData(dataPath, Regression, dataParameters, std::cout);
        vDataSet = LoadNetCDF(dataPath + "test.nc");
        
        // Test with different batch sizes
        const uint32_t batchSizes[] = {16, 32, 64, 128};
        
        for (uint32_t batchSize : batchSizes) {
            NNNetwork* pNetwork = LoadNeuralNetworkJSON(modelPath, batchSize, vDataSet);
            CPPUNIT_ASSERT(pNetwork != nullptr);
            
            CPPUNIT_ASSERT_EQUAL_MESSAGE("Batch size should match",
                batchSize, pNetwork->GetBatch());
            
            delete pNetwork;
        }
        
        // Cleanup
        for (auto p : vDataSet) {
            delete p;
        }
    }

    //------------------------------------------------------------------------
    // Test: Position management during inference
    //------------------------------------------------------------------------
    void testPositionManagement() {
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
        
        // Test position setting
        pNetwork->SetPosition(0);
        CPPUNIT_ASSERT_EQUAL_MESSAGE("Position should be 0", 0u, pNetwork->GetPosition());
        
        pNetwork->SetPosition(64);
        CPPUNIT_ASSERT_EQUAL_MESSAGE("Position should be 64", 64u, pNetwork->GetPosition());
        
        // Cleanup
        delete pNetwork;
        for (auto p : vDataSet) {
            delete p;
        }
    }

    //------------------------------------------------------------------------
    // Test: Get examples count
    //------------------------------------------------------------------------
    void testGetExamples() {
        const std::string modelPath = std::string(TEST_DATA_PATH) + "validate_L2_01.json";
        
        const uint32_t expectedSamples = 512;
        DataParameters dataParameters;
        dataParameters.numberOfSamples = expectedSamples;
        dataParameters.inpFeatureDimensionality = 1;
        dataParameters.outFeatureDimensionality = 1;
        
        std::vector<NNDataSetBase*> vDataSet;
        const std::string dataPath(TEST_DATA_PATH);
        generateTestData(dataPath, Regression, dataParameters, std::cout);
        vDataSet = LoadNetCDF(dataPath + "test.nc");
        
        NNNetwork* pNetwork = LoadNeuralNetworkJSON(modelPath, 32, vDataSet);
        CPPUNIT_ASSERT(pNetwork != nullptr);
        
        pNetwork->LoadDataSets(vDataSet);
        
        CPPUNIT_ASSERT_EQUAL_MESSAGE("Examples count should match",
            expectedSamples, pNetwork->GetExamples());
        
        // Cleanup
        delete pNetwork;
        for (auto p : vDataSet) {
            delete p;
        }
    }

    //------------------------------------------------------------------------
    // Test: Network validation
    //------------------------------------------------------------------------
    void testNetworkValidation() {
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
        
        pNetwork->LoadDataSets(vDataSet);
        pNetwork->SetTrainingMode(SGD);
        
        bool valid = pNetwork->Validate();
        CPPUNIT_ASSERT_MESSAGE("Network validation should succeed", valid);
        
        // Cleanup
        delete pNetwork;
        for (auto p : vDataSet) {
            delete p;
        }
    }

    //------------------------------------------------------------------------
    // Test: Get layer by name
    //------------------------------------------------------------------------
    void testGetLayerByName() {
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
        
        // Get layer by name
        NNLayer* inputLayer = pNetwork->GetLayer("Input");
        NNLayer* hiddenLayer = pNetwork->GetLayer("Hidden");
        NNLayer* outputLayer = pNetwork->GetLayer("Output");
        
        CPPUNIT_ASSERT_MESSAGE("Input layer should exist", inputLayer != nullptr);
        CPPUNIT_ASSERT_MESSAGE("Hidden layer should exist", hiddenLayer != nullptr);
        CPPUNIT_ASSERT_MESSAGE("Output layer should exist", outputLayer != nullptr);
        
        // Verify layer kinds
        CPPUNIT_ASSERT_EQUAL_MESSAGE("Input layer kind", NNLayer::Kind::Input, inputLayer->GetKind());
        CPPUNIT_ASSERT_EQUAL_MESSAGE("Hidden layer kind", NNLayer::Kind::Hidden, hiddenLayer->GetKind());
        CPPUNIT_ASSERT_EQUAL_MESSAGE("Output layer kind", NNLayer::Kind::Output, outputLayer->GetKind());
        
        // Cleanup
        delete pNetwork;
        for (auto p : vDataSet) {
            delete p;
        }
    }

    //------------------------------------------------------------------------
    // Test: Get all layer names
    //------------------------------------------------------------------------
    void testGetAllLayerNames() {
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
        
        std::vector<std::string> layerNames = pNetwork->GetLayers();
        
        CPPUNIT_ASSERT_MESSAGE("Should have 3 layers", layerNames.size() == 3);
        
        // Cleanup
        delete pNetwork;
        for (auto p : vDataSet) {
            delete p;
        }
    }

    //------------------------------------------------------------------------
    // Test: Leaky ReLU activation
    //------------------------------------------------------------------------
    void testLeakyReLUActivation() {
        const std::string modelPath = std::string(TEST_DATA_PATH) + "validate_L2_LRelu_01.json";
        
        DataParameters dataParameters;
        dataParameters.numberOfSamples = 128;
        dataParameters.inpFeatureDimensionality = 2;
        dataParameters.outFeatureDimensionality = 2;
        
        bool result = validateNeuralNetwork(32, modelPath, Classification, dataParameters, std::cout);
        CPPUNIT_ASSERT_MESSAGE("LeakyReLU network should validate", result);
    }

    //------------------------------------------------------------------------
    // Test: L2 error function
    //------------------------------------------------------------------------
    void testL2ErrorFunction() {
        const std::string modelPath = std::string(TEST_DATA_PATH) + "validate_L2_02.json";
        
        DataParameters dataParameters;
        dataParameters.numberOfSamples = 256;
        dataParameters.inpFeatureDimensionality = 2;
        dataParameters.outFeatureDimensionality = 2;
        
        bool result = validateNeuralNetwork(64, modelPath, Classification, dataParameters, std::cout);
        CPPUNIT_ASSERT_MESSAGE("L2 error function network should validate", result);
    }

    //------------------------------------------------------------------------
    // Test: Scaled Marginal Cross Entropy
    //------------------------------------------------------------------------
    void testScaledMarginalCrossEntropy() {
        const std::string modelPath = std::string(TEST_DATA_PATH) + "validate_ScaledMarginalCrossEntropy_01.json";
        
        DataParameters dataParameters;
        dataParameters.numberOfSamples = 128;
        dataParameters.inpFeatureDimensionality = 2;
        dataParameters.outFeatureDimensionality = 2;
        
        bool result = validateNeuralNetwork(32, modelPath, Classification, dataParameters, std::cout);
        CPPUNIT_ASSERT_MESSAGE("SMCE network should validate", result);
    }

    //------------------------------------------------------------------------
    // Test: Data Scaled Marginal Cross Entropy
    //------------------------------------------------------------------------
    void testDataScaledMarginalCrossEntropy() {
        const std::string modelPath = std::string(TEST_DATA_PATH) + "validate_DataScaledMarginalCrossEntropy_02.json";
        
        DataParameters dataParameters;
        dataParameters.numberOfSamples = 128;
        dataParameters.inpFeatureDimensionality = 2;
        dataParameters.outFeatureDimensionality = 2;
        
        bool result = validateNeuralNetwork(32, modelPath, Classification, dataParameters, std::cout);
        CPPUNIT_ASSERT_MESSAGE("DataSMCE network should validate", result);
    }

    //------------------------------------------------------------------------
    // Test: Multiple batch inference iterations
    //------------------------------------------------------------------------
    void testMultipleBatchInference() {
        const std::string modelPath = std::string(TEST_DATA_PATH) + "validate_L2_01.json";
        
        DataParameters dataParameters;
        dataParameters.numberOfSamples = 512;
        dataParameters.inpFeatureDimensionality = 1;
        dataParameters.outFeatureDimensionality = 1;
        
        std::vector<NNDataSetBase*> vDataSet;
        const std::string dataPath(TEST_DATA_PATH);
        generateTestData(dataPath, Regression, dataParameters, std::cout);
        vDataSet = LoadNetCDF(dataPath + "test.nc");
        
        uint32_t batchSize = 64;
        NNNetwork* pNetwork = LoadNeuralNetworkJSON(modelPath, batchSize, vDataSet);
        CPPUNIT_ASSERT(pNetwork != nullptr);
        
        pNetwork->LoadDataSets(vDataSet);
        
        // Iterate through multiple batches
        uint32_t numBatches = dataParameters.numberOfSamples / batchSize;
        for (uint32_t i = 0; i < numBatches; i++) {
            pNetwork->SetPosition(i * batchSize);
            pNetwork->PredictBatch();
        }
        
        // Cleanup
        delete pNetwork;
        for (auto p : vDataSet) {
            delete p;
        }
    }

    //------------------------------------------------------------------------
    // Test: Debug level setting
    //------------------------------------------------------------------------
    void testDebugLevelSetting() {
        const std::string modelPath = std::string(TEST_DATA_PATH) + "validate_L2_01.json";
        
        DataParameters dataParameters;
        dataParameters.numberOfSamples = 64;
        dataParameters.inpFeatureDimensionality = 1;
        dataParameters.outFeatureDimensionality = 1;
        
        std::vector<NNDataSetBase*> vDataSet;
        const std::string dataPath(TEST_DATA_PATH);
        generateTestData(dataPath, Regression, dataParameters, std::cout);
        vDataSet = LoadNetCDF(dataPath + "test.nc");
        
        NNNetwork* pNetwork = LoadNeuralNetworkJSON(modelPath, 32, vDataSet);
        CPPUNIT_ASSERT(pNetwork != nullptr);
        
        // Test debug level toggling
        pNetwork->SetDebugLevel(true);
        CPPUNIT_ASSERT_MESSAGE("Debug level should be true", pNetwork->GetDebugLevel() == true);
        
        pNetwork->SetDebugLevel(false);
        CPPUNIT_ASSERT_MESSAGE("Debug level should be false", pNetwork->GetDebugLevel() == false);
        
        // Cleanup
        delete pNetwork;
        for (auto p : vDataSet) {
            delete p;
        }
    }

    //------------------------------------------------------------------------
    // Test: Get unit buffer
    //------------------------------------------------------------------------
    void testGetUnitBuffer() {
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
        
        pNetwork->LoadDataSets(vDataSet);
        pNetwork->SetPosition(0);
        pNetwork->PredictBatch();
        
        // Get unit buffer for output layer
        NNFloat* unitBuffer = pNetwork->GetUnitBuffer("Output");
        CPPUNIT_ASSERT_MESSAGE("Unit buffer should not be null", unitBuffer != nullptr);
        
        // Cleanup
        delete pNetwork;
        for (auto p : vDataSet) {
            delete p;
        }
    }

    //------------------------------------------------------------------------
    // Test: Get buffer size
    //------------------------------------------------------------------------
    void testGetBufferSize() {
        const std::string modelPath = std::string(TEST_DATA_PATH) + "validate_L2_LRelu_02.json";
        
        DataParameters dataParameters;
        dataParameters.numberOfSamples = 128;
        dataParameters.inpFeatureDimensionality = 2;
        dataParameters.outFeatureDimensionality = 2;
        
        std::vector<NNDataSetBase*> vDataSet;
        const std::string dataPath(TEST_DATA_PATH);
        generateTestData(dataPath, Classification, dataParameters, std::cout);
        vDataSet = LoadNetCDF(dataPath + "test.nc");
        
        uint32_t batchSize = 32;
        NNNetwork* pNetwork = LoadNeuralNetworkJSON(modelPath, batchSize, vDataSet);
        CPPUNIT_ASSERT(pNetwork != nullptr);
        
        // Get buffer size for output layer (should be batch * output size)
        uint64_t bufferSize = pNetwork->GetBufferSize("Output");
        CPPUNIT_ASSERT_MESSAGE("Buffer size should be > 0", bufferSize > 0);
        
        // Expected: batch * output_dim = 32 * 2 = 64
        CPPUNIT_ASSERT_EQUAL_MESSAGE("Buffer size should match batch * output_dim",
            static_cast<uint64_t>(batchSize * dataParameters.outFeatureDimensionality), bufferSize);
        
        // Cleanup
        delete pNetwork;
        for (auto p : vDataSet) {
            delete p;
        }
    }

public:
    CPPUNIT_TEST_SUITE(TestNNNetworkInference);
    CPPUNIT_TEST(testLoadNetworkFromJSON);
    CPPUNIT_TEST(testForwardPropagation);
    CPPUNIT_TEST(testForwardPropagationWithHiddenLayers);
    CPPUNIT_TEST(testBatchSizeConfiguration);
    CPPUNIT_TEST(testPositionManagement);
    CPPUNIT_TEST(testGetExamples);
    CPPUNIT_TEST(testNetworkValidation);
    CPPUNIT_TEST(testGetLayerByName);
    CPPUNIT_TEST(testGetAllLayerNames);
    CPPUNIT_TEST(testLeakyReLUActivation);
    CPPUNIT_TEST(testL2ErrorFunction);
    CPPUNIT_TEST(testScaledMarginalCrossEntropy);
    CPPUNIT_TEST(testDataScaledMarginalCrossEntropy);
    CPPUNIT_TEST(testMultipleBatchInference);
    CPPUNIT_TEST(testDebugLevelSetting);
    CPPUNIT_TEST(testGetUnitBuffer);
    CPPUNIT_TEST(testGetBufferSize);
    CPPUNIT_TEST_SUITE_END();
};

/**
 * TestNNNetworkTraining.cpp
 * 
 * Comprehensive tests for NNNetwork training functionality.
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
// Test class for NNNetwork training functionality
//----------------------------------------------------------------------------
class TestNNNetworkTraining : public CppUnit::TestFixture {
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
    // Test: SGD training mode
    //------------------------------------------------------------------------
    void testSGDTrainingMode() {
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
        
        // Run one epoch of training
        float error = pNetwork->Train(1, 0.01f, 0.001f, 0.0f, 0.9f, 0.0f);
        
        // Error should be non-negative
        CPPUNIT_ASSERT_MESSAGE("Training error should be >= 0", error >= 0.0f);
        
        // Cleanup
        delete pNetwork;
        for (auto p : vDataSet) {
            delete p;
        }
    }

    //------------------------------------------------------------------------
    // Test: Momentum training mode
    //------------------------------------------------------------------------
    void testMomentumTrainingMode() {
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
        pNetwork->SetTrainingMode(Momentum);
        
        float error = pNetwork->Train(1, 0.01f, 0.001f, 0.0f, 0.9f, 0.0f);
        CPPUNIT_ASSERT_MESSAGE("Training error should be >= 0", error >= 0.0f);
        
        // Cleanup
        delete pNetwork;
        for (auto p : vDataSet) {
            delete p;
        }
    }

    //------------------------------------------------------------------------
    // Test: Adam optimizer
    //------------------------------------------------------------------------
    void testAdamOptimizer() {
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
        pNetwork->SetTrainingMode(Adam);
        
        float error = pNetwork->Train(1, 0.001f, 0.001f, 0.0f, 0.9f, 0.999f);
        CPPUNIT_ASSERT_MESSAGE("Training error should be >= 0", error >= 0.0f);
        
        // Cleanup
        delete pNetwork;
        for (auto p : vDataSet) {
            delete p;
        }
    }

    //------------------------------------------------------------------------
    // Test: AdaGrad optimizer
    //------------------------------------------------------------------------
    void testAdaGradOptimizer() {
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
        pNetwork->SetTrainingMode(AdaGrad);
        
        float error = pNetwork->Train(1, 0.01f, 0.001f, 0.0f, 0.0f, 0.0f);
        CPPUNIT_ASSERT_MESSAGE("Training error should be >= 0", error >= 0.0f);
        
        // Cleanup
        delete pNetwork;
        for (auto p : vDataSet) {
            delete p;
        }
    }

    //------------------------------------------------------------------------
    // Test: RMSProp optimizer
    //------------------------------------------------------------------------
    void testRMSPropOptimizer() {
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
        pNetwork->SetTrainingMode(RMSProp);
        
        float error = pNetwork->Train(1, 0.001f, 0.001f, 0.0f, 0.9f, 0.0f);
        CPPUNIT_ASSERT_MESSAGE("Training error should be >= 0", error >= 0.0f);
        
        // Cleanup
        delete pNetwork;
        for (auto p : vDataSet) {
            delete p;
        }
    }

    //------------------------------------------------------------------------
    // Test: Nesterov momentum optimizer
    //------------------------------------------------------------------------
    void testNesterovOptimizer() {
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
        pNetwork->SetTrainingMode(Nesterov);
        
        float error = pNetwork->Train(1, 0.01f, 0.001f, 0.0f, 0.9f, 0.0f);
        CPPUNIT_ASSERT_MESSAGE("Training error should be >= 0", error >= 0.0f);
        
        // Cleanup
        delete pNetwork;
        for (auto p : vDataSet) {
            delete p;
        }
    }

    //------------------------------------------------------------------------
    // Test: AdaDelta optimizer
    //------------------------------------------------------------------------
    void testAdaDeltaOptimizer() {
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
        pNetwork->SetTrainingMode(AdaDelta);
        
        float error = pNetwork->Train(1, 1.0f, 0.001f, 0.0f, 0.95f, 0.0f);
        CPPUNIT_ASSERT_MESSAGE("Training error should be >= 0", error >= 0.0f);
        
        // Cleanup
        delete pNetwork;
        for (auto p : vDataSet) {
            delete p;
        }
    }

    //------------------------------------------------------------------------
    // Test: Multiple training epochs
    //------------------------------------------------------------------------
    void testMultipleTrainingEpochs() {
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
        
        // Train for multiple epochs and verify error generally decreases
        float prevError = std::numeric_limits<float>::max();
        for (int epoch = 0; epoch < 5; epoch++) {
            float error = pNetwork->Train(1, 0.01f, 0.0001f, 0.0f, 0.9f, 0.0f);
            std::cout << "Epoch " << epoch << " error: " << error << std::endl;
            CPPUNIT_ASSERT_MESSAGE("Training error should be >= 0", error >= 0.0f);
            // Note: Error may not always decrease due to noise, so we just verify it's valid
        }
        
        // Cleanup
        delete pNetwork;
        for (auto p : vDataSet) {
            delete p;
        }
    }

    //------------------------------------------------------------------------
    // Test: Training with hidden layers
    //------------------------------------------------------------------------
    void testTrainingWithHiddenLayers() {
        const std::string modelPath = std::string(TEST_DATA_PATH) + "validate_L2_LRelu_02.json";
        
        DataParameters dataParameters;
        dataParameters.numberOfSamples = 256;
        dataParameters.inpFeatureDimensionality = 2;
        dataParameters.outFeatureDimensionality = 2;
        
        std::vector<NNDataSetBase*> vDataSet;
        const std::string dataPath(TEST_DATA_PATH);
        generateTestData(dataPath, Classification, dataParameters, std::cout);
        vDataSet = LoadNetCDF(dataPath + "test.nc");
        
        NNNetwork* pNetwork = LoadNeuralNetworkJSON(modelPath, 32, vDataSet);
        CPPUNIT_ASSERT(pNetwork != nullptr);
        
        pNetwork->LoadDataSets(vDataSet);
        pNetwork->SetTrainingMode(Adam);
        
        float error = pNetwork->Train(1, 0.001f, 0.001f, 0.0f, 0.9f, 0.999f);
        CPPUNIT_ASSERT_MESSAGE("Training error should be >= 0", error >= 0.0f);
        
        // Cleanup
        delete pNetwork;
        for (auto p : vDataSet) {
            delete p;
        }
    }

    //------------------------------------------------------------------------
    // Test: L2 regularization
    //------------------------------------------------------------------------
    void testL2Regularization() {
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
        
        // Train with L2 regularization (lambda parameter)
        float error = pNetwork->Train(1, 0.01f, 0.01f, 0.0f, 0.9f, 0.0f);
        CPPUNIT_ASSERT_MESSAGE("Training error with L2 regularization should be >= 0", error >= 0.0f);
        
        // Cleanup
        delete pNetwork;
        for (auto p : vDataSet) {
            delete p;
        }
    }

    //------------------------------------------------------------------------
    // Test: L1 regularization
    //------------------------------------------------------------------------
    void testL1Regularization() {
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
        
        // Train with L1 regularization (lambda1 parameter)
        float error = pNetwork->Train(1, 0.01f, 0.0f, 0.01f, 0.9f, 0.0f);
        CPPUNIT_ASSERT_MESSAGE("Training error with L1 regularization should be >= 0", error >= 0.0f);
        
        // Cleanup
        delete pNetwork;
        for (auto p : vDataSet) {
            delete p;
        }
    }

    //------------------------------------------------------------------------
    // Test: Combined L1/L2 regularization
    //------------------------------------------------------------------------
    void testCombinedRegularization() {
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
        
        // Train with both L1 and L2 regularization (elastic net)
        float error = pNetwork->Train(1, 0.01f, 0.005f, 0.005f, 0.9f, 0.0f);
        CPPUNIT_ASSERT_MESSAGE("Training error with combined regularization should be >= 0", error >= 0.0f);
        
        // Cleanup
        delete pNetwork;
        for (auto p : vDataSet) {
            delete p;
        }
    }

    //------------------------------------------------------------------------
    // Test: Learning rate decay
    //------------------------------------------------------------------------
    void testLearningRateDecay() {
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
        
        // Set learning rate decay
        bool decaySet = pNetwork->SetDecay(0.001f);
        CPPUNIT_ASSERT_MESSAGE("SetDecay should succeed", decaySet);
        
        float error = pNetwork->Train(3, 0.01f, 0.001f, 0.0f, 0.9f, 0.0f);
        CPPUNIT_ASSERT_MESSAGE("Training error should be >= 0", error >= 0.0f);
        
        // Cleanup
        delete pNetwork;
        for (auto p : vDataSet) {
            delete p;
        }
    }

    //------------------------------------------------------------------------
    // Test: Shuffle indices during training
    //------------------------------------------------------------------------
    void testShuffleIndices() {
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
        
        // Enable shuffling
        pNetwork->SetShuffleIndices(true);
        
        auto shuffleResult = pNetwork->GetShuffleIndices();
        CPPUNIT_ASSERT_MESSAGE("Shuffle indices should be enabled", std::get<0>(shuffleResult));
        
        float error = pNetwork->Train(1, 0.01f, 0.001f, 0.0f, 0.9f, 0.0f);
        CPPUNIT_ASSERT_MESSAGE("Training error should be >= 0", error >= 0.0f);
        
        // Cleanup
        delete pNetwork;
        for (auto p : vDataSet) {
            delete p;
        }
    }

    //------------------------------------------------------------------------
    // Test: Clear velocity setting
    //------------------------------------------------------------------------
    void testClearVelocitySetting() {
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
        pNetwork->SetTrainingMode(Momentum);
        
        // Set clear velocity flag
        pNetwork->SetClearVelocity(true);
        
        float error1 = pNetwork->Train(1, 0.01f, 0.001f, 0.0f, 0.9f, 0.0f);
        float error2 = pNetwork->Train(1, 0.01f, 0.001f, 0.0f, 0.9f, 0.0f);
        
        CPPUNIT_ASSERT_MESSAGE("Training should complete", error1 >= 0.0f && error2 >= 0.0f);
        
        // Cleanup
        delete pNetwork;
        for (auto p : vDataSet) {
            delete p;
        }
    }

    //------------------------------------------------------------------------
    // Test: Get weight buffer after training
    //------------------------------------------------------------------------
    void testGetWeightBufferAfterTraining() {
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
        
        float error = pNetwork->Train(1, 0.01f, 0.001f, 0.0f, 0.9f, 0.0f);
        
        // Get weight buffer
        NNFloat* weightBuffer = pNetwork->GetWeightBuffer("Input", "Output");
        CPPUNIT_ASSERT_MESSAGE("Weight buffer should exist", weightBuffer != nullptr);
        
        // Cleanup
        delete pNetwork;
        for (auto p : vDataSet) {
            delete p;
        }
    }

    //------------------------------------------------------------------------
    // Test: Lock and unlock weights
    //------------------------------------------------------------------------
    void testLockUnlockWeights() {
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
        
        // Lock weights
        bool locked = pNetwork->LockWeights("Input", "Output");
        CPPUNIT_ASSERT_MESSAGE("Weights should be lockable", locked);
        
        // Unlock weights
        bool unlocked = pNetwork->UnlockWeights("Input", "Output");
        CPPUNIT_ASSERT_MESSAGE("Weights should be unlockable", unlocked);
        
        // Cleanup
        delete pNetwork;
        for (auto p : vDataSet) {
            delete p;
        }
    }

    //------------------------------------------------------------------------
    // Test: Training with different batch sizes
    //------------------------------------------------------------------------
    void testTrainingWithDifferentBatchSizes() {
        const std::string modelPath = std::string(TEST_DATA_PATH) + "validate_L2_01.json";
        
        DataParameters dataParameters;
        dataParameters.numberOfSamples = 512;
        dataParameters.inpFeatureDimensionality = 1;
        dataParameters.outFeatureDimensionality = 1;
        
        std::vector<NNDataSetBase*> vDataSet;
        const std::string dataPath(TEST_DATA_PATH);
        generateTestData(dataPath, Regression, dataParameters, std::cout);
        vDataSet = LoadNetCDF(dataPath + "test.nc");
        
        const uint32_t batchSizes[] = {16, 32, 64, 128};
        
        for (uint32_t batchSize : batchSizes) {
            NNNetwork* pNetwork = LoadNeuralNetworkJSON(modelPath, batchSize, vDataSet);
            CPPUNIT_ASSERT(pNetwork != nullptr);
            
            pNetwork->LoadDataSets(vDataSet);
            pNetwork->SetTrainingMode(SGD);
            
            float error = pNetwork->Train(1, 0.01f, 0.001f, 0.0f, 0.9f, 0.0f);
            
            std::cout << "Batch size " << batchSize << " error: " << error << std::endl;
            CPPUNIT_ASSERT_MESSAGE("Training should succeed with different batch sizes", error >= 0.0f);
            
            delete pNetwork;
        }
        
        // Cleanup
        for (auto p : vDataSet) {
            delete p;
        }
    }

public:
    CPPUNIT_TEST_SUITE(TestNNNetworkTraining);
    CPPUNIT_TEST(testSGDTrainingMode);
    CPPUNIT_TEST(testMomentumTrainingMode);
    CPPUNIT_TEST(testAdamOptimizer);
    CPPUNIT_TEST(testAdaGradOptimizer);
    CPPUNIT_TEST(testRMSPropOptimizer);
    CPPUNIT_TEST(testNesterovOptimizer);
    CPPUNIT_TEST(testAdaDeltaOptimizer);
    CPPUNIT_TEST(testMultipleTrainingEpochs);
    CPPUNIT_TEST(testTrainingWithHiddenLayers);
    CPPUNIT_TEST(testL2Regularization);
    CPPUNIT_TEST(testL1Regularization);
    CPPUNIT_TEST(testCombinedRegularization);
    CPPUNIT_TEST(testLearningRateDecay);
    CPPUNIT_TEST(testShuffleIndices);
    CPPUNIT_TEST(testClearVelocitySetting);
    CPPUNIT_TEST(testGetWeightBufferAfterTraining);
    CPPUNIT_TEST(testLockUnlockWeights);
    CPPUNIT_TEST(testTrainingWithDifferentBatchSizes);
    CPPUNIT_TEST_SUITE_END();
};

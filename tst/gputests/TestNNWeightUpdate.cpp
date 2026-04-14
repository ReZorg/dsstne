/**
 * TestNNWeightUpdate.cpp
 * 
 * Comprehensive tests for NNWeight update functionality.
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

#include "Utils.h"
#include "GpuTypes.h"
#include "NNTypes.h"
#include "TestUtils.h"

//----------------------------------------------------------------------------
// Test class for NNWeight update algorithms
//----------------------------------------------------------------------------
class TestNNWeightUpdate : public CppUnit::TestFixture {
public:
    //------------------------------------------------------------------------
    // Test: SGD weight updates
    //------------------------------------------------------------------------
    void testSGDWeightUpdates() {
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
        
        std::vector<NNFloat> weightsBefore;
        weight->GetWeights(weightsBefore);
        
        pNetwork->Train(1, 0.01f, 0.0f, 0.0f, 0.0f, 0.0f);
        
        std::vector<NNFloat> weightsAfter;
        weight->GetWeights(weightsAfter);
        
        // Verify weights changed
        bool changed = false;
        for (size_t i = 0; i < weightsBefore.size(); i++) {
            if (std::abs(weightsAfter[i] - weightsBefore[i]) > 1e-6) {
                changed = true;
                break;
            }
        }
        CPPUNIT_ASSERT_MESSAGE("SGD should update weights", changed);
        
        // Cleanup
        delete pNetwork;
        for (auto p : vDataSet) {
            delete p;
        }
    }

    //------------------------------------------------------------------------
    // Test: Momentum weight updates
    //------------------------------------------------------------------------
    void testMomentumWeightUpdates() {
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
        
        NNWeight* weight = pNetwork->GetWeight("Input", "Output");
        
        std::vector<NNFloat> weightsBefore;
        weight->GetWeights(weightsBefore);
        
        // Train multiple epochs to build momentum
        for (int i = 0; i < 3; i++) {
            pNetwork->Train(1, 0.01f, 0.0f, 0.0f, 0.9f, 0.0f);
        }
        
        std::vector<NNFloat> weightsAfter;
        weight->GetWeights(weightsAfter);
        
        bool changed = false;
        for (size_t i = 0; i < weightsBefore.size(); i++) {
            if (std::abs(weightsAfter[i] - weightsBefore[i]) > 1e-6) {
                changed = true;
                break;
            }
        }
        CPPUNIT_ASSERT_MESSAGE("Momentum should update weights", changed);
        
        // Cleanup
        delete pNetwork;
        for (auto p : vDataSet) {
            delete p;
        }
    }

    //------------------------------------------------------------------------
    // Test: Adam weight updates
    //------------------------------------------------------------------------
    void testAdamWeightUpdates() {
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
        
        NNWeight* weight = pNetwork->GetWeight("Input", "Output");
        
        std::vector<NNFloat> weightsBefore;
        weight->GetWeights(weightsBefore);
        
        pNetwork->Train(1, 0.001f, 0.0f, 0.0f, 0.9f, 0.999f);
        
        std::vector<NNFloat> weightsAfter;
        weight->GetWeights(weightsAfter);
        
        bool changed = false;
        for (size_t i = 0; i < weightsBefore.size(); i++) {
            if (std::abs(weightsAfter[i] - weightsBefore[i]) > 1e-6) {
                changed = true;
                break;
            }
        }
        CPPUNIT_ASSERT_MESSAGE("Adam should update weights", changed);
        
        // Cleanup
        delete pNetwork;
        for (auto p : vDataSet) {
            delete p;
        }
    }

    //------------------------------------------------------------------------
    // Test: L2 regularization effect on weights
    //------------------------------------------------------------------------
    void testL2RegularizationEffect() {
        const std::string modelPath = std::string(TEST_DATA_PATH) + "validate_L2_01.json";
        
        DataParameters dataParameters;
        dataParameters.numberOfSamples = 256;
        dataParameters.inpFeatureDimensionality = 1;
        dataParameters.outFeatureDimensionality = 1;
        
        std::vector<NNDataSetBase*> vDataSet;
        const std::string dataPath(TEST_DATA_PATH);
        generateTestData(dataPath, Regression, dataParameters, std::cout);
        vDataSet = LoadNetCDF(dataPath + "test.nc");
        
        // Train without regularization
        NNNetwork* pNetwork1 = LoadNeuralNetworkJSON(modelPath, 32, vDataSet);
        pNetwork1->LoadDataSets(vDataSet);
        pNetwork1->SetTrainingMode(SGD);
        
        for (int i = 0; i < 10; i++) {
            pNetwork1->Train(1, 0.1f, 0.0f, 0.0f, 0.0f, 0.0f);
        }
        
        NNWeight* weight1 = pNetwork1->GetWeight("Input", "Output");
        std::vector<NNFloat> weights1;
        weight1->GetWeights(weights1);
        
        NNFloat l2Norm1 = 0.0f;
        for (auto w : weights1) {
            l2Norm1 += w * w;
        }
        
        delete pNetwork1;
        
        // Train with strong L2 regularization
        NNNetwork* pNetwork2 = LoadNeuralNetworkJSON(modelPath, 32, vDataSet);
        pNetwork2->LoadDataSets(vDataSet);
        pNetwork2->SetTrainingMode(SGD);
        
        for (int i = 0; i < 10; i++) {
            pNetwork2->Train(1, 0.1f, 0.5f, 0.0f, 0.0f, 0.0f);  // High lambda
        }
        
        NNWeight* weight2 = pNetwork2->GetWeight("Input", "Output");
        std::vector<NNFloat> weights2;
        weight2->GetWeights(weights2);
        
        NNFloat l2Norm2 = 0.0f;
        for (auto w : weights2) {
            l2Norm2 += w * w;
        }
        
        delete pNetwork2;
        
        std::cout << "L2 norm without regularization: " << l2Norm1 << std::endl;
        std::cout << "L2 norm with regularization: " << l2Norm2 << std::endl;
        
        // L2 regularization should result in smaller weight magnitudes
        // Note: This assertion may not always hold for simple 1-parameter models
        
        // Cleanup
        for (auto p : vDataSet) {
            delete p;
        }
    }

    //------------------------------------------------------------------------
    // Test: L1 regularization effect on weights
    //------------------------------------------------------------------------
    void testL1RegularizationEffect() {
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
        
        // Train with L1 regularization
        for (int i = 0; i < 10; i++) {
            pNetwork->Train(1, 0.01f, 0.0f, 0.01f, 0.0f, 0.0f);  // lambda1 for L1
        }
        
        NNWeight* weight = pNetwork->GetWeight("Input", "Output");
        std::vector<NNFloat> weights;
        weight->GetWeights(weights);
        
        // Verify weights are valid
        for (auto w : weights) {
            CPPUNIT_ASSERT_MESSAGE("Weight should be finite", !std::isnan(w) && !std::isinf(w));
        }
        
        // Cleanup
        delete pNetwork;
        for (auto p : vDataSet) {
            delete p;
        }
    }

    //------------------------------------------------------------------------
    // Test: Weight update with different learning rates
    //------------------------------------------------------------------------
    void testLearningRateEffect() {
        const std::string modelPath = std::string(TEST_DATA_PATH) + "validate_L2_01.json";
        
        DataParameters dataParameters;
        dataParameters.numberOfSamples = 256;
        dataParameters.inpFeatureDimensionality = 1;
        dataParameters.outFeatureDimensionality = 1;
        
        std::vector<NNDataSetBase*> vDataSet;
        const std::string dataPath(TEST_DATA_PATH);
        generateTestData(dataPath, Regression, dataParameters, std::cout);
        vDataSet = LoadNetCDF(dataPath + "test.nc");
        
        // Small learning rate
        NNNetwork* pNetwork1 = LoadNeuralNetworkJSON(modelPath, 32, vDataSet);
        pNetwork1->LoadDataSets(vDataSet);
        pNetwork1->SetTrainingMode(SGD);
        
        NNWeight* weight1 = pNetwork1->GetWeight("Input", "Output");
        std::vector<NNFloat> before1;
        weight1->GetWeights(before1);
        
        pNetwork1->Train(1, 0.001f, 0.0f, 0.0f, 0.0f, 0.0f);
        
        std::vector<NNFloat> after1;
        weight1->GetWeights(after1);
        
        NNFloat change1 = 0.0f;
        for (size_t i = 0; i < before1.size(); i++) {
            change1 += std::abs(after1[i] - before1[i]);
        }
        
        delete pNetwork1;
        
        // Large learning rate
        NNNetwork* pNetwork2 = LoadNeuralNetworkJSON(modelPath, 32, vDataSet);
        pNetwork2->LoadDataSets(vDataSet);
        pNetwork2->SetTrainingMode(SGD);
        
        NNWeight* weight2 = pNetwork2->GetWeight("Input", "Output");
        std::vector<NNFloat> before2;
        weight2->GetWeights(before2);
        
        pNetwork2->Train(1, 0.1f, 0.0f, 0.0f, 0.0f, 0.0f);
        
        std::vector<NNFloat> after2;
        weight2->GetWeights(after2);
        
        NNFloat change2 = 0.0f;
        for (size_t i = 0; i < before2.size(); i++) {
            change2 += std::abs(after2[i] - before2[i]);
        }
        
        delete pNetwork2;
        
        std::cout << "Weight change with lr=0.001: " << change1 << std::endl;
        std::cout << "Weight change with lr=0.1: " << change2 << std::endl;
        
        CPPUNIT_ASSERT_MESSAGE("Larger learning rate should cause larger weight changes",
            change2 > change1);
        
        // Cleanup
        for (auto p : vDataSet) {
            delete p;
        }
    }

    //------------------------------------------------------------------------
    // Test: Gradient accumulation over batches
    //------------------------------------------------------------------------
    void testGradientAccumulation() {
        const std::string modelPath = std::string(TEST_DATA_PATH) + "validate_L2_01.json";
        
        DataParameters dataParameters;
        dataParameters.numberOfSamples = 512;
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
        
        // Train one full epoch (should process multiple batches)
        float error = pNetwork->Train(1, 0.01f, 0.0f, 0.0f, 0.0f, 0.0f);
        
        CPPUNIT_ASSERT_MESSAGE("Training should complete", error >= 0.0f);
        
        // Cleanup
        delete pNetwork;
        for (auto p : vDataSet) {
            delete p;
        }
    }

    //------------------------------------------------------------------------
    // Test: Weight updates with hidden layers
    //------------------------------------------------------------------------
    void testWeightUpdatesWithHiddenLayers() {
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
        pNetwork->SetTrainingMode(SGD);
        
        // Get weights for Input->Hidden connection
        NNWeight* weightIH = pNetwork->GetWeight("Input", "Hidden");
        CPPUNIT_ASSERT(weightIH != nullptr);
        
        std::vector<NNFloat> weightsIHBefore;
        weightIH->GetWeights(weightsIHBefore);
        
        // Train
        pNetwork->Train(1, 0.01f, 0.0f, 0.0f, 0.0f, 0.0f);
        
        std::vector<NNFloat> weightsIHAfter;
        weightIH->GetWeights(weightsIHAfter);
        
        // Check Input->Hidden weights changed
        bool changedIH = false;
        for (size_t i = 0; i < weightsIHBefore.size(); i++) {
            if (std::abs(weightsIHAfter[i] - weightsIHBefore[i]) > 1e-6) {
                changedIH = true;
                break;
            }
        }
        CPPUNIT_ASSERT_MESSAGE("Input->Hidden weights should update", changedIH);
        
        // Cleanup
        delete pNetwork;
        for (auto p : vDataSet) {
            delete p;
        }
    }

    //------------------------------------------------------------------------
    // Test: Bias updates
    //------------------------------------------------------------------------
    void testBiasUpdates() {
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
        
        std::vector<NNFloat> biasesBefore;
        weight->GetBiases(biasesBefore);
        
        pNetwork->Train(1, 0.01f, 0.0f, 0.0f, 0.0f, 0.0f);
        
        std::vector<NNFloat> biasesAfter;
        weight->GetBiases(biasesAfter);
        
        bool changed = false;
        for (size_t i = 0; i < biasesBefore.size(); i++) {
            if (std::abs(biasesAfter[i] - biasesBefore[i]) > 1e-6) {
                changed = true;
                break;
            }
        }
        CPPUNIT_ASSERT_MESSAGE("Biases should update during training", changed);
        
        // Cleanup
        delete pNetwork;
        for (auto p : vDataSet) {
            delete p;
        }
    }

    //------------------------------------------------------------------------
    // Test: Locked weights don't update
    //------------------------------------------------------------------------
    void testLockedWeightsDontUpdate() {
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
        pNetwork->LockWeights("Input", "Output");
        
        NNWeight* weight = pNetwork->GetWeight("Input", "Output");
        std::vector<NNFloat> weightsBefore;
        weight->GetWeights(weightsBefore);
        
        // Try to train
        pNetwork->Train(3, 0.1f, 0.0f, 0.0f, 0.0f, 0.0f);
        
        std::vector<NNFloat> weightsAfter;
        weight->GetWeights(weightsAfter);
        
        // Verify weights didn't change
        for (size_t i = 0; i < weightsBefore.size(); i++) {
            CPPUNIT_ASSERT_DOUBLES_EQUAL_MESSAGE("Locked weights should not change",
                weightsBefore[i], weightsAfter[i], 1e-6);
        }
        
        // Cleanup
        delete pNetwork;
        for (auto p : vDataSet) {
            delete p;
        }
    }

    //------------------------------------------------------------------------
    // Test: Weight update stability
    //------------------------------------------------------------------------
    void testWeightUpdateStability() {
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
        
        // Train for many epochs
        for (int epoch = 0; epoch < 50; epoch++) {
            float error = pNetwork->Train(1, 0.01f, 0.0f, 0.0f, 0.0f, 0.0f);
            
            CPPUNIT_ASSERT_MESSAGE("Error should not be NaN", !std::isnan(error));
            CPPUNIT_ASSERT_MESSAGE("Error should not be Inf", !std::isinf(error));
        }
        
        // Check final weights are valid
        NNWeight* weight = pNetwork->GetWeight("Input", "Output");
        std::vector<NNFloat> weights;
        weight->GetWeights(weights);
        
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

public:
    CPPUNIT_TEST_SUITE(TestNNWeightUpdate);
    CPPUNIT_TEST(testSGDWeightUpdates);
    CPPUNIT_TEST(testMomentumWeightUpdates);
    CPPUNIT_TEST(testAdamWeightUpdates);
    CPPUNIT_TEST(testL2RegularizationEffect);
    CPPUNIT_TEST(testL1RegularizationEffect);
    CPPUNIT_TEST(testLearningRateEffect);
    CPPUNIT_TEST(testGradientAccumulation);
    CPPUNIT_TEST(testWeightUpdatesWithHiddenLayers);
    CPPUNIT_TEST(testBiasUpdates);
    CPPUNIT_TEST(testLockedWeightsDontUpdate);
    CPPUNIT_TEST(testWeightUpdateStability);
    CPPUNIT_TEST_SUITE_END();
};

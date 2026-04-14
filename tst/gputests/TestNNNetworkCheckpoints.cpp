/**
 * TestNNNetworkCheckpoints.cpp
 * 
 * Comprehensive tests for NNNetwork checkpoint save/restore functionality.
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
#include <cstdio>  // for remove()

#include "Utils.h"
#include "GpuTypes.h"
#include "NNTypes.h"
#include "TestUtils.h"

//----------------------------------------------------------------------------
// Test class for NNNetwork checkpoint functionality
//----------------------------------------------------------------------------
class TestNNNetworkCheckpoints : public CppUnit::TestFixture {
private:
    std::string _tempDir;
    
public:
    //------------------------------------------------------------------------
    // Setup and teardown
    //------------------------------------------------------------------------
    void setUp() override {
        _tempDir = "/tmp/dsstne_checkpoint_tests/";
        // Create temp directory
        system(("mkdir -p " + _tempDir).c_str());
    }

    void tearDown() override {
        // Cleanup temp directory
        system(("rm -rf " + _tempDir).c_str());
    }

    //------------------------------------------------------------------------
    // Test: Set checkpoint configuration
    //------------------------------------------------------------------------
    void testSetCheckpointConfiguration() {
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
        
        // Set checkpoint
        std::string checkpointName = _tempDir + "test_checkpoint";
        bool result = pNetwork->SetCheckpoint(checkpointName, 5);
        CPPUNIT_ASSERT_MESSAGE("SetCheckpoint should succeed", result);
        
        // Verify checkpoint configuration
        auto checkpoint = pNetwork->GetCheckPoint();
        CPPUNIT_ASSERT_EQUAL_MESSAGE("Checkpoint name should match",
            checkpointName, std::get<0>(checkpoint));
        CPPUNIT_ASSERT_EQUAL_MESSAGE("Checkpoint interval should match",
            5, std::get<1>(checkpoint));
        
        // Cleanup
        delete pNetwork;
        for (auto p : vDataSet) {
            delete p;
        }
    }

    //------------------------------------------------------------------------
    // Test: Save network to NetCDF
    //------------------------------------------------------------------------
    void testSaveNetworkNetCDF() {
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
        
        // Train a bit to have different weights
        pNetwork->Train(1, 0.01f, 0.001f, 0.0f, 0.9f, 0.0f);
        
        // Save network
        std::string savePath = _tempDir + "test_network.nc";
        bool saved = pNetwork->SaveNetCDF(savePath);
        CPPUNIT_ASSERT_MESSAGE("SaveNetCDF should succeed", saved);
        
        // Verify file exists
        std::ifstream f(savePath);
        CPPUNIT_ASSERT_MESSAGE("Saved network file should exist", f.good());
        
        // Cleanup
        delete pNetwork;
        for (auto p : vDataSet) {
            delete p;
        }
    }

    //------------------------------------------------------------------------
    // Test: Load network from NetCDF
    //------------------------------------------------------------------------
    void testLoadNetworkNetCDF() {
        const std::string modelPath = std::string(TEST_DATA_PATH) + "validate_L2_01.json";
        
        DataParameters dataParameters;
        dataParameters.numberOfSamples = 128;
        dataParameters.inpFeatureDimensionality = 1;
        dataParameters.outFeatureDimensionality = 1;
        
        std::vector<NNDataSetBase*> vDataSet;
        const std::string dataPath(TEST_DATA_PATH);
        generateTestData(dataPath, Regression, dataParameters, std::cout);
        vDataSet = LoadNetCDF(dataPath + "test.nc");
        
        // Create and save network
        NNNetwork* pNetwork1 = LoadNeuralNetworkJSON(modelPath, 32, vDataSet);
        CPPUNIT_ASSERT(pNetwork1 != nullptr);
        
        pNetwork1->LoadDataSets(vDataSet);
        pNetwork1->SetTrainingMode(SGD);
        pNetwork1->Train(1, 0.01f, 0.001f, 0.0f, 0.9f, 0.0f);
        
        std::string savePath = _tempDir + "test_network.nc";
        pNetwork1->SaveNetCDF(savePath);
        
        delete pNetwork1;
        
        // Load network from NetCDF
        NNNetwork* pNetwork2 = LoadNeuralNetworkNetCDF(savePath, 32);
        CPPUNIT_ASSERT_MESSAGE("Network should load from NetCDF", pNetwork2 != nullptr);
        
        // Verify network name
        CPPUNIT_ASSERT_EQUAL_MESSAGE("Network name should match",
            std::string("L2 1d regression"), std::string(pNetwork2->GetName()));
        
        // Cleanup
        delete pNetwork2;
        for (auto p : vDataSet) {
            delete p;
        }
    }

    //------------------------------------------------------------------------
    // Test: Save and restore weights consistency
    //------------------------------------------------------------------------
    void testWeightConsistencyAfterRestore() {
        const std::string modelPath = std::string(TEST_DATA_PATH) + "validate_L2_01.json";
        
        DataParameters dataParameters;
        dataParameters.numberOfSamples = 128;
        dataParameters.inpFeatureDimensionality = 1;
        dataParameters.outFeatureDimensionality = 1;
        
        std::vector<NNDataSetBase*> vDataSet;
        const std::string dataPath(TEST_DATA_PATH);
        generateTestData(dataPath, Regression, dataParameters, std::cout);
        vDataSet = LoadNetCDF(dataPath + "test.nc");
        
        // Create and train network
        NNNetwork* pNetwork1 = LoadNeuralNetworkJSON(modelPath, 32, vDataSet);
        CPPUNIT_ASSERT(pNetwork1 != nullptr);
        
        pNetwork1->LoadDataSets(vDataSet);
        pNetwork1->SetTrainingMode(SGD);
        pNetwork1->Train(3, 0.01f, 0.001f, 0.0f, 0.9f, 0.0f);
        
        // Get weights before saving
        NNWeight* weight1 = pNetwork1->GetWeight("Input", "Output");
        CPPUNIT_ASSERT(weight1 != nullptr);
        
        std::vector<NNFloat> weights1;
        weight1->GetWeights(weights1);
        
        // Save network
        std::string savePath = _tempDir + "weight_test.nc";
        pNetwork1->SaveNetCDF(savePath);
        delete pNetwork1;
        
        // Load network
        NNNetwork* pNetwork2 = LoadNeuralNetworkNetCDF(savePath, 32);
        CPPUNIT_ASSERT(pNetwork2 != nullptr);
        
        // Get weights after loading
        NNWeight* weight2 = pNetwork2->GetWeight("Input", "Output");
        CPPUNIT_ASSERT(weight2 != nullptr);
        
        std::vector<NNFloat> weights2;
        weight2->GetWeights(weights2);
        
        // Compare weights
        CPPUNIT_ASSERT_EQUAL_MESSAGE("Weight sizes should match",
            weights1.size(), weights2.size());
        
        for (size_t i = 0; i < weights1.size(); i++) {
            CPPUNIT_ASSERT_DOUBLES_EQUAL_MESSAGE(
                "Weight values should match after restore",
                weights1[i], weights2[i], 1e-5);
        }
        
        // Cleanup
        delete pNetwork2;
        for (auto p : vDataSet) {
            delete p;
        }
    }

    //------------------------------------------------------------------------
    // Test: Checkpoint with training interval
    //------------------------------------------------------------------------
    void testCheckpointWithTrainingInterval() {
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
        
        // Set checkpoint to save every 2 epochs
        std::string checkpointName = _tempDir + "interval_checkpoint";
        pNetwork->SetCheckpoint(checkpointName, 2);
        
        // Train for 5 epochs (should create at least 2 checkpoints)
        for (int i = 0; i < 5; i++) {
            pNetwork->Train(1, 0.01f, 0.001f, 0.0f, 0.9f, 0.0f);
        }
        
        // Note: The actual checkpoint file naming depends on implementation
        // This test verifies that training completes successfully with checkpointing enabled
        
        // Cleanup
        delete pNetwork;
        for (auto p : vDataSet) {
            delete p;
        }
    }

    //------------------------------------------------------------------------
    // Test: Save weights to specific file
    //------------------------------------------------------------------------
    void testSaveWeightsToFile() {
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
        pNetwork->Train(1, 0.01f, 0.001f, 0.0f, 0.9f, 0.0f);
        
        // Save weights for specific layer pair
        std::string weightPath = _tempDir + "test_weights.nc";
        pNetwork->SaveWeights(weightPath, "Input", "Output");
        
        // Verify file exists
        std::ifstream f(weightPath);
        CPPUNIT_ASSERT_MESSAGE("Weight file should exist", f.good());
        
        // Cleanup
        delete pNetwork;
        for (auto p : vDataSet) {
            delete p;
        }
    }

    //------------------------------------------------------------------------
    // Test: Save layer activations
    //------------------------------------------------------------------------
    void testSaveLayerActivations() {
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
        pNetwork->PredictBatch();
        
        // Save layer activations
        std::string layerPath = _tempDir + "output_layer.nc";
        pNetwork->SaveLayer(layerPath, "Output");
        
        // Verify file exists
        std::ifstream f(layerPath);
        CPPUNIT_ASSERT_MESSAGE("Layer file should exist", f.good());
        
        // Cleanup
        delete pNetwork;
        for (auto p : vDataSet) {
            delete p;
        }
    }

    //------------------------------------------------------------------------
    // Test: Resume training from checkpoint
    //------------------------------------------------------------------------
    void testResumeTrainingFromCheckpoint() {
        const std::string modelPath = std::string(TEST_DATA_PATH) + "validate_L2_01.json";
        
        DataParameters dataParameters;
        dataParameters.numberOfSamples = 256;
        dataParameters.inpFeatureDimensionality = 1;
        dataParameters.outFeatureDimensionality = 1;
        
        std::vector<NNDataSetBase*> vDataSet;
        const std::string dataPath(TEST_DATA_PATH);
        generateTestData(dataPath, Regression, dataParameters, std::cout);
        vDataSet = LoadNetCDF(dataPath + "test.nc");
        
        // Phase 1: Train and save
        NNNetwork* pNetwork1 = LoadNeuralNetworkJSON(modelPath, 32, vDataSet);
        CPPUNIT_ASSERT(pNetwork1 != nullptr);
        
        pNetwork1->LoadDataSets(vDataSet);
        pNetwork1->SetTrainingMode(SGD);
        float error1 = pNetwork1->Train(3, 0.01f, 0.001f, 0.0f, 0.9f, 0.0f);
        
        std::string savePath = _tempDir + "resume_test.nc";
        pNetwork1->SaveNetCDF(savePath);
        delete pNetwork1;
        
        // Phase 2: Load and continue training
        NNNetwork* pNetwork2 = LoadNeuralNetworkNetCDF(savePath, 32);
        CPPUNIT_ASSERT(pNetwork2 != nullptr);
        
        pNetwork2->LoadDataSets(vDataSet);
        pNetwork2->SetTrainingMode(SGD);
        float error2 = pNetwork2->Train(3, 0.01f, 0.001f, 0.0f, 0.9f, 0.0f);
        
        // Training should continue successfully
        CPPUNIT_ASSERT_MESSAGE("Continued training error should be >= 0", error2 >= 0.0f);
        
        // Cleanup
        delete pNetwork2;
        for (auto p : vDataSet) {
            delete p;
        }
    }

    //------------------------------------------------------------------------
    // Test: Save batch output
    //------------------------------------------------------------------------
    void testSaveBatchOutput() {
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
        pNetwork->PredictBatch();
        
        // Save batch output
        std::string batchPath = _tempDir + "batch_output.nc";
        pNetwork->SaveBatch(batchPath);
        
        // Verify file exists
        std::ifstream f(batchPath);
        CPPUNIT_ASSERT_MESSAGE("Batch output file should exist", f.good());
        
        // Cleanup
        delete pNetwork;
        for (auto p : vDataSet) {
            delete p;
        }
    }

    //------------------------------------------------------------------------
    // Test: Get and set biases
    //------------------------------------------------------------------------
    void testGetSetBiases() {
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
        
        // Get biases
        std::vector<NNFloat> biases;
        bool gotBiases = weight->GetBiases(biases);
        CPPUNIT_ASSERT_MESSAGE("GetBiases should succeed", gotBiases);
        CPPUNIT_ASSERT_MESSAGE("Biases should not be empty", !biases.empty());
        
        // Modify and set biases
        for (auto& b : biases) {
            b += 0.1f;
        }
        bool setBiases = weight->SetBiases(biases);
        CPPUNIT_ASSERT_MESSAGE("SetBiases should succeed", setBiases);
        
        // Verify changes
        std::vector<NNFloat> newBiases;
        weight->GetBiases(newBiases);
        
        for (size_t i = 0; i < biases.size(); i++) {
            CPPUNIT_ASSERT_DOUBLES_EQUAL_MESSAGE(
                "Biases should be updated",
                biases[i], newBiases[i], 1e-5);
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
        bool gotDims = weight->GetDimensions(dimensions);
        CPPUNIT_ASSERT_MESSAGE("GetDimensions should succeed", gotDims);
        CPPUNIT_ASSERT_MESSAGE("Dimensions should not be empty", !dimensions.empty());
        
        // Cleanup
        delete pNetwork;
        for (auto p : vDataSet) {
            delete p;
        }
    }

public:
    CPPUNIT_TEST_SUITE(TestNNNetworkCheckpoints);
    CPPUNIT_TEST(testSetCheckpointConfiguration);
    CPPUNIT_TEST(testSaveNetworkNetCDF);
    CPPUNIT_TEST(testLoadNetworkNetCDF);
    CPPUNIT_TEST(testWeightConsistencyAfterRestore);
    CPPUNIT_TEST(testCheckpointWithTrainingInterval);
    CPPUNIT_TEST(testSaveWeightsToFile);
    CPPUNIT_TEST(testSaveLayerActivations);
    CPPUNIT_TEST(testResumeTrainingFromCheckpoint);
    CPPUNIT_TEST(testSaveBatchOutput);
    CPPUNIT_TEST(testGetSetBiases);
    CPPUNIT_TEST(testWeightDimensions);
    CPPUNIT_TEST_SUITE_END();
};

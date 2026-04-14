/**
 * TestNNLayerForward.cpp
 * 
 * Comprehensive tests for NNLayer forward propagation functionality.
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

#include "Utils.h"
#include "GpuTypes.h"
#include "NNTypes.h"
#include "TestUtils.h"

//----------------------------------------------------------------------------
// Test class for NNLayer forward propagation
//----------------------------------------------------------------------------
class TestNNLayerForward : public CppUnit::TestFixture {
public:
    //------------------------------------------------------------------------
    // Test: Input layer kind
    //------------------------------------------------------------------------
    void testInputLayerKind() {
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
        
        NNLayer* inputLayer = pNetwork->GetLayer("Input");
        CPPUNIT_ASSERT(inputLayer != nullptr);
        
        CPPUNIT_ASSERT_EQUAL_MESSAGE("Input layer should have Input kind",
            NNLayer::Kind::Input, inputLayer->GetKind());
        
        // Cleanup
        delete pNetwork;
        for (auto p : vDataSet) {
            delete p;
        }
    }

    //------------------------------------------------------------------------
    // Test: Hidden layer kind
    //------------------------------------------------------------------------
    void testHiddenLayerKind() {
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
        
        NNLayer* hiddenLayer = pNetwork->GetLayer("Hidden");
        CPPUNIT_ASSERT(hiddenLayer != nullptr);
        
        CPPUNIT_ASSERT_EQUAL_MESSAGE("Hidden layer should have Hidden kind",
            NNLayer::Kind::Hidden, hiddenLayer->GetKind());
        
        // Cleanup
        delete pNetwork;
        for (auto p : vDataSet) {
            delete p;
        }
    }

    //------------------------------------------------------------------------
    // Test: Output layer kind
    //------------------------------------------------------------------------
    void testOutputLayerKind() {
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
        
        NNLayer* outputLayer = pNetwork->GetLayer("Output");
        CPPUNIT_ASSERT(outputLayer != nullptr);
        
        CPPUNIT_ASSERT_EQUAL_MESSAGE("Output layer should have Output kind",
            NNLayer::Kind::Output, outputLayer->GetKind());
        
        // Cleanup
        delete pNetwork;
        for (auto p : vDataSet) {
            delete p;
        }
    }

    //------------------------------------------------------------------------
    // Test: Layer type (FullyConnected)
    //------------------------------------------------------------------------
    void testFullyConnectedLayerType() {
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
        
        NNLayer* outputLayer = pNetwork->GetLayer("Output");
        CPPUNIT_ASSERT(outputLayer != nullptr);
        
        CPPUNIT_ASSERT_EQUAL_MESSAGE("Layer should be FullyConnected",
            NNLayer::Type::FullyConnected, outputLayer->GetType());
        
        // Cleanup
        delete pNetwork;
        for (auto p : vDataSet) {
            delete p;
        }
    }

    //------------------------------------------------------------------------
    // Test: Layer name
    //------------------------------------------------------------------------
    void testLayerName() {
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
        
        NNLayer* inputLayer = pNetwork->GetLayer("Input");
        CPPUNIT_ASSERT(inputLayer != nullptr);
        
        CPPUNIT_ASSERT_EQUAL_MESSAGE("Layer name should match",
            std::string("Input"), inputLayer->GetName());
        
        // Cleanup
        delete pNetwork;
        for (auto p : vDataSet) {
            delete p;
        }
    }

    //------------------------------------------------------------------------
    // Test: Layer dimensions
    //------------------------------------------------------------------------
    void testLayerDimensions() {
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
        
        NNLayer* inputLayer = pNetwork->GetLayer("Input");
        CPPUNIT_ASSERT(inputLayer != nullptr);
        
        auto dims = inputLayer->GetDimensions();
        uint32_t nx = std::get<0>(dims);
        
        CPPUNIT_ASSERT_EQUAL_MESSAGE("Input layer X dimension should match",
            static_cast<uint32_t>(2), nx);
        
        // Cleanup
        delete pNetwork;
        for (auto p : vDataSet) {
            delete p;
        }
    }

    //------------------------------------------------------------------------
    // Test: Number of dimensions
    //------------------------------------------------------------------------
    void testNumberOfDimensions() {
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
        
        NNLayer* inputLayer = pNetwork->GetLayer("Input");
        CPPUNIT_ASSERT(inputLayer != nullptr);
        
        uint32_t numDims = inputLayer->GetNumDimensions();
        CPPUNIT_ASSERT_MESSAGE("Number of dimensions should be >= 1", numDims >= 1);
        
        // Cleanup
        delete pNetwork;
        for (auto p : vDataSet) {
            delete p;
        }
    }

    //------------------------------------------------------------------------
    // Test: Dataset name association
    //------------------------------------------------------------------------
    void testDataSetNameAssociation() {
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
        
        NNLayer* inputLayer = pNetwork->GetLayer("Input");
        CPPUNIT_ASSERT(inputLayer != nullptr);
        
        const std::string& dataSetName = inputLayer->GetDataSetName();
        CPPUNIT_ASSERT_EQUAL_MESSAGE("Input layer should reference 'input' dataset",
            std::string("input"), dataSetName);
        
        // Cleanup
        delete pNetwork;
        for (auto p : vDataSet) {
            delete p;
        }
    }

    //------------------------------------------------------------------------
    // Test: Get units after forward propagation
    //------------------------------------------------------------------------
    void testGetUnitsAfterForward() {
        const std::string modelPath = std::string(TEST_DATA_PATH) + "validate_L2_01.json";
        
        DataParameters dataParameters;
        dataParameters.numberOfSamples = 128;
        dataParameters.inpFeatureDimensionality = 1;
        dataParameters.outFeatureDimensionality = 1;
        
        std::vector<NNDataSetBase*> vDataSet;
        const std::string dataPath(TEST_DATA_PATH);
        generateTestData(dataPath, Regression, dataParameters, std::cout);
        vDataSet = LoadNetCDF(dataPath + "test.nc");
        
        uint32_t batchSize = 32;
        NNNetwork* pNetwork = LoadNeuralNetworkJSON(modelPath, batchSize, vDataSet);
        CPPUNIT_ASSERT(pNetwork != nullptr);
        
        pNetwork->LoadDataSets(vDataSet);
        pNetwork->SetPosition(0);
        pNetwork->PredictBatch();
        
        NNLayer* outputLayer = pNetwork->GetLayer("Output");
        CPPUNIT_ASSERT(outputLayer != nullptr);
        
        std::vector<NNFloat> units;
        bool gotUnits = outputLayer->GetUnits(units);
        CPPUNIT_ASSERT_MESSAGE("GetUnits should succeed", gotUnits);
        CPPUNIT_ASSERT_MESSAGE("Units should not be empty", !units.empty());
        
        // Expected size: batch * output_dim
        size_t expectedSize = batchSize * dataParameters.outFeatureDimensionality;
        CPPUNIT_ASSERT_EQUAL_MESSAGE("Units size should match batch * output_dim",
            expectedSize, units.size());
        
        // Cleanup
        delete pNetwork;
        for (auto p : vDataSet) {
            delete p;
        }
    }

    //------------------------------------------------------------------------
    // Test: Set units
    //------------------------------------------------------------------------
    void testSetUnits() {
        const std::string modelPath = std::string(TEST_DATA_PATH) + "validate_L2_01.json";
        
        DataParameters dataParameters;
        dataParameters.numberOfSamples = 128;
        dataParameters.inpFeatureDimensionality = 1;
        dataParameters.outFeatureDimensionality = 1;
        
        std::vector<NNDataSetBase*> vDataSet;
        const std::string dataPath(TEST_DATA_PATH);
        generateTestData(dataPath, Regression, dataParameters, std::cout);
        vDataSet = LoadNetCDF(dataPath + "test.nc");
        
        uint32_t batchSize = 32;
        NNNetwork* pNetwork = LoadNeuralNetworkJSON(modelPath, batchSize, vDataSet);
        CPPUNIT_ASSERT(pNetwork != nullptr);
        
        pNetwork->LoadDataSets(vDataSet);
        
        NNLayer* outputLayer = pNetwork->GetLayer("Output");
        CPPUNIT_ASSERT(outputLayer != nullptr);
        
        // Create test units
        size_t unitSize = batchSize * dataParameters.outFeatureDimensionality;
        std::vector<NNFloat> testUnits(unitSize, 0.5f);
        
        bool setUnits = outputLayer->SetUnits(testUnits);
        CPPUNIT_ASSERT_MESSAGE("SetUnits should succeed", setUnits);
        
        // Verify units were set
        std::vector<NNFloat> retrievedUnits;
        outputLayer->GetUnits(retrievedUnits);
        
        CPPUNIT_ASSERT_EQUAL_MESSAGE("Retrieved units size should match",
            testUnits.size(), retrievedUnits.size());
        
        // Cleanup
        delete pNetwork;
        for (auto p : vDataSet) {
            delete p;
        }
    }

    //------------------------------------------------------------------------
    // Test: Layer attributes (Sparse)
    //------------------------------------------------------------------------
    void testLayerSparseAttribute() {
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
        
        NNLayer* inputLayer = pNetwork->GetLayer("Input");
        CPPUNIT_ASSERT(inputLayer != nullptr);
        
        uint32_t attributes = inputLayer->GetAttributes();
        
        // Check if Sparse attribute is set (model specifies Sparse: true)
        bool isSparse = (attributes & NNLayer::Attributes::Sparse) != 0;
        CPPUNIT_ASSERT_MESSAGE("Layer should have Sparse attribute", isSparse);
        
        // Cleanup
        delete pNetwork;
        for (auto p : vDataSet) {
            delete p;
        }
    }

    //------------------------------------------------------------------------
    // Test: Get dataset pointer
    //------------------------------------------------------------------------
    void testGetDataSetPointer() {
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
        
        NNLayer* inputLayer = pNetwork->GetLayer("Input");
        CPPUNIT_ASSERT(inputLayer != nullptr);
        
        NNDataSetBase* dataSet = inputLayer->GetDataSet();
        CPPUNIT_ASSERT_MESSAGE("DataSet should be associated with input layer", 
            dataSet != nullptr);
        
        // Cleanup
        delete pNetwork;
        for (auto p : vDataSet) {
            delete p;
        }
    }

    //------------------------------------------------------------------------
    // Test: Forward propagation consistency
    //------------------------------------------------------------------------
    void testForwardPropagationConsistency() {
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
        
        // Run forward propagation twice and verify same outputs
        pNetwork->PredictBatch();
        
        NNLayer* outputLayer = pNetwork->GetLayer("Output");
        std::vector<NNFloat> units1;
        outputLayer->GetUnits(units1);
        
        pNetwork->SetPosition(0);
        pNetwork->PredictBatch();
        
        std::vector<NNFloat> units2;
        outputLayer->GetUnits(units2);
        
        CPPUNIT_ASSERT_EQUAL_MESSAGE("Output sizes should match", units1.size(), units2.size());
        
        for (size_t i = 0; i < units1.size(); i++) {
            CPPUNIT_ASSERT_DOUBLES_EQUAL_MESSAGE(
                "Outputs should be consistent across runs",
                units1[i], units2[i], 1e-5);
        }
        
        // Cleanup
        delete pNetwork;
        for (auto p : vDataSet) {
            delete p;
        }
    }

    //------------------------------------------------------------------------
    // Test: Multiple hidden layers forward propagation
    //------------------------------------------------------------------------
    void testMultipleHiddenLayersForward() {
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
        
        // Check that all layers have valid outputs
        NNLayer* inputLayer = pNetwork->GetLayer("Input");
        NNLayer* hiddenLayer = pNetwork->GetLayer("Hidden");
        NNLayer* outputLayer = pNetwork->GetLayer("Output");
        
        std::vector<NNFloat> inputUnits, hiddenUnits, outputUnits;
        inputLayer->GetUnits(inputUnits);
        hiddenLayer->GetUnits(hiddenUnits);
        outputLayer->GetUnits(outputUnits);
        
        CPPUNIT_ASSERT_MESSAGE("Input layer should have units", !inputUnits.empty());
        CPPUNIT_ASSERT_MESSAGE("Hidden layer should have units", !hiddenUnits.empty());
        CPPUNIT_ASSERT_MESSAGE("Output layer should have units", !outputUnits.empty());
        
        // Cleanup
        delete pNetwork;
        for (auto p : vDataSet) {
            delete p;
        }
    }

    //------------------------------------------------------------------------
    // Test: Get local dimensions
    //------------------------------------------------------------------------
    void testGetLocalDimensions() {
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
        
        NNLayer* outputLayer = pNetwork->GetLayer("Output");
        CPPUNIT_ASSERT(outputLayer != nullptr);
        
        auto localDims = outputLayer->GetLocalDimensions();
        uint32_t localNx = std::get<0>(localDims);
        
        // In single-GPU mode, local dimensions should match global dimensions
        auto globalDims = outputLayer->GetDimensions();
        uint32_t globalNx = std::get<0>(globalDims);
        
        CPPUNIT_ASSERT_EQUAL_MESSAGE("Local and global X dimensions should match in single-GPU",
            globalNx, localNx);
        
        // Cleanup
        delete pNetwork;
        for (auto p : vDataSet) {
            delete p;
        }
    }

public:
    CPPUNIT_TEST_SUITE(TestNNLayerForward);
    CPPUNIT_TEST(testInputLayerKind);
    CPPUNIT_TEST(testHiddenLayerKind);
    CPPUNIT_TEST(testOutputLayerKind);
    CPPUNIT_TEST(testFullyConnectedLayerType);
    CPPUNIT_TEST(testLayerName);
    CPPUNIT_TEST(testLayerDimensions);
    CPPUNIT_TEST(testNumberOfDimensions);
    CPPUNIT_TEST(testDataSetNameAssociation);
    CPPUNIT_TEST(testGetUnitsAfterForward);
    CPPUNIT_TEST(testSetUnits);
    CPPUNIT_TEST(testLayerSparseAttribute);
    CPPUNIT_TEST(testGetDataSetPointer);
    CPPUNIT_TEST(testForwardPropagationConsistency);
    CPPUNIT_TEST(testMultipleHiddenLayersForward);
    CPPUNIT_TEST(testGetLocalDimensions);
    CPPUNIT_TEST_SUITE_END();
};

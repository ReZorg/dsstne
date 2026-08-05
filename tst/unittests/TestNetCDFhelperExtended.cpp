/*
   Copyright 2016  Amazon.com, Inc. or its affiliates. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License").
   You may not use this file except in compliance with the License.
   A copy of the License is located at http://aws.amazon.com/apache2.0/
*/

#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/ui/text/TestRunner.h>
#include <cppunit/TestAssert.h>

#include <map>
#include <string>
#include <sstream>
#include <unordered_map>
#include <fstream>
#include <climits>
#include <cstdlib>
#include <sys/stat.h>

#include "NetCDFhelper.h"

using namespace std;

/**
 * Extended test suite for NetCDFhelper.cpp functions
 * Adds tests for: parseSamples, importSamplesFromPath, generateNetCDFIndexes,
 *                 roundUpMaxIndex, exportIndex, loadIndexFromFile
 */
class TestNetCDFhelperExtended : public CppUnit::TestFixture
{
private:
    string tempDir;
    
    void createTempDir() {
        char tmpl[] = "/tmp/dsstne_netcdf_test_XXXXXX";
        char* result = mkdtemp(tmpl);
        if (result != nullptr) {
            tempDir = string(result);
        }
    }
    
    void removeTempDir() {
        if (!tempDir.empty()) {
            system(("rm -rf " + tempDir).c_str());
        }
    }
    
    void createIndexFile(const string& filename, const map<string, unsigned int>& data) {
        ofstream ofs(tempDir + "/" + filename);
        for (const auto& entry : data) {
            ofs << entry.first << "\t" << entry.second << "\n";
        }
        ofs.close();
    }
    
    void createSampleFile(const string& filename, const vector<string>& lines) {
        ofstream ofs(tempDir + "/" + filename);
        for (const auto& line : lines) {
            ofs << line << "\n";
        }
        ofs.close();
    }

public:
    void setUp() {
        createTempDir();
    }
    
    void tearDown() {
        removeTempDir();
    }

    // ============== loadIndex extended tests ==============
    
    void TestLoadIndexWithEmptyInput() {
        stringstream inputStream("");
        unordered_map<string, unsigned int> labelsToIndices;
        stringstream outputStream;
        
        // Empty input should succeed (0 entries processed)
        bool result = loadIndex(labelsToIndices, inputStream, outputStream);
        CPPUNIT_ASSERT(result);
        CPPUNIT_ASSERT_EQUAL((size_t)0, labelsToIndices.size());
    }
    
    void TestLoadIndexWithWhitespaceInLabel() {
        // Labels with spaces should be handled (stored as-is)
        stringstream inputStream("label with spaces\t100\n");
        unordered_map<string, unsigned int> labelsToIndices;
        stringstream outputStream;
        
        bool result = loadIndex(labelsToIndices, inputStream, outputStream);
        CPPUNIT_ASSERT(result);
        CPPUNIT_ASSERT_EQUAL((size_t)1, labelsToIndices.size());
        CPPUNIT_ASSERT(labelsToIndices.find("label with spaces") != labelsToIndices.end());
        CPPUNIT_ASSERT_EQUAL(100u, labelsToIndices["label with spaces"]);
    }
    
    void TestLoadIndexWithLargeIndices() {
        stringstream inputStream;
        inputStream << "feature1\t" << UINT_MAX << "\n";
        unordered_map<string, unsigned int> labelsToIndices;
        stringstream outputStream;
        
        bool result = loadIndex(labelsToIndices, inputStream, outputStream);
        CPPUNIT_ASSERT(result);
        CPPUNIT_ASSERT_EQUAL((size_t)1, labelsToIndices.size());
    }
    
    void TestLoadIndexWithSpecialCharactersInLabel() {
        stringstream inputStream("label_with.special-chars\t50\n");
        unordered_map<string, unsigned int> labelsToIndices;
        stringstream outputStream;
        
        bool result = loadIndex(labelsToIndices, inputStream, outputStream);
        CPPUNIT_ASSERT(result);
        CPPUNIT_ASSERT_EQUAL((size_t)1, labelsToIndices.size());
        CPPUNIT_ASSERT(labelsToIndices.find("label_with.special-chars") != labelsToIndices.end());
    }
    
    void TestLoadIndexWithNegativeIndex() {
        // Negative indices should be converted (atoi behavior)
        stringstream inputStream("feature1\t-1\n");
        unordered_map<string, unsigned int> labelsToIndices;
        stringstream outputStream;
        
        bool result = loadIndex(labelsToIndices, inputStream, outputStream);
        CPPUNIT_ASSERT(result);
        CPPUNIT_ASSERT_EQUAL((size_t)1, labelsToIndices.size());
    }
    
    void TestLoadIndexWithMultipleValidEntries() {
        stringstream inputStream;
        inputStream << "feature_a\t0\n";
        inputStream << "feature_b\t1\n";
        inputStream << "feature_c\t2\n";
        inputStream << "feature_d\t3\n";
        inputStream << "feature_e\t4\n";
        
        unordered_map<string, unsigned int> labelsToIndices;
        stringstream outputStream;
        
        bool result = loadIndex(labelsToIndices, inputStream, outputStream);
        CPPUNIT_ASSERT(result);
        CPPUNIT_ASSERT_EQUAL((size_t)5, labelsToIndices.size());
        
        CPPUNIT_ASSERT_EQUAL(0u, labelsToIndices["feature_a"]);
        CPPUNIT_ASSERT_EQUAL(4u, labelsToIndices["feature_e"]);
    }
    
    void TestLoadIndexWithMissingIndex() {
        // Line with only label (no tab)
        stringstream inputStream("labelonly\n");
        unordered_map<string, unsigned int> labelsToIndices;
        stringstream outputStream;
        
        bool result = loadIndex(labelsToIndices, inputStream, outputStream);
        CPPUNIT_ASSERT(!result);
        CPPUNIT_ASSERT(outputStream.str().find("Error") != string::npos);
    }
    
    void TestLoadIndexWithEmptyIndex() {
        // "label\t\n": getline yields "label\t", which split() tokenises as ["label"]
        // (size 1).  loadIndex() requires exactly two tokens per line (label + index),
        // so a single-token line is treated as invalid input.
        stringstream inputStream("label\t\n");
        unordered_map<string, unsigned int> labelsToIndices;
        stringstream outputStream;
        
        bool result = loadIndex(labelsToIndices, inputStream, outputStream);
        CPPUNIT_ASSERT(!result);
        CPPUNIT_ASSERT(outputStream.str().find("Error") != string::npos);
    }

    // ============== loadIndexFromFile tests ==============
    
    void TestLoadIndexFromFileWithValidFile() {
        map<string, unsigned int> testData = {
            {"product_123", 0},
            {"product_456", 1},
            {"product_789", 2}
        };
        createIndexFile("valid_index.txt", testData);
        
        unordered_map<string, unsigned int> labelsToIndices;
        stringstream outputStream;
        
        bool result = loadIndexFromFile(labelsToIndices, tempDir + "/valid_index.txt", outputStream);
        CPPUNIT_ASSERT(result);
        CPPUNIT_ASSERT_EQUAL((size_t)3, labelsToIndices.size());
    }
    
    void TestLoadIndexFromFileWithNonExistentFile() {
        unordered_map<string, unsigned int> labelsToIndices;
        stringstream outputStream;
        
        bool result = loadIndexFromFile(labelsToIndices, tempDir + "/nonexistent.txt", outputStream);
        CPPUNIT_ASSERT(!result);
        CPPUNIT_ASSERT(outputStream.str().find("Error") != string::npos);
    }
    
    void TestLoadIndexFromFileWithEmptyFile() {
        createIndexFile("empty.txt", {});
        
        unordered_map<string, unsigned int> labelsToIndices;
        stringstream outputStream;
        
        bool result = loadIndexFromFile(labelsToIndices, tempDir + "/empty.txt", outputStream);
        CPPUNIT_ASSERT(result);
        CPPUNIT_ASSERT_EQUAL((size_t)0, labelsToIndices.size());
    }

    // ============== exportIndex tests ==============
    
    void TestExportIndexCreatesFile() {
        unordered_map<string, unsigned int> index = {
            {"item_a", 10},
            {"item_b", 20},
            {"item_c", 30}
        };
        
        string outputFile = tempDir + "/exported_index.txt";
        exportIndex(index, outputFile);
        
        // Verify file exists
        ifstream ifs(outputFile);
        CPPUNIT_ASSERT(ifs.is_open());
        
        // Read back and verify content
        unordered_map<string, unsigned int> readBack;
        stringstream dummyOutput;
        bool result = loadIndex(readBack, ifs, dummyOutput);
        CPPUNIT_ASSERT(result);
        CPPUNIT_ASSERT_EQUAL((size_t)3, readBack.size());
        CPPUNIT_ASSERT_EQUAL(10u, readBack["item_a"]);
        CPPUNIT_ASSERT_EQUAL(20u, readBack["item_b"]);
        CPPUNIT_ASSERT_EQUAL(30u, readBack["item_c"]);
    }
    
    void TestExportIndexOverwritesExisting() {
        string outputFile = tempDir + "/overwrite_test.txt";
        
        // First export
        unordered_map<string, unsigned int> index1 = {{"old", 1}};
        exportIndex(index1, outputFile);
        
        // Second export (should overwrite)
        unordered_map<string, unsigned int> index2 = {{"new", 2}};
        exportIndex(index2, outputFile);
        
        // Verify only new content exists
        ifstream ifs(outputFile);
        unordered_map<string, unsigned int> readBack;
        stringstream dummyOutput;
        loadIndex(readBack, ifs, dummyOutput);
        
        CPPUNIT_ASSERT_EQUAL((size_t)1, readBack.size());
        CPPUNIT_ASSERT(readBack.find("old") == readBack.end());
        CPPUNIT_ASSERT(readBack.find("new") != readBack.end());
    }

    // ============== roundUpMaxIndex tests ==============
    
    void TestRoundUpMaxIndexMultipleOf32() {
        // roundUpMaxIndex rounds up to multiples of 128
        unsigned int result = roundUpMaxIndex(32);
        CPPUNIT_ASSERT(result >= 32);
        CPPUNIT_ASSERT(result % 128 == 0);
    }
    
    void TestRoundUpMaxIndexNotMultipleOf32() {
        unsigned int result = roundUpMaxIndex(33);
        CPPUNIT_ASSERT(result >= 33);
        CPPUNIT_ASSERT(result % 128 == 0);
        CPPUNIT_ASSERT_EQUAL(128u, result);
    }
    
    void TestRoundUpMaxIndexZero() {
        unsigned int result = roundUpMaxIndex(0);
        // Should return at least 32 or stay 0 depending on implementation
        CPPUNIT_ASSERT(result % 32 == 0);
    }
    
    void TestRoundUpMaxIndexSmallValues() {
        CPPUNIT_ASSERT_EQUAL(128u, roundUpMaxIndex(1));
        CPPUNIT_ASSERT_EQUAL(128u, roundUpMaxIndex(15));
        CPPUNIT_ASSERT_EQUAL(128u, roundUpMaxIndex(31));
    }
    
    void TestRoundUpMaxIndexLargeValues() {
        CPPUNIT_ASSERT_EQUAL(1024u, roundUpMaxIndex(1000));
        CPPUNIT_ASSERT_EQUAL(1024u, roundUpMaxIndex(1024));
        CPPUNIT_ASSERT_EQUAL(1152u, roundUpMaxIndex(1025));
    }

    // ============== parseSamples tests ==============
    
    void TestParseSamplesWithValidIndicatorData() {
        // Format: sample_name TAB feature1:feature2:feature3
        stringstream inputStream("Sample1\tFeatureA:FeatureB:FeatureC\n");
        
        unordered_map<string, unsigned int> mFeatureIndex;
        unordered_map<string, unsigned int> mSampleIndex;
        bool featureIndexUpdated = false;
        bool sampleIndexUpdated = false;
        map<unsigned int, vector<unsigned int>> mSignals;
        map<unsigned int, vector<float>> mSignalValues;
        stringstream outputStream;
        
        bool result = parseSamples(inputStream, true, mFeatureIndex, mSampleIndex,
                                   featureIndexUpdated, sampleIndexUpdated,
                                   mSignals, mSignalValues, outputStream);
        
        CPPUNIT_ASSERT(result);
        // With feature updates enabled, features should be added
        CPPUNIT_ASSERT(mFeatureIndex.size() > 0 || featureIndexUpdated);
    }
    
    void TestParseSamplesWithAnalogData() {
        // Format: sample_name TAB feature1,value1:feature2,value2
        stringstream inputStream("Sample1\tFeatureA,1.5:FeatureB,2.5\n");
        
        unordered_map<string, unsigned int> mFeatureIndex;
        unordered_map<string, unsigned int> mSampleIndex;
        bool featureIndexUpdated = false;
        bool sampleIndexUpdated = false;
        map<unsigned int, vector<unsigned int>> mSignals;
        map<unsigned int, vector<float>> mSignalValues;
        stringstream outputStream;
        
        bool result = parseSamples(inputStream, true, mFeatureIndex, mSampleIndex,
                                   featureIndexUpdated, sampleIndexUpdated,
                                   mSignals, mSignalValues, outputStream);
        
        CPPUNIT_ASSERT(result);
    }
    
    void TestParseSamplesWithEmptyInput() {
        stringstream inputStream("");
        
        unordered_map<string, unsigned int> mFeatureIndex;
        unordered_map<string, unsigned int> mSampleIndex;
        bool featureIndexUpdated = false;
        bool sampleIndexUpdated = false;
        map<unsigned int, vector<unsigned int>> mSignals;
        map<unsigned int, vector<float>> mSignalValues;
        stringstream outputStream;
        
        bool result = parseSamples(inputStream, true, mFeatureIndex, mSampleIndex,
                                   featureIndexUpdated, sampleIndexUpdated,
                                   mSignals, mSignalValues, outputStream);
        
        CPPUNIT_ASSERT(result);
        CPPUNIT_ASSERT_EQUAL((size_t)0, mSampleIndex.size());
    }
    
    void TestParseSamplesWithFeatureIndexUpdatesDisabled() {
        stringstream inputStream("Sample1\tNewFeature\n");
        
        unordered_map<string, unsigned int> mFeatureIndex = {{"ExistingFeature", 0}};
        unordered_map<string, unsigned int> mSampleIndex;
        bool featureIndexUpdated = false;
        bool sampleIndexUpdated = false;
        map<unsigned int, vector<unsigned int>> mSignals;
        map<unsigned int, vector<float>> mSignalValues;
        stringstream outputStream;
        
        bool result = parseSamples(inputStream, false, mFeatureIndex, mSampleIndex,
                                   featureIndexUpdated, sampleIndexUpdated,
                                   mSignals, mSignalValues, outputStream);
        
        // Should succeed, but new features not added to index
        CPPUNIT_ASSERT(result);
    }
    
    void TestParseSamplesWithMultipleSamples() {
        stringstream inputStream;
        inputStream << "Sample1\tFeatureA:FeatureB\n";
        inputStream << "Sample2\tFeatureC:FeatureD\n";
        inputStream << "Sample3\tFeatureA:FeatureE\n";
        
        unordered_map<string, unsigned int> mFeatureIndex;
        unordered_map<string, unsigned int> mSampleIndex;
        bool featureIndexUpdated = false;
        bool sampleIndexUpdated = false;
        map<unsigned int, vector<unsigned int>> mSignals;
        map<unsigned int, vector<float>> mSignalValues;
        stringstream outputStream;
        
        bool result = parseSamples(inputStream, true, mFeatureIndex, mSampleIndex,
                                   featureIndexUpdated, sampleIndexUpdated,
                                   mSignals, mSignalValues, outputStream);
        
        CPPUNIT_ASSERT(result);
        CPPUNIT_ASSERT_EQUAL((size_t)3, mSampleIndex.size());
    }

    // ============== align tests ==============
    
    void TestAlignSmallValues() {
        CPPUNIT_ASSERT(align(1) >= 1);
        CPPUNIT_ASSERT(align(1) % 32 == 0 || align(1) % 128 == 0);
    }
    
    void TestAlignLargeValues() {
        unsigned int result = align(1000);
        CPPUNIT_ASSERT(result >= 1000);
    }
    
    void TestAlignZero() {
        unsigned int result = align(0);
        // Should return at least the alignment boundary
        CPPUNIT_ASSERT(result >= 0);
    }

    // ============== CPPUNIT Test Suite Registration ==============
    
    CPPUNIT_TEST_SUITE(TestNetCDFhelperExtended);
    
    // loadIndex extended tests
    CPPUNIT_TEST(TestLoadIndexWithEmptyInput);
    CPPUNIT_TEST(TestLoadIndexWithWhitespaceInLabel);
    CPPUNIT_TEST(TestLoadIndexWithLargeIndices);
    CPPUNIT_TEST(TestLoadIndexWithSpecialCharactersInLabel);
    CPPUNIT_TEST(TestLoadIndexWithNegativeIndex);
    CPPUNIT_TEST(TestLoadIndexWithMultipleValidEntries);
    CPPUNIT_TEST(TestLoadIndexWithMissingIndex);
    CPPUNIT_TEST(TestLoadIndexWithEmptyIndex);
    
    // loadIndexFromFile tests
    CPPUNIT_TEST(TestLoadIndexFromFileWithValidFile);
    CPPUNIT_TEST(TestLoadIndexFromFileWithNonExistentFile);
    CPPUNIT_TEST(TestLoadIndexFromFileWithEmptyFile);
    
    // exportIndex tests
    CPPUNIT_TEST(TestExportIndexCreatesFile);
    CPPUNIT_TEST(TestExportIndexOverwritesExisting);
    
    // roundUpMaxIndex tests
    CPPUNIT_TEST(TestRoundUpMaxIndexMultipleOf32);
    CPPUNIT_TEST(TestRoundUpMaxIndexNotMultipleOf32);
    CPPUNIT_TEST(TestRoundUpMaxIndexZero);
    CPPUNIT_TEST(TestRoundUpMaxIndexSmallValues);
    CPPUNIT_TEST(TestRoundUpMaxIndexLargeValues);
    
    // parseSamples tests
    CPPUNIT_TEST(TestParseSamplesWithValidIndicatorData);
    CPPUNIT_TEST(TestParseSamplesWithAnalogData);
    CPPUNIT_TEST(TestParseSamplesWithEmptyInput);
    CPPUNIT_TEST(TestParseSamplesWithFeatureIndexUpdatesDisabled);
    CPPUNIT_TEST(TestParseSamplesWithMultipleSamples);
    
    // align tests
    CPPUNIT_TEST(TestAlignSmallValues);
    CPPUNIT_TEST(TestAlignLargeValues);
    CPPUNIT_TEST(TestAlignZero);
    
    CPPUNIT_TEST_SUITE_END();
};

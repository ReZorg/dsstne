/*
   Copyright 2016  Amazon.com, Inc. or its affiliates. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License").
   You may not use this file except in compliance with the License.
   A copy of the License is located at http://aws.amazon.com/apache2.0/
*/

#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/ui/text/TestRunner.h>
#include <cppunit/TestAssert.h>

#include <fstream>
#include <map>
#include <string>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <cstdlib>
#include <sys/stat.h>

#include "NetCDFhelper.h"
#include "Utils.h"

using namespace std;

/**
 * Tests for importSamplesFromPath and related integration scenarios.
 */
class TestImportSamples : public CppUnit::TestFixture
{
private:
    string tempDir;

    void createTempDir() {
        char tmpl[] = "/tmp/dsstne_import_test_XXXXXX";
        char* result = mkdtemp(tmpl);
        if (result != nullptr) {
            tempDir = string(result);
        }
    }

    void removeTempDir() {
        if (!tempDir.empty()) {
            // tempDir comes from mkdtemp() which produces only alphanumeric/underscore
            // paths, so there are no shell metacharacters.  Verify this before use.
            for (char c : tempDir) {
                if (!isalnum(c) && c != '/' && c != '_' && c != '-' && c != '.') {
                    return; // unexpected character - skip removal
                }
            }
            system(("rm -rf " + tempDir).c_str());
        }
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

    // ============== importSamplesFromPath ==============

    void TestImportSamplesFromPath_NonExistentPath() {
        unordered_map<string, unsigned int> mFeatureIndex;
        unordered_map<string, unsigned int> mSampleIndex;
        bool featureIndexUpdated = false;
        bool sampleIndexUpdated = false;
        vector<unsigned int> vSparseStart, vSparseEnd, vSparseIndex;
        vector<float> vSparseData;
        stringstream outputStream;

        bool result = importSamplesFromPath(
            tempDir + "/no_such_file.txt", true,
            mFeatureIndex, mSampleIndex,
            featureIndexUpdated, sampleIndexUpdated,
            vSparseStart, vSparseEnd, vSparseIndex, vSparseData,
            outputStream);

        CPPUNIT_ASSERT(!result);
        CPPUNIT_ASSERT_MESSAGE("Error message expected for non-existent path",
            outputStream.str().find("Error") != string::npos);
    }

    void TestImportSamplesFromPath_SingleFileSingleSample() {
        createSampleFile("samples.txt", {"customer1\tfeatureA,1.0:featureB,2.0"});

        unordered_map<string, unsigned int> mFeatureIndex;
        unordered_map<string, unsigned int> mSampleIndex;
        bool featureIndexUpdated = false;
        bool sampleIndexUpdated = false;
        vector<unsigned int> vSparseStart, vSparseEnd, vSparseIndex;
        vector<float> vSparseData;
        stringstream outputStream;

        bool result = importSamplesFromPath(
            tempDir + "/samples.txt", true,
            mFeatureIndex, mSampleIndex,
            featureIndexUpdated, sampleIndexUpdated,
            vSparseStart, vSparseEnd, vSparseIndex, vSparseData,
            outputStream);

        CPPUNIT_ASSERT(result);
        CPPUNIT_ASSERT(sampleIndexUpdated);
        CPPUNIT_ASSERT(featureIndexUpdated);
        CPPUNIT_ASSERT_EQUAL((size_t)1, mSampleIndex.size());
        CPPUNIT_ASSERT_EQUAL((size_t)2, mFeatureIndex.size());
        // One sample, so one start and end entry
        CPPUNIT_ASSERT_EQUAL((size_t)1, vSparseStart.size());
        CPPUNIT_ASSERT_EQUAL((size_t)1, vSparseEnd.size());
        // Two features → two sparse entries
        CPPUNIT_ASSERT_EQUAL((size_t)2, vSparseIndex.size());
        CPPUNIT_ASSERT_EQUAL((size_t)2, vSparseData.size());
    }

    void TestImportSamplesFromPath_SingleFileMultipleSamples() {
        createSampleFile("samples.txt", {
            "customer1\tfeatureA,1.0:featureB,2.0",
            "customer2\tfeatureC,3.0",
            "customer3\tfeatureA,4.0:featureC,5.0"
        });

        unordered_map<string, unsigned int> mFeatureIndex;
        unordered_map<string, unsigned int> mSampleIndex;
        bool featureIndexUpdated = false;
        bool sampleIndexUpdated = false;
        vector<unsigned int> vSparseStart, vSparseEnd, vSparseIndex;
        vector<float> vSparseData;
        stringstream outputStream;

        bool result = importSamplesFromPath(
            tempDir + "/samples.txt", true,
            mFeatureIndex, mSampleIndex,
            featureIndexUpdated, sampleIndexUpdated,
            vSparseStart, vSparseEnd, vSparseIndex, vSparseData,
            outputStream);

        CPPUNIT_ASSERT(result);
        CPPUNIT_ASSERT_EQUAL((size_t)3, mSampleIndex.size());
        // featureA, featureB, featureC → 3 features
        CPPUNIT_ASSERT_EQUAL((size_t)3, mFeatureIndex.size());
        // 3 samples → 3 start/end entries
        CPPUNIT_ASSERT_EQUAL((size_t)3, vSparseStart.size());
        CPPUNIT_ASSERT_EQUAL((size_t)3, vSparseEnd.size());
        // Total: 2 + 1 + 2 = 5 sparse entries
        CPPUNIT_ASSERT_EQUAL((size_t)5, vSparseIndex.size());
        CPPUNIT_ASSERT_EQUAL((size_t)5, vSparseData.size());
    }

    void TestImportSamplesFromPath_SparseStartAndEndConsistency() {
        createSampleFile("samples.txt", {
            "s1\tfa,1.0:fb,2.0",
            "s2\tfc,3.0"
        });

        unordered_map<string, unsigned int> mFeatureIndex;
        unordered_map<string, unsigned int> mSampleIndex;
        bool featureIndexUpdated = false;
        bool sampleIndexUpdated = false;
        vector<unsigned int> vSparseStart, vSparseEnd, vSparseIndex;
        vector<float> vSparseData;
        stringstream outputStream;

        importSamplesFromPath(
            tempDir + "/samples.txt", true,
            mFeatureIndex, mSampleIndex,
            featureIndexUpdated, sampleIndexUpdated,
            vSparseStart, vSparseEnd, vSparseIndex, vSparseData,
            outputStream);

        // Verify that vSparseStart[0] == 0 and vSparseEnd[last] == total sparse entries
        CPPUNIT_ASSERT_EQUAL((unsigned int)0, vSparseStart[0]);
        CPPUNIT_ASSERT_EQUAL((unsigned int)vSparseIndex.size(), vSparseEnd.back());

        // Each sample's sparse data range should be consistent
        for (size_t i = 0; i < vSparseStart.size(); ++i) {
            CPPUNIT_ASSERT(vSparseStart[i] <= vSparseEnd[i]);
            CPPUNIT_ASSERT(vSparseEnd[i] <= vSparseIndex.size());
        }
    }

    void TestImportSamplesFromPath_DirectoryWithMultipleFiles() {
        string subDir = tempDir + "/datadir";
        mkdir(subDir.c_str(), 0755);

        ofstream f1(subDir + "/file1.txt");
        f1 << "customer1\tfeatureA,1.0\n";
        f1.close();

        ofstream f2(subDir + "/file2.txt");
        f2 << "customer2\tfeatureB,2.0\n";
        f2.close();

        unordered_map<string, unsigned int> mFeatureIndex;
        unordered_map<string, unsigned int> mSampleIndex;
        bool featureIndexUpdated = false;
        bool sampleIndexUpdated = false;
        vector<unsigned int> vSparseStart, vSparseEnd, vSparseIndex;
        vector<float> vSparseData;
        stringstream outputStream;

        bool result = importSamplesFromPath(
            subDir, true,
            mFeatureIndex, mSampleIndex,
            featureIndexUpdated, sampleIndexUpdated,
            vSparseStart, vSparseEnd, vSparseIndex, vSparseData,
            outputStream);

        CPPUNIT_ASSERT(result);
        CPPUNIT_ASSERT_EQUAL((size_t)2, mSampleIndex.size());
        CPPUNIT_ASSERT_EQUAL((size_t)2, mFeatureIndex.size());
    }

    void TestImportSamplesFromPath_EmptyFile() {
        createSampleFile("empty.txt", {});

        unordered_map<string, unsigned int> mFeatureIndex;
        unordered_map<string, unsigned int> mSampleIndex;
        bool featureIndexUpdated = false;
        bool sampleIndexUpdated = false;
        vector<unsigned int> vSparseStart, vSparseEnd, vSparseIndex;
        vector<float> vSparseData;
        stringstream outputStream;

        bool result = importSamplesFromPath(
            tempDir + "/empty.txt", true,
            mFeatureIndex, mSampleIndex,
            featureIndexUpdated, sampleIndexUpdated,
            vSparseStart, vSparseEnd, vSparseIndex, vSparseData,
            outputStream);

        CPPUNIT_ASSERT(result);
        CPPUNIT_ASSERT(!featureIndexUpdated);
        CPPUNIT_ASSERT(!sampleIndexUpdated);
        CPPUNIT_ASSERT(vSparseStart.empty());
        CPPUNIT_ASSERT(vSparseEnd.empty());
        CPPUNIT_ASSERT(vSparseIndex.empty());
    }

    void TestImportSamplesFromPath_FeatureIndexUpdatesDisabled() {
        createSampleFile("samples.txt", {"customer1\tfeatureA,1.0:featureB,2.0"});

        unordered_map<string, unsigned int> mFeatureIndex = {{"featureA", 0}};
        unordered_map<string, unsigned int> mSampleIndex;
        bool featureIndexUpdated = false;
        bool sampleIndexUpdated = false;
        vector<unsigned int> vSparseStart, vSparseEnd, vSparseIndex;
        vector<float> vSparseData;
        stringstream outputStream;

        bool result = importSamplesFromPath(
            tempDir + "/samples.txt", false,  // feature index updates disabled
            mFeatureIndex, mSampleIndex,
            featureIndexUpdated, sampleIndexUpdated,
            vSparseStart, vSparseEnd, vSparseIndex, vSparseData,
            outputStream);

        CPPUNIT_ASSERT(result);
        // featureB is unknown and index updates disabled → not added
        CPPUNIT_ASSERT_EQUAL((size_t)1, mFeatureIndex.size());
        CPPUNIT_ASSERT(!featureIndexUpdated);
        // customer1 is still indexed
        CPPUNIT_ASSERT(sampleIndexUpdated);
    }

    // ============== parseSamples edge cases ==============

    void TestParseSamples_EmptyLines() {
        // Lines that are empty should be skipped
        stringstream inputStream;
        inputStream << "\n";
        inputStream << "customer1\tfeatureA,1.0\n";
        inputStream << "\n";

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
        CPPUNIT_ASSERT_EQUAL((size_t)1, mSampleIndex.size());
        CPPUNIT_ASSERT_EQUAL((size_t)1, mFeatureIndex.size());
    }

    void TestParseSamples_DuplicateSample() {
        // Same sample appearing twice - second occurrence updates/overwrites signals
        stringstream inputStream;
        inputStream << "customer1\tfeatureA,1.0\n";
        inputStream << "customer1\tfeatureB,2.0\n";

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
        // Only one unique sample
        CPPUNIT_ASSERT_EQUAL((size_t)1, mSampleIndex.size());
        // Two features total
        CPPUNIT_ASSERT_EQUAL((size_t)2, mFeatureIndex.size());
    }

    void TestParseSamples_MultipleValuesForFeatureWarning() {
        // Data point with more than 1 value should generate a warning
        stringstream inputStream;
        inputStream << "customer1\tfeatureA,1.0,extra_value\n";

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
        CPPUNIT_ASSERT_MESSAGE("Warning should be emitted for extra values",
            outputStream.str().find("Warning") != string::npos);
        // Feature should still be added with first value
        CPPUNIT_ASSERT_EQUAL((size_t)1, mFeatureIndex.size());
    }

    void TestParseSamples_FeatureWithNoValue() {
        // Feature with no value uses default 0.0
        stringstream inputStream;
        inputStream << "customer1\tfeatureA\n";

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
        CPPUNIT_ASSERT_EQUAL((size_t)1, mFeatureIndex.size());
        // Signal value should default to 0.0
        CPPUNIT_ASSERT_EQUAL((size_t)1, mSignalValues[0].size());
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0f, mSignalValues[0][0], 1e-6f);
    }

    void TestParseSamples_ExistingFeatureIndexReused() {
        // When feature already exists in index, it should reuse its index
        stringstream inputStream;
        inputStream << "customer1\tfeatureA,1.0\n";

        unordered_map<string, unsigned int> mFeatureIndex = {{"featureA", 42}};
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
        CPPUNIT_ASSERT_EQUAL((size_t)1, mFeatureIndex.size());
        // feature index not updated (already existed)
        CPPUNIT_ASSERT(!featureIndexUpdated);
        // Signal index should use existing index 42
        CPPUNIT_ASSERT_EQUAL((size_t)1, mSignals[0].size());
        CPPUNIT_ASSERT_EQUAL(42u, mSignals[0][0]);
    }

    void TestParseSamples_SignalValuesMatchSignalIndices() {
        stringstream inputStream;
        inputStream << "customer1\tfeatureA,1.5:featureB,2.5:featureC,3.5\n";

        unordered_map<string, unsigned int> mFeatureIndex;
        unordered_map<string, unsigned int> mSampleIndex;
        bool featureIndexUpdated = false;
        bool sampleIndexUpdated = false;
        map<unsigned int, vector<unsigned int>> mSignals;
        map<unsigned int, vector<float>> mSignalValues;
        stringstream outputStream;

        parseSamples(inputStream, true, mFeatureIndex, mSampleIndex,
                     featureIndexUpdated, sampleIndexUpdated,
                     mSignals, mSignalValues, outputStream);

        CPPUNIT_ASSERT_EQUAL((size_t)3, mSignals[0].size());
        CPPUNIT_ASSERT_EQUAL((size_t)3, mSignalValues[0].size());
        // Values should match what we put in
        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.5f, mSignalValues[0][0], 1e-6f);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(2.5f, mSignalValues[0][1], 1e-6f);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(3.5f, mSignalValues[0][2], 1e-6f);
    }

    CPPUNIT_TEST_SUITE(TestImportSamples);

    // importSamplesFromPath tests
    CPPUNIT_TEST(TestImportSamplesFromPath_NonExistentPath);
    CPPUNIT_TEST(TestImportSamplesFromPath_SingleFileSingleSample);
    CPPUNIT_TEST(TestImportSamplesFromPath_SingleFileMultipleSamples);
    CPPUNIT_TEST(TestImportSamplesFromPath_SparseStartAndEndConsistency);
    CPPUNIT_TEST(TestImportSamplesFromPath_DirectoryWithMultipleFiles);
    CPPUNIT_TEST(TestImportSamplesFromPath_EmptyFile);
    CPPUNIT_TEST(TestImportSamplesFromPath_FeatureIndexUpdatesDisabled);

    // parseSamples edge cases
    CPPUNIT_TEST(TestParseSamples_EmptyLines);
    CPPUNIT_TEST(TestParseSamples_DuplicateSample);
    CPPUNIT_TEST(TestParseSamples_MultipleValuesForFeatureWarning);
    CPPUNIT_TEST(TestParseSamples_FeatureWithNoValue);
    CPPUNIT_TEST(TestParseSamples_ExistingFeatureIndexReused);
    CPPUNIT_TEST(TestParseSamples_SignalValuesMatchSignalIndices);

    CPPUNIT_TEST_SUITE_END();
};

#include <cstdio>
#include <fstream>
#include <map>
#include <string>
#include <sstream>
#include <unordered_map>
#include <vector>

#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/ui/text/TestRunner.h>
#include <cppunit/TestAssert.h>

#include "NetCDFhelper.h"

using namespace std;

class TestNetCDFhelper : public CppUnit::TestFixture
{
    const static map<string, unsigned int> validFeatureIndex;

public:
    void TestLoadIndexWithValidInput() {
        // Seed the input stream with valid input
        stringstream inputStream;
        for (const auto &entry : validFeatureIndex) {
            inputStream << entry.first << "\t" << entry.second << "\n";
        }

        unordered_map<string, unsigned int> labelsToIndices;
        stringstream outputStream;
        CPPUNIT_ASSERT(loadIndex(labelsToIndices, inputStream, outputStream));
        CPPUNIT_ASSERT_MESSAGE("Output stream should contain no error messages", 
            outputStream.str().find("Error") == string::npos);
        CPPUNIT_ASSERT_MESSAGE("Number of entries in feature index should be equal to number of lines of input",
            validFeatureIndex.size() == labelsToIndices.size());

        // Check that all of the feature map entries have been correctly loaded
        for (const auto &entry : validFeatureIndex) {
            const auto itr = labelsToIndices.find(entry.first);
            CPPUNIT_ASSERT_MESSAGE("Each feature label from input should be present in feature index",
                itr != labelsToIndices.end());
            CPPUNIT_ASSERT_MESSAGE("Each feature index from input should be present in feature index",
                entry.second == itr->second);
        }
    }

    void TestLoadIndexWithDuplicateEntry() {
        // Seed the input stream with valid input
        stringstream inputStream;
        for (const auto &entry : validFeatureIndex) {
            inputStream << entry.first << "\t" << entry.second << "\n";
        }

        // Duplicate first entry
        const auto itr = validFeatureIndex.begin();
        inputStream << validFeatureIndex.begin()->first << "\t" << itr->second << "\n";

        unordered_map<string, unsigned int> labelsToIndices;
        stringstream outputStream;
        CPPUNIT_ASSERT(!loadIndex(labelsToIndices, inputStream, outputStream));
        CPPUNIT_ASSERT_MESSAGE("Output stream should contain an error message", 
            outputStream.str().find("Error") != string::npos);
    }

    void TestLoadIndexWithDuplicateLabelOnly() {
        // Seed the input stream with valid input
        stringstream inputStream;
        for (const auto &entry : validFeatureIndex) {
            inputStream << entry.first << "\t" << entry.second << "\n";
        }

        // Duplicate just the label used in the first entry
        inputStream << validFeatureIndex.begin()->first << "\t123\n";

        unordered_map<string, unsigned int> labelsToIndices;
        stringstream outputStream;
        CPPUNIT_ASSERT(!loadIndex(labelsToIndices, inputStream, outputStream));
        CPPUNIT_ASSERT_MESSAGE("Output stream should contain an error message", 
            outputStream.str().find("Error") != string::npos);
    }

    void TestLoadIndexWithMissingLabel() {
        stringstream inputStream;
        inputStream << "\t123\n";
        unordered_map<string, unsigned int> labelsToIndices;
        stringstream outputStream;
        CPPUNIT_ASSERT(!loadIndex(labelsToIndices, inputStream, outputStream));
        CPPUNIT_ASSERT_MESSAGE("Output stream should contain an error message", 
            outputStream.str().find("Error") != string::npos);
    }

    void TestLoadIndexWithMissingLabelAndTab() {
        stringstream inputStream;
        inputStream << "123\n";
        unordered_map<string, unsigned int> labelsToIndices;
        stringstream outputStream;
        CPPUNIT_ASSERT(!loadIndex(labelsToIndices, inputStream, outputStream));
        CPPUNIT_ASSERT_MESSAGE("Output stream should contain an error message", 
            outputStream.str().find("Error") != string::npos);
    }

    void TestLoadIndexWithExtraTab() {
        stringstream inputStream;
        inputStream << "110510\t123\t121017\n";
        unordered_map<string, unsigned int> labelsToIndices;
        stringstream outputStream;
        CPPUNIT_ASSERT(!loadIndex(labelsToIndices, inputStream, outputStream));
        CPPUNIT_ASSERT_MESSAGE("Output stream should contain an error message", 
            outputStream.str().find("Error") != string::npos);
    }

    // -------------------------------------------------------------------------
    // roundUpMaxIndex
    // -------------------------------------------------------------------------

    void TestRoundUpMaxIndex_Zero()
    {
        CPPUNIT_ASSERT_EQUAL(0u, roundUpMaxIndex(0u));
    }

    void TestRoundUpMaxIndex_One()
    {
        // 1 should round up to 128
        CPPUNIT_ASSERT_EQUAL(128u, roundUpMaxIndex(1u));
    }

    void TestRoundUpMaxIndex_AlreadyMultiple()
    {
        CPPUNIT_ASSERT_EQUAL(128u, roundUpMaxIndex(128u));
        CPPUNIT_ASSERT_EQUAL(256u, roundUpMaxIndex(256u));
        CPPUNIT_ASSERT_EQUAL(512u, roundUpMaxIndex(512u));
    }

    void TestRoundUpMaxIndex_NeedsRounding()
    {
        CPPUNIT_ASSERT_EQUAL(128u, roundUpMaxIndex(64u));
        CPPUNIT_ASSERT_EQUAL(128u, roundUpMaxIndex(127u));
        CPPUNIT_ASSERT_EQUAL(256u, roundUpMaxIndex(129u));
        CPPUNIT_ASSERT_EQUAL(256u, roundUpMaxIndex(255u));
    }

    // -------------------------------------------------------------------------
    // parseSamples
    // -------------------------------------------------------------------------

    void TestParseSamples_EmptyStream()
    {
        stringstream inputStream;
        unordered_map<string, unsigned int> mFeatureIndex;
        unordered_map<string, unsigned int> mSampleIndex;
        bool featureIndexUpdated = false;
        bool sampleIndexUpdated  = false;
        map<unsigned int, vector<unsigned int>> mSignals;
        map<unsigned int, vector<float>> mSignalValues;
        stringstream outputStream;

        bool result = parseSamples(inputStream, true, mFeatureIndex, mSampleIndex,
                                   featureIndexUpdated, sampleIndexUpdated,
                                   mSignals, mSignalValues, outputStream);

        CPPUNIT_ASSERT(result);
        CPPUNIT_ASSERT(!featureIndexUpdated);
        CPPUNIT_ASSERT(!sampleIndexUpdated);
        CPPUNIT_ASSERT(mSignals.empty());
    }

    void TestParseSamples_ValidData()
    {
        stringstream inputStream;
        inputStream << "customer1\tfeature1,1.0:feature2,2.0\n";

        unordered_map<string, unsigned int> mFeatureIndex;
        unordered_map<string, unsigned int> mSampleIndex;
        bool featureIndexUpdated = false;
        bool sampleIndexUpdated  = false;
        map<unsigned int, vector<unsigned int>> mSignals;
        map<unsigned int, vector<float>> mSignalValues;
        stringstream outputStream;

        bool result = parseSamples(inputStream, true, mFeatureIndex, mSampleIndex,
                                   featureIndexUpdated, sampleIndexUpdated,
                                   mSignals, mSignalValues, outputStream);

        CPPUNIT_ASSERT(result);
        CPPUNIT_ASSERT(sampleIndexUpdated);
        CPPUNIT_ASSERT(featureIndexUpdated);
        CPPUNIT_ASSERT_EQUAL((size_t) 1, mSampleIndex.size());
        CPPUNIT_ASSERT_EQUAL((size_t) 2, mFeatureIndex.size());
        CPPUNIT_ASSERT_EQUAL((size_t) 1, mSignals.size());
        // sample 0 should have 2 features
        CPPUNIT_ASSERT_EQUAL((size_t) 2, mSignals[0].size());
        CPPUNIT_ASSERT_EQUAL((size_t) 2, mSignalValues[0].size());
        CPPUNIT_ASSERT_EQUAL(1.0f, mSignalValues[0][0]);
        CPPUNIT_ASSERT_EQUAL(2.0f, mSignalValues[0][1]);
    }

    void TestParseSamples_MalformedLine_NoTab()
    {
        // Line has no tab - should generate a Warning and be skipped, not fail
        stringstream inputStream;
        inputStream << "malformed_line_without_tab\n";

        unordered_map<string, unsigned int> mFeatureIndex;
        unordered_map<string, unsigned int> mSampleIndex;
        bool featureIndexUpdated = false;
        bool sampleIndexUpdated  = false;
        map<unsigned int, vector<unsigned int>> mSignals;
        map<unsigned int, vector<float>> mSignalValues;
        stringstream outputStream;

        bool result = parseSamples(inputStream, true, mFeatureIndex, mSampleIndex,
                                   featureIndexUpdated, sampleIndexUpdated,
                                   mSignals, mSignalValues, outputStream);

        CPPUNIT_ASSERT(result);
        CPPUNIT_ASSERT(!sampleIndexUpdated);
        CPPUNIT_ASSERT_MESSAGE("Output should contain a Warning",
            outputStream.str().find("Warning") != string::npos);
    }

    void TestParseSamples_FeatureIndexUpdatesDisabled()
    {
        // With enableFeatureIndexUpdates=false, unknown features are ignored
        stringstream inputStream;
        inputStream << "customer1\tfeature1,1.0\n";

        unordered_map<string, unsigned int> mFeatureIndex;
        unordered_map<string, unsigned int> mSampleIndex;
        bool featureIndexUpdated = false;
        bool sampleIndexUpdated  = false;
        map<unsigned int, vector<unsigned int>> mSignals;
        map<unsigned int, vector<float>> mSignalValues;
        stringstream outputStream;

        bool result = parseSamples(inputStream, false, mFeatureIndex, mSampleIndex,
                                   featureIndexUpdated, sampleIndexUpdated,
                                   mSignals, mSignalValues, outputStream);

        CPPUNIT_ASSERT(result);
        CPPUNIT_ASSERT(!featureIndexUpdated);
        CPPUNIT_ASSERT(sampleIndexUpdated);
        CPPUNIT_ASSERT_EQUAL((size_t) 0, mFeatureIndex.size());
    }

    // -------------------------------------------------------------------------
    // loadIndexFromFile
    // -------------------------------------------------------------------------

    void TestLoadIndexFromFile_ValidFile()
    {
        const string tempFile = "/tmp/test_dsstne_loadindex_valid.txt";
        {
            ofstream ofs(tempFile);
            ofs << "feature1\t0\n";
            ofs << "feature2\t1\n";
            ofs << "feature3\t2\n";
        }

        unordered_map<string, unsigned int> labelsToIndices;
        stringstream outputStream;
        bool result = loadIndexFromFile(labelsToIndices, tempFile, outputStream);

        remove(tempFile.c_str());

        CPPUNIT_ASSERT(result);
        CPPUNIT_ASSERT_EQUAL((size_t) 3, labelsToIndices.size());
        CPPUNIT_ASSERT_EQUAL(0u, labelsToIndices["feature1"]);
        CPPUNIT_ASSERT_EQUAL(1u, labelsToIndices["feature2"]);
        CPPUNIT_ASSERT_EQUAL(2u, labelsToIndices["feature3"]);
    }

    void TestLoadIndexFromFile_EmptyFile()
    {
        const string tempFile = "/tmp/test_dsstne_loadindex_empty.txt";
        {
            ofstream ofs(tempFile);
            // intentionally empty
        }

        unordered_map<string, unsigned int> labelsToIndices;
        stringstream outputStream;
        bool result = loadIndexFromFile(labelsToIndices, tempFile, outputStream);

        remove(tempFile.c_str());

        CPPUNIT_ASSERT(result);
        CPPUNIT_ASSERT(labelsToIndices.empty());
    }

    void TestLoadIndexFromFile_NonExistentFile()
    {
        unordered_map<string, unsigned int> labelsToIndices;
        stringstream outputStream;
        bool result = loadIndexFromFile(labelsToIndices, "/tmp/dsstne_no_such_file.txt", outputStream);

        CPPUNIT_ASSERT(!result);
        CPPUNIT_ASSERT_MESSAGE("Output should contain an error message",
            outputStream.str().find("Error") != string::npos);
    }

    // -------------------------------------------------------------------------
    // exportIndex (round-trip)
    // -------------------------------------------------------------------------

    void TestExportIndex_RoundTrip()
    {
        unordered_map<string, unsigned int> original;
        original["apple"]  = 1;
        original["banana"] = 2;
        original["cherry"] = 3;

        const string tempFile = "/tmp/test_dsstne_exportindex.txt";
        exportIndex(original, tempFile);

        unordered_map<string, unsigned int> loaded;
        stringstream outputStream;
        bool result = loadIndexFromFile(loaded, tempFile, outputStream);

        remove(tempFile.c_str());

        CPPUNIT_ASSERT(result);
        CPPUNIT_ASSERT_EQUAL(original.size(), loaded.size());
        for (const auto &entry : original)
        {
            const auto it = loaded.find(entry.first);
            CPPUNIT_ASSERT_MESSAGE("Exported key should be present after reload",
                it != loaded.end());
            CPPUNIT_ASSERT_EQUAL(entry.second, it->second);
        }
    }


    CPPUNIT_TEST_SUITE(TestNetCDFhelper);
    CPPUNIT_TEST(TestLoadIndexWithValidInput);
    CPPUNIT_TEST(TestLoadIndexWithDuplicateEntry);
    CPPUNIT_TEST(TestLoadIndexWithDuplicateLabelOnly);
    CPPUNIT_TEST(TestLoadIndexWithMissingLabel);
    CPPUNIT_TEST(TestLoadIndexWithMissingLabelAndTab);
    CPPUNIT_TEST(TestLoadIndexWithExtraTab);
    CPPUNIT_TEST(TestRoundUpMaxIndex_Zero);
    CPPUNIT_TEST(TestRoundUpMaxIndex_One);
    CPPUNIT_TEST(TestRoundUpMaxIndex_AlreadyMultiple);
    CPPUNIT_TEST(TestRoundUpMaxIndex_NeedsRounding);
    CPPUNIT_TEST(TestParseSamples_EmptyStream);
    CPPUNIT_TEST(TestParseSamples_ValidData);
    CPPUNIT_TEST(TestParseSamples_MalformedLine_NoTab);
    CPPUNIT_TEST(TestParseSamples_FeatureIndexUpdatesDisabled);
    CPPUNIT_TEST(TestLoadIndexFromFile_ValidFile);
    CPPUNIT_TEST(TestLoadIndexFromFile_EmptyFile);
    CPPUNIT_TEST(TestLoadIndexFromFile_NonExistentFile);
    CPPUNIT_TEST(TestExportIndex_RoundTrip);
    CPPUNIT_TEST_SUITE_END();
};

const map<string, unsigned int> TestNetCDFhelper::validFeatureIndex = {
    { "110510", 26743 },
    { "121019", 26740 },
    { "121017", 26739 },
    { "106401", 26736 },
    { "104307", 26734 }};

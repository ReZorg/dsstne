#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/ui/text/TestRunner.h>
#include <cppunit/TestAssert.h>

#include <string>
#include <vector>

#include "Utils.h"

class TestUtils : public CppUnit::TestFixture
{
public:
    // -------------------------------------------------------------------------
    // isNetCDFfile
    // -------------------------------------------------------------------------

    void TestIsNetCDFfile()
    {
        CPPUNIT_ASSERT(isNetCDFfile("network.nc"));
        CPPUNIT_ASSERT(!isNetCDFfile("network.nic"));
    }

    void TestIsNetCDFfile_EmptyString()
    {
        CPPUNIT_ASSERT(!isNetCDFfile(""));
    }

    void TestIsNetCDFfile_NoExtension()
    {
        CPPUNIT_ASSERT(!isNetCDFfile("network"));
    }

    void TestIsNetCDFfile_DoubleExtension()
    {
        // last extension is .txt, not .nc
        CPPUNIT_ASSERT(!isNetCDFfile("network.nc.txt"));
    }

    void TestIsNetCDFfile_JustExtension()
    {
        // filename that is only an extension
        CPPUNIT_ASSERT(isNetCDFfile(".nc"));
    }

    // -------------------------------------------------------------------------
    // split
    // -------------------------------------------------------------------------

    void TestSplit_BasicSplit()
    {
        vector<string> result = split("a,b,c", ',');
        CPPUNIT_ASSERT_EQUAL((size_t) 3, result.size());
        CPPUNIT_ASSERT_EQUAL(string("a"), result[0]);
        CPPUNIT_ASSERT_EQUAL(string("b"), result[1]);
        CPPUNIT_ASSERT_EQUAL(string("c"), result[2]);
    }

    void TestSplit_EmptyString()
    {
        vector<string> result = split("", ',');
        CPPUNIT_ASSERT_EQUAL((size_t) 0, result.size());
    }

    void TestSplit_NoDelimiter()
    {
        vector<string> result = split("abc", ',');
        CPPUNIT_ASSERT_EQUAL((size_t) 1, result.size());
        CPPUNIT_ASSERT_EQUAL(string("abc"), result[0]);
    }

    void TestSplit_TabDelimiter()
    {
        vector<string> result = split("hello\tworld", '\t');
        CPPUNIT_ASSERT_EQUAL((size_t) 2, result.size());
        CPPUNIT_ASSERT_EQUAL(string("hello"), result[0]);
        CPPUNIT_ASSERT_EQUAL(string("world"), result[1]);
    }

    void TestSplit_ConsecutiveDelimiters()
    {
        // middle element should be empty
        vector<string> result = split("a,,b", ',');
        CPPUNIT_ASSERT_EQUAL((size_t) 3, result.size());
        CPPUNIT_ASSERT_EQUAL(string("a"), result[0]);
        CPPUNIT_ASSERT_EQUAL(string(""), result[1]);
        CPPUNIT_ASSERT_EQUAL(string("b"), result[2]);
    }

    void TestSplit_ColonDelimiter()
    {
        vector<string> result = split("feat1,1.0:feat2,2.0", ':');
        CPPUNIT_ASSERT_EQUAL((size_t) 2, result.size());
        CPPUNIT_ASSERT_EQUAL(string("feat1,1.0"), result[0]);
        CPPUNIT_ASSERT_EQUAL(string("feat2,2.0"), result[1]);
    }

    // -------------------------------------------------------------------------
    // getCmdOption
    // -------------------------------------------------------------------------

    void TestGetCmdOption_Present()
    {
        char arg0[] = "program";
        char flag[] = "-f";
        char value[] = "myfile.nc";
        char* argv[] = {arg0, flag, value};

        char* result = getCmdOption(argv, argv + 3, "-f");
        CPPUNIT_ASSERT(result != nullptr);
        CPPUNIT_ASSERT_EQUAL(string("myfile.nc"), string(result));
    }

    void TestGetCmdOption_Absent()
    {
        char arg0[] = "program";
        char* argv[] = {arg0};

        char* result = getCmdOption(argv, argv + 1, "-f");
        CPPUNIT_ASSERT(result == nullptr);
    }

    void TestGetCmdOption_FlagIsLastArg()
    {
        // flag exists but has no following value
        char arg0[] = "program";
        char flag[] = "-f";
        char* argv[] = {arg0, flag};

        char* result = getCmdOption(argv, argv + 2, "-f");
        CPPUNIT_ASSERT(result == nullptr);
    }

    // -------------------------------------------------------------------------
    // cmdOptionExists
    // -------------------------------------------------------------------------

    void TestCmdOptionExists_Present()
    {
        char arg0[] = "program";
        char flag[] = "-v";
        char* argv[] = {arg0, flag};

        CPPUNIT_ASSERT(cmdOptionExists(argv, argv + 2, "-v"));
    }

    void TestCmdOptionExists_Absent()
    {
        char arg0[] = "program";
        char* argv[] = {arg0};

        CPPUNIT_ASSERT(!cmdOptionExists(argv, argv + 1, "-v"));
    }

    // -------------------------------------------------------------------------
    // getOptionalArgValue
    // -------------------------------------------------------------------------

    void TestGetOptionalArgValue_Present()
    {
        char arg0[] = "program";
        char flag[] = "-f";
        char value[] = "custom.nc";
        char* argv[] = {arg0, flag, value};

        string result = getOptionalArgValue(3, argv, "-f", "default.nc");
        CPPUNIT_ASSERT_EQUAL(string("custom.nc"), result);
    }

    void TestGetOptionalArgValue_Absent()
    {
        char arg0[] = "program";
        char* argv[] = {arg0};

        string result = getOptionalArgValue(1, argv, "-f", "default.nc");
        CPPUNIT_ASSERT_EQUAL(string("default.nc"), result);
    }

    // -------------------------------------------------------------------------
    // isArgSet
    // -------------------------------------------------------------------------

    void TestIsArgSet_Set()
    {
        char arg0[] = "program";
        char flag[] = "-v";
        char* argv[] = {arg0, flag};

        CPPUNIT_ASSERT(isArgSet(2, argv, "-v"));
    }

    void TestIsArgSet_NotSet()
    {
        char arg0[] = "program";
        char* argv[] = {arg0};

        CPPUNIT_ASSERT(!isArgSet(1, argv, "-v"));
    }

    // -------------------------------------------------------------------------
    // topKsort
    // -------------------------------------------------------------------------

    void TestTopKsort_ByKey()
    {
        float keys[]         = {3.0f, 1.0f, 4.0f, 1.0f, 5.0f, 9.0f, 2.0f};
        unsigned int vals[]  = {   0,    1,    2,    3,    4,    5,    6  };
        const int size = 7;
        const int topK = 3;
        float topKkeys[topK];
        unsigned int topKvals[topK];

        topKsort(keys, vals, size, topKkeys, topKvals, topK, true);

        // Top-3 descending by key: 9, 5, 4
        CPPUNIT_ASSERT_EQUAL(9.0f, topKkeys[0]);
        CPPUNIT_ASSERT_EQUAL(5.0f, topKkeys[1]);
        CPPUNIT_ASSERT_EQUAL(4.0f, topKkeys[2]);
    }

    void TestTopKsort_ByValue()
    {
        float keys[] = {1.0f, 2.0f, 3.0f};
        float vals[] = {30.0f, 10.0f, 20.0f};
        const int size = 3;
        const int topK = 2;
        float topKkeys[topK];
        float topKvals[topK];

        topKsort(keys, vals, size, topKkeys, topKvals, topK, false);

        // Top-2 descending by value: 30 (key=1), 20 (key=3)
        CPPUNIT_ASSERT_EQUAL(30.0f, topKvals[0]);
        CPPUNIT_ASSERT_EQUAL(20.0f, topKvals[1]);
    }

    void TestTopKsort_TopKEqualsSize()
    {
        float keys[]        = {3.0f, 1.0f, 2.0f};
        unsigned int vals[] = {   0,    1,    2 };
        const int size = 3;
        const int topK = 3;
        float topKkeys[topK];
        unsigned int topKvals[topK];

        topKsort(keys, vals, size, topKkeys, topKvals, topK, true);

        // All elements, sorted descending by key: 3, 2, 1
        CPPUNIT_ASSERT_EQUAL(3.0f, topKkeys[0]);
        CPPUNIT_ASSERT_EQUAL(2.0f, topKkeys[1]);
        CPPUNIT_ASSERT_EQUAL(1.0f, topKkeys[2]);
    }

    CPPUNIT_TEST_SUITE(TestUtils);
    CPPUNIT_TEST(TestIsNetCDFfile);
    CPPUNIT_TEST(TestIsNetCDFfile_EmptyString);
    CPPUNIT_TEST(TestIsNetCDFfile_NoExtension);
    CPPUNIT_TEST(TestIsNetCDFfile_DoubleExtension);
    CPPUNIT_TEST(TestIsNetCDFfile_JustExtension);
    CPPUNIT_TEST(TestSplit_BasicSplit);
    CPPUNIT_TEST(TestSplit_EmptyString);
    CPPUNIT_TEST(TestSplit_NoDelimiter);
    CPPUNIT_TEST(TestSplit_TabDelimiter);
    CPPUNIT_TEST(TestSplit_ConsecutiveDelimiters);
    CPPUNIT_TEST(TestSplit_ColonDelimiter);
    CPPUNIT_TEST(TestGetCmdOption_Present);
    CPPUNIT_TEST(TestGetCmdOption_Absent);
    CPPUNIT_TEST(TestGetCmdOption_FlagIsLastArg);
    CPPUNIT_TEST(TestCmdOptionExists_Present);
    CPPUNIT_TEST(TestCmdOptionExists_Absent);
    CPPUNIT_TEST(TestGetOptionalArgValue_Present);
    CPPUNIT_TEST(TestGetOptionalArgValue_Absent);
    CPPUNIT_TEST(TestIsArgSet_Set);
    CPPUNIT_TEST(TestIsArgSet_NotSet);
    CPPUNIT_TEST(TestTopKsort_ByKey);
    CPPUNIT_TEST(TestTopKsort_ByValue);
    CPPUNIT_TEST(TestTopKsort_TopKEqualsSize);
    CPPUNIT_TEST_SUITE_END();
};

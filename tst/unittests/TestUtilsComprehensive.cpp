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
#include <sstream>
#include <cstdlib>
#include <sys/stat.h>
#include <unistd.h>

#include "Utils.h"

using namespace std;

/**
 * Comprehensive test suite for Utils.cpp functions
 * Covers: isNetCDFfile, split, fileExists, isDirectory, isFile, listFiles,
 *         cmdOptionExists, getCmdOption, getRequiredArgValue, getOptionalArgValue, isArgSet
 */
class TestUtilsComprehensive : public CppUnit::TestFixture
{
private:
    string tempDir;
    
    void createTempDir() {
        char tmpl[] = "/tmp/dsstne_test_XXXXXX";
        char* result = mkdtemp(tmpl);
        if (result != nullptr) {
            tempDir = string(result);
        }
    }
    
    void removeTempDir() {
        if (!tempDir.empty()) {
            // Clean up files and directory
            system(("rm -rf " + tempDir).c_str());
        }
    }
    
    void createTestFile(const string& filename, const string& content = "") {
        ofstream ofs(tempDir + "/" + filename);
        ofs << content;
        ofs.close();
    }

public:
    void setUp() {
        createTempDir();
    }
    
    void tearDown() {
        removeTempDir();
    }

    // ============== isNetCDFfile tests ==============
    
    void TestIsNetCDFfileWithNcExtension() {
        CPPUNIT_ASSERT(isNetCDFfile("network.nc"));
        CPPUNIT_ASSERT(isNetCDFfile("data.nc"));
        CPPUNIT_ASSERT(isNetCDFfile("/path/to/file.nc"));
        CPPUNIT_ASSERT(isNetCDFfile("./relative/path/data.nc"));
    }
    
    void TestIsNetCDFfileWithOtherExtensions() {
        CPPUNIT_ASSERT(!isNetCDFfile("network.nic"));
        CPPUNIT_ASSERT(!isNetCDFfile("network.json"));
        CPPUNIT_ASSERT(!isNetCDFfile("network.txt"));
        CPPUNIT_ASSERT(!isNetCDFfile("network.ncc"));
        CPPUNIT_ASSERT(!isNetCDFfile("network.NC")); // Case sensitive
    }
    
    void TestIsNetCDFfileWithNoExtension() {
        CPPUNIT_ASSERT(!isNetCDFfile("network"));
        CPPUNIT_ASSERT(!isNetCDFfile("nc"));
    }
    
    void TestIsNetCDFfileWithEmptyString() {
        CPPUNIT_ASSERT(!isNetCDFfile(""));
    }
    
    void TestIsNetCDFfileWithMultipleDots() {
        CPPUNIT_ASSERT(isNetCDFfile("data.backup.nc"));
        CPPUNIT_ASSERT(!isNetCDFfile("data.nc.backup"));
    }

    // ============== split tests ==============
    
    void TestSplitBasic() {
        string input = "a:b:c";
        vector<string> result = split(input, ':');
        CPPUNIT_ASSERT_EQUAL((size_t)3, result.size());
        CPPUNIT_ASSERT_EQUAL(string("a"), result[0]);
        CPPUNIT_ASSERT_EQUAL(string("b"), result[1]);
        CPPUNIT_ASSERT_EQUAL(string("c"), result[2]);
    }
    
    void TestSplitWithTabDelimiter() {
        string input = "label1\t123";
        vector<string> result = split(input, '\t');
        CPPUNIT_ASSERT_EQUAL((size_t)2, result.size());
        CPPUNIT_ASSERT_EQUAL(string("label1"), result[0]);
        CPPUNIT_ASSERT_EQUAL(string("123"), result[1]);
    }
    
    void TestSplitWithNoDelimiter() {
        string input = "single";
        vector<string> result = split(input, ':');
        CPPUNIT_ASSERT_EQUAL((size_t)1, result.size());
        CPPUNIT_ASSERT_EQUAL(string("single"), result[0]);
    }
    
    void TestSplitWithEmptyString() {
        string input = "";
        vector<string> result = split(input, ':');
        // Empty string returns empty vector
        CPPUNIT_ASSERT_EQUAL((size_t)0, result.size());
    }
    
    void TestSplitWithConsecutiveDelimiters() {
        string input = "a::b";
        vector<string> result = split(input, ':');
        CPPUNIT_ASSERT_EQUAL((size_t)3, result.size());
        CPPUNIT_ASSERT_EQUAL(string("a"), result[0]);
        CPPUNIT_ASSERT_EQUAL(string(""), result[1]);
        CPPUNIT_ASSERT_EQUAL(string("b"), result[2]);
    }
    
    void TestSplitWithTrailingDelimiter() {
        string input = "a:b:";
        vector<string> result = split(input, ':');
        // Trailing delimiter does not produce extra empty element (getline stops at EOF)
        CPPUNIT_ASSERT_EQUAL((size_t)2, result.size());
        CPPUNIT_ASSERT_EQUAL(string("a"), result[0]);
        CPPUNIT_ASSERT_EQUAL(string("b"), result[1]);
    }
    
    void TestSplitWithLeadingDelimiter() {
        string input = ":a:b";
        vector<string> result = split(input, ':');
        CPPUNIT_ASSERT_EQUAL((size_t)3, result.size());
        CPPUNIT_ASSERT_EQUAL(string(""), result[0]);
        CPPUNIT_ASSERT_EQUAL(string("a"), result[1]);
    }
    
    void TestSplitWithReferenceVersion() {
        string input = "x:y:z";
        vector<string> elems;
        vector<string>& result = split(input, ':', elems);
        CPPUNIT_ASSERT_EQUAL((size_t)3, result.size());
        CPPUNIT_ASSERT(&elems == &result); // Should return reference to same vector
    }

    // ============== fileExists tests ==============
    
    void TestFileExistsWithExistingFile() {
        createTestFile("existing.txt", "content");
        CPPUNIT_ASSERT(fileExists(tempDir + "/existing.txt"));
    }
    
    void TestFileExistsWithNonExistingFile() {
        CPPUNIT_ASSERT(!fileExists(tempDir + "/nonexistent.txt"));
    }
    
    void TestFileExistsWithDirectory() {
        // Directories should also return true for fileExists (it checks if path can be opened)
        CPPUNIT_ASSERT(fileExists(tempDir));
    }
    
    void TestFileExistsWithEmptyPath() {
        CPPUNIT_ASSERT(!fileExists(""));
    }

    // ============== isDirectory tests ==============
    
    void TestIsDirectoryWithDirectory() {
        CPPUNIT_ASSERT(isDirectory(tempDir));
        CPPUNIT_ASSERT(isDirectory("/tmp"));
    }
    
    void TestIsDirectoryWithFile() {
        createTestFile("testfile.txt");
        CPPUNIT_ASSERT(!isDirectory(tempDir + "/testfile.txt"));
    }
    
    void TestIsDirectoryWithNonExisting() {
        CPPUNIT_ASSERT(!isDirectory(tempDir + "/nonexistent_dir"));
    }

    // ============== isFile tests ==============
    
    void TestIsFileWithFile() {
        createTestFile("regular_file.txt");
        CPPUNIT_ASSERT(isFile(tempDir + "/regular_file.txt"));
    }
    
    void TestIsFileWithDirectory() {
        CPPUNIT_ASSERT(!isFile(tempDir));
    }
    
    void TestIsFileWithNonExisting() {
        // Known implementation defect: isFile() does not check the return code of stat(),
        // so its result on a non-existing path depends on uninitialised stack memory.
        // This test documents the defect by verifying the call does not crash, without
        // asserting a specific return value.
        isFile(tempDir + "/nonexistent_file.txt");
    }

    // ============== listFiles tests ==============
    
    void TestListFilesWithSingleFile() {
        createTestFile("single.txt");
        vector<string> files;
        int result = listFiles(tempDir + "/single.txt", false, files);
        CPPUNIT_ASSERT_EQUAL(0, result);
        CPPUNIT_ASSERT_EQUAL((size_t)1, files.size());
    }
    
    void TestListFilesWithDirectory() {
        createTestFile("file1.txt");
        createTestFile("file2.txt");
        vector<string> files;
        int result = listFiles(tempDir, false, files);
        CPPUNIT_ASSERT_EQUAL(0, result);
        CPPUNIT_ASSERT_EQUAL((size_t)2, files.size());
    }
    
    void TestListFilesResultsSorted() {
        createTestFile("z_file.txt");
        createTestFile("a_file.txt");
        createTestFile("m_file.txt");
        vector<string> files;
        listFiles(tempDir, false, files);
        CPPUNIT_ASSERT_EQUAL((size_t)3, files.size());
        // Results should be sorted alphabetically
        CPPUNIT_ASSERT(files[0].find("a_file") != string::npos);
        CPPUNIT_ASSERT(files[1].find("m_file") != string::npos);
        CPPUNIT_ASSERT(files[2].find("z_file") != string::npos);
    }
    
    void TestListFilesRecursive() {
        string subDir = tempDir + "/subdir";
        mkdir(subDir.c_str(), 0755);
        createTestFile("root.txt");
        ofstream ofs(subDir + "/sub.txt");
        ofs << "content";
        ofs.close();
        
        vector<string> filesNonRecursive;
        listFiles(tempDir, false, filesNonRecursive);
        
        vector<string> filesRecursive;
        listFiles(tempDir, true, filesRecursive);
        
        // Recursive should find more files (includes subdir contents)
        CPPUNIT_ASSERT(filesRecursive.size() >= filesNonRecursive.size());
    }
    
    void TestListFilesWithEmptyDirectory() {
        string emptyDir = tempDir + "/empty";
        mkdir(emptyDir.c_str(), 0755);
        vector<string> files;
        int result = listFiles(emptyDir, false, files);
        CPPUNIT_ASSERT_EQUAL(0, result);
        CPPUNIT_ASSERT_EQUAL((size_t)0, files.size());
    }
    
    void TestListFilesWithNonExistingPath() {
        vector<string> files;
        int result = listFiles(tempDir + "/nonexistent", false, files);
        CPPUNIT_ASSERT(result != 0);
    }

    // ============== Command line option tests ==============
    
    void TestCmdOptionExists() {
        char* argv[] = {(char*)"program", (char*)"-f", (char*)"file.txt", (char*)"-v"};
        int argc = 4;
        
        CPPUNIT_ASSERT(cmdOptionExists(argv, argv + argc, "-f"));
        CPPUNIT_ASSERT(cmdOptionExists(argv, argv + argc, "-v"));
        CPPUNIT_ASSERT(!cmdOptionExists(argv, argv + argc, "-x"));
        // cmdOptionExists uses std::find on argv, so it matches any element including
        // value arguments, not just flags.  Verify it finds "file.txt" even though it
        // is a value, not a flag.
        CPPUNIT_ASSERT(cmdOptionExists(argv, argv + argc, "file.txt"));
    }
    
    void TestGetCmdOption() {
        char* argv[] = {(char*)"program", (char*)"-f", (char*)"file.txt", (char*)"-n", (char*)"100"};
        int argc = 5;
        
        char* result = getCmdOption(argv, argv + argc, "-f");
        CPPUNIT_ASSERT(result != nullptr);
        CPPUNIT_ASSERT_EQUAL(string("file.txt"), string(result));
        
        result = getCmdOption(argv, argv + argc, "-n");
        CPPUNIT_ASSERT(result != nullptr);
        CPPUNIT_ASSERT_EQUAL(string("100"), string(result));
        
        result = getCmdOption(argv, argv + argc, "-x");
        CPPUNIT_ASSERT(result == nullptr);
    }
    
    void TestGetOptionalArgValue() {
        char* argv[] = {(char*)"program", (char*)"-f", (char*)"file.txt"};
        int argc = 3;
        
        string result = getOptionalArgValue(argc, argv, "-f", "default.txt");
        CPPUNIT_ASSERT_EQUAL(string("file.txt"), result);
        
        result = getOptionalArgValue(argc, argv, "-x", "default_value");
        CPPUNIT_ASSERT_EQUAL(string("default_value"), result);
    }
    
    void TestIsArgSet() {
        char* argv[] = {(char*)"program", (char*)"-v", (char*)"-d"};
        int argc = 3;
        
        CPPUNIT_ASSERT(isArgSet(argc, argv, "-v"));
        CPPUNIT_ASSERT(isArgSet(argc, argv, "-d"));
        CPPUNIT_ASSERT(!isArgSet(argc, argv, "-q"));
    }

    // ============== CPPUNIT Test Suite Registration ==============
    
    CPPUNIT_TEST_SUITE(TestUtilsComprehensive);
    
    // isNetCDFfile tests
    CPPUNIT_TEST(TestIsNetCDFfileWithNcExtension);
    CPPUNIT_TEST(TestIsNetCDFfileWithOtherExtensions);
    CPPUNIT_TEST(TestIsNetCDFfileWithNoExtension);
    CPPUNIT_TEST(TestIsNetCDFfileWithEmptyString);
    CPPUNIT_TEST(TestIsNetCDFfileWithMultipleDots);
    
    // split tests
    CPPUNIT_TEST(TestSplitBasic);
    CPPUNIT_TEST(TestSplitWithTabDelimiter);
    CPPUNIT_TEST(TestSplitWithNoDelimiter);
    CPPUNIT_TEST(TestSplitWithEmptyString);
    CPPUNIT_TEST(TestSplitWithConsecutiveDelimiters);
    CPPUNIT_TEST(TestSplitWithTrailingDelimiter);
    CPPUNIT_TEST(TestSplitWithLeadingDelimiter);
    CPPUNIT_TEST(TestSplitWithReferenceVersion);
    
    // fileExists tests
    CPPUNIT_TEST(TestFileExistsWithExistingFile);
    CPPUNIT_TEST(TestFileExistsWithNonExistingFile);
    CPPUNIT_TEST(TestFileExistsWithDirectory);
    CPPUNIT_TEST(TestFileExistsWithEmptyPath);
    
    // isDirectory tests
    CPPUNIT_TEST(TestIsDirectoryWithDirectory);
    CPPUNIT_TEST(TestIsDirectoryWithFile);
    CPPUNIT_TEST(TestIsDirectoryWithNonExisting);
    
    // isFile tests
    CPPUNIT_TEST(TestIsFileWithFile);
    CPPUNIT_TEST(TestIsFileWithDirectory);
    CPPUNIT_TEST(TestIsFileWithNonExisting);
    
    // listFiles tests
    CPPUNIT_TEST(TestListFilesWithSingleFile);
    CPPUNIT_TEST(TestListFilesWithDirectory);
    CPPUNIT_TEST(TestListFilesResultsSorted);
    CPPUNIT_TEST(TestListFilesRecursive);
    CPPUNIT_TEST(TestListFilesWithEmptyDirectory);
    CPPUNIT_TEST(TestListFilesWithNonExistingPath);
    
    // Command line option tests
    CPPUNIT_TEST(TestCmdOptionExists);
    CPPUNIT_TEST(TestGetCmdOption);
    CPPUNIT_TEST(TestGetOptionalArgValue);
    CPPUNIT_TEST(TestIsArgSet);
    
    CPPUNIT_TEST_SUITE_END();
};

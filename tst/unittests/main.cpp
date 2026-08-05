#include <cstdlib>

#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/ui/text/TestRunner.h>

// Test files
#include "TestNetCDFhelper.cpp"
#include "TestUtils.cpp"
#include "TestUtilsComprehensive.cpp"
#include "TestNetCDFhelperExtended.cpp"
#include "TestDataTypes.cpp"
#include "TestCDLParser.cpp"
#include "TestImportSamples.cpp"

//
// In order to write a new test case, create a Test<File>.cpp and write the
// test methods in that file. Include the cpp file in this file and add:
//
//    runner.addTest(Test<Class>::suite());
//
// Unit test file names have to start with 'Test'
//
int main()
{
    CppUnit::TextUi::TestRunner runner;
    runner.addTest(TestNetCDFhelper::suite());
    runner.addTest(TestUtils::suite());
    runner.addTest(TestUtilsComprehensive::suite());
    runner.addTest(TestNetCDFhelperExtended::suite());
    // Note: TestNNDataSetDimensionsExtended tests are in tst/amazon/dsstne/engine/
    // and require GPU dependencies. They run as part of the GPU test suite.
    runner.addTest(TestElapsedSeconds::suite());
    runner.addTest(TestRandomUtils::suite());
    runner.addTest(TestCWMetric::suite());
    runner.addTest(TestConstants::suite());
    runner.addTest(TestCDLParser::suite());
    runner.addTest(TestImportSamples::suite());
    return runner.run() ? EXIT_SUCCESS : EXIT_FAILURE;
}

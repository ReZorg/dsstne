// CppUnit
#include "cppunit/extensions/HelperMacros.h"
#include "cppunit/ui/text/TestRunner.h"
// STL
#include <string>

// Original tests
#include "TestSort.cpp"
#include "TestActivationFunctions.cpp"
#include "TestCostFunctions.cpp"

// Phase 1: Core Inference & Training Tests
#include "TestNNNetworkInference.cpp"
#include "TestNNNetworkTraining.cpp"
#include "TestNNNetworkCheckpoints.cpp"
#include "TestNNLayerForward.cpp"
#include "TestNNLayerBackward.cpp"
#include "TestNNWeightInit.cpp"
#include "TestNNWeightUpdate.cpp"

/**
 * In order to write a new test case, create a Test<File>.cpp and write the test
 * methods in that file. Include the cpp file in this file and also 
 *
 * add runner.addTest(Test<Class>::suite());
 * Unit test file name has to start with Test
 *
 */

int main() {
    getGpu().Startup(0, NULL);
    getGpu().SetRandomSeed(12345);
    getGpu().CopyConstants();
    CppUnit::TextUi::TestRunner runner;
    
    // Original tests
    runner.addTest(TestSort::suite());
    runner.addTest(TestActivationFunctions::suite());
    runner.addTest(TestCostFunctions::suite());
    
    // Phase 1: NNNetwork Core Tests
    runner.addTest(TestNNNetworkInference::suite());
    runner.addTest(TestNNNetworkTraining::suite());
    runner.addTest(TestNNNetworkCheckpoints::suite());
    
    // Phase 1: NNLayer Core Tests
    runner.addTest(TestNNLayerForward::suite());
    runner.addTest(TestNNLayerBackward::suite());
    
    // Phase 1: NNWeight Core Tests
    runner.addTest(TestNNWeightInit::suite());
    runner.addTest(TestNNWeightUpdate::suite());
    
    const bool result = runner.run();
    getGpu().Shutdown();
    return result ? EXIT_SUCCESS : EXIT_FAILURE;
}

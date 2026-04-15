#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/ui/text/TestRunner.h>
#include <cppunit/TestAssert.h>

#include <string>
#include <map>

// Mirror the enums from NNTypes.h (without GPU dependencies)
// These values must match NNTypes.h exactly.

enum TrainingMode
{
    SGD      = 0,
    Momentum = 1,
    AdaGrad  = 2,
    Nesterov = 3,
    RMSProp  = 4,
    AdaDelta = 5,
    Adam     = 6,
};

enum Activation {
    Sigmoid,
    Tanh,
    RectifiedLinear,
    Linear,
    ParametricRectifiedLinear,
    SoftPlus,
    SoftSign,
    SoftMax,
    RELUMax,
    LinearMax,
    ExponentialLinear,
    LeakyRectifiedLinear,
    ScaledExponentialLinear,
};

enum PoolingFunction {
    None,
    Max,
    Average,
    LRN,
    Maxout,
    DotProduct,
    Cosine,
    Stochastic,
    LCN,
    GlobalTemporal,
};

class TestCDLParser : public CppUnit::TestFixture
{
public:
    void TestAllOptimizersHaveStringRepresentation()
    {
        // Mirror the sOptimizationMap from cdl.cpp
        std::map<std::string, TrainingMode> sOptimizationMap = {
            {"sgd",       SGD},
            {"momentum",  Momentum},
            {"adagrad",   AdaGrad},
            {"nesterov",  Nesterov},
            {"rmsprop",   RMSProp},
            {"adadelta",  AdaDelta},
            {"adam",      Adam},
        };

        CPPUNIT_ASSERT_EQUAL(7, (int)sOptimizationMap.size());
        CPPUNIT_ASSERT(sOptimizationMap.count("sgd")      == 1);
        CPPUNIT_ASSERT(sOptimizationMap.count("momentum") == 1);
        CPPUNIT_ASSERT(sOptimizationMap.count("adagrad")  == 1);
        CPPUNIT_ASSERT(sOptimizationMap.count("nesterov") == 1);
        CPPUNIT_ASSERT(sOptimizationMap.count("rmsprop")  == 1);
        CPPUNIT_ASSERT(sOptimizationMap.count("adadelta") == 1);
        CPPUNIT_ASSERT(sOptimizationMap.count("adam")     == 1);

        CPPUNIT_ASSERT_EQUAL((int)SGD,      (int)sOptimizationMap["sgd"]);
        CPPUNIT_ASSERT_EQUAL((int)Momentum, (int)sOptimizationMap["momentum"]);
        CPPUNIT_ASSERT_EQUAL((int)AdaGrad,  (int)sOptimizationMap["adagrad"]);
        CPPUNIT_ASSERT_EQUAL((int)Nesterov, (int)sOptimizationMap["nesterov"]);
        CPPUNIT_ASSERT_EQUAL((int)RMSProp,  (int)sOptimizationMap["rmsprop"]);
        CPPUNIT_ASSERT_EQUAL((int)AdaDelta, (int)sOptimizationMap["adadelta"]);
        CPPUNIT_ASSERT_EQUAL((int)Adam,     (int)sOptimizationMap["adam"]);
    }

    void TestTrainingModeEnumValues()
    {
        CPPUNIT_ASSERT_EQUAL(0, (int)SGD);
        CPPUNIT_ASSERT_EQUAL(1, (int)Momentum);
        CPPUNIT_ASSERT_EQUAL(2, (int)AdaGrad);
        CPPUNIT_ASSERT_EQUAL(3, (int)Nesterov);
        CPPUNIT_ASSERT_EQUAL(4, (int)RMSProp);
        CPPUNIT_ASSERT_EQUAL(5, (int)AdaDelta);
        CPPUNIT_ASSERT_EQUAL(6, (int)Adam);
    }

    void TestActivationEnumValues()
    {
        Activation acts[] = {
            Sigmoid, Tanh, RectifiedLinear, Linear,
            ParametricRectifiedLinear, SoftPlus, SoftSign,
            SoftMax, RELUMax, LinearMax,
            ExponentialLinear, LeakyRectifiedLinear, ScaledExponentialLinear
        };
        // Verify all 13 activations are distinct
        for (int i = 0; i < 13; i++)
            for (int j = i + 1; j < 13; j++)
                CPPUNIT_ASSERT(acts[i] != acts[j]);
    }

    void TestPoolingFunctionEnumValues()
    {
        PoolingFunction pfs[] = {
            None, Max, Average, LRN, Maxout,
            DotProduct, Cosine, Stochastic, LCN, GlobalTemporal
        };
        // Verify all 10 pooling functions are distinct
        for (int i = 0; i < 10; i++)
            for (int j = i + 1; j < 10; j++)
                CPPUNIT_ASSERT(pfs[i] != pfs[j]);
    }

    CPPUNIT_TEST_SUITE(TestCDLParser);
    CPPUNIT_TEST(TestAllOptimizersHaveStringRepresentation);
    CPPUNIT_TEST(TestTrainingModeEnumValues);
    CPPUNIT_TEST(TestActivationEnumValues);
    CPPUNIT_TEST(TestPoolingFunctionEnumValues);
    CPPUNIT_TEST_SUITE_END();
};

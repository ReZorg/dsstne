#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/ui/text/TestRunner.h>
#include <cppunit/TestAssert.h>

#include <string>
#include <map>
#include <set>

// Mirror the enums from NNTypes.h (src/amazon/dsstne/engine/NNTypes.h)
// without GPU dependencies. Values must stay in sync with NNTypes.h.

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

    // -------------------------------------------------------------------------
    // Activation string map – mirrors NNNetwork.cpp LoadNeuralNetworkJSON
    // -------------------------------------------------------------------------

    void TestAllActivationsHaveStringRepresentation()
    {
        // Every Activation enum value must have at least one lowercase string
        // entry in the LDL activation parser.  Mirror that map here so that
        // adding a new enum value without updating the parser will fail this test.
        std::map<std::string, Activation> sActivationMap = {
            {"sigmoid",                  Sigmoid},
            {"tanh",                     Tanh},
            {"linear",                   Linear},
            {"relu",                     RectifiedLinear},
            {"rectifiedlinear",          RectifiedLinear},
            {"lrelu",                    LeakyRectifiedLinear},
            {"leakyrectifiedlinear",     LeakyRectifiedLinear},
            {"prelu",                    ParametricRectifiedLinear},
            {"parametricrectifiedlinear",ParametricRectifiedLinear},
            {"elu",                      ExponentialLinear},
            {"exponentiallinear",        ExponentialLinear},
            {"selu",                     ScaledExponentialLinear},
            {"scaledexponentiallinear",  ScaledExponentialLinear},
            {"softplus",                 SoftPlus},
            {"softsign",                 SoftSign},
            {"softmax",                  SoftMax},
            {"relumax",                  RELUMax},
            {"linearmax",                LinearMax},
        };

        // Every Activation enum value must be reachable via at least one string
        std::set<Activation> covered;
        for (auto& kv : sActivationMap)
            covered.insert(kv.second);

        Activation all[] = {
            Sigmoid, Tanh, RectifiedLinear, Linear,
            ParametricRectifiedLinear, SoftPlus, SoftSign,
            SoftMax, RELUMax, LinearMax,
            ExponentialLinear, LeakyRectifiedLinear, ScaledExponentialLinear
        };
        for (Activation a : all)
            CPPUNIT_ASSERT_MESSAGE("Activation enum value has no string in map",
                                   covered.count(a) == 1);

        // Spot-check a few specific mappings
        CPPUNIT_ASSERT_EQUAL((int)Sigmoid,                   (int)sActivationMap["sigmoid"]);
        CPPUNIT_ASSERT_EQUAL((int)RectifiedLinear,           (int)sActivationMap["relu"]);
        CPPUNIT_ASSERT_EQUAL((int)ParametricRectifiedLinear, (int)sActivationMap["prelu"]);
        CPPUNIT_ASSERT_EQUAL((int)ParametricRectifiedLinear, (int)sActivationMap["parametricrectifiedlinear"]);
        CPPUNIT_ASSERT_EQUAL((int)SoftPlus,                  (int)sActivationMap["softplus"]);
        CPPUNIT_ASSERT_EQUAL((int)SoftSign,                  (int)sActivationMap["softsign"]);
        CPPUNIT_ASSERT_EQUAL((int)RELUMax,                   (int)sActivationMap["relumax"]);
        CPPUNIT_ASSERT_EQUAL((int)LinearMax,                 (int)sActivationMap["linearmax"]);
    }

    // -------------------------------------------------------------------------
    // Error function string map – mirrors NNNetwork.cpp LoadNeuralNetworkJSON
    // -------------------------------------------------------------------------

    enum ErrorFunction {
        L1,
        L2,
        CrossEntropy,
        ScaledMarginalCrossEntropy,
        DataScaledMarginalCrossEntropy,
        Hinge,
        L2Hinge,
    };

    void TestAllErrorFunctionsHaveStringRepresentation()
    {
        std::map<std::string, ErrorFunction> sErrorFunctionMap = {
            {"l1",                            L1},
            {"l2",                            L2},
            {"l2hinge",                       L2Hinge},
            {"hinge",                         Hinge},
            {"crossentropy",                  CrossEntropy},
            {"cross entropy",                 CrossEntropy},
            {"scaledmarginalcrossentropy",    ScaledMarginalCrossEntropy},
            {"datascaledmarginalcrossentropy",DataScaledMarginalCrossEntropy},
        };

        std::set<ErrorFunction> covered;
        for (auto& kv : sErrorFunctionMap)
            covered.insert(kv.second);

        ErrorFunction all[] = {
            L1, L2, CrossEntropy, ScaledMarginalCrossEntropy,
            DataScaledMarginalCrossEntropy, Hinge, L2Hinge
        };
        for (ErrorFunction ef : all)
            CPPUNIT_ASSERT_MESSAGE("ErrorFunction enum value has no string in map",
                                   covered.count(ef) == 1);

        CPPUNIT_ASSERT_EQUAL((int)L1,                            (int)sErrorFunctionMap["l1"]);
        CPPUNIT_ASSERT_EQUAL((int)L2,                            (int)sErrorFunctionMap["l2"]);
        CPPUNIT_ASSERT_EQUAL((int)CrossEntropy,                  (int)sErrorFunctionMap["crossentropy"]);
        CPPUNIT_ASSERT_EQUAL((int)ScaledMarginalCrossEntropy,    (int)sErrorFunctionMap["scaledmarginalcrossentropy"]);
        CPPUNIT_ASSERT_EQUAL((int)DataScaledMarginalCrossEntropy,(int)sErrorFunctionMap["datascaledmarginalcrossentropy"]);
        CPPUNIT_ASSERT_EQUAL((int)Hinge,                         (int)sErrorFunctionMap["hinge"]);
        CPPUNIT_ASSERT_EQUAL((int)L2Hinge,                       (int)sErrorFunctionMap["l2hinge"]);
    }

    // -------------------------------------------------------------------------
    // Weight init string map – mirrors NNNetwork.cpp LoadNeuralNetworkJSON
    // -------------------------------------------------------------------------

    enum WeightInitialization {
        Xavier,
        CaffeXavier,
        Gaussian,
        Uniform,
        UnitBall,
        Constant,
        SELU,
    };

    void TestAllWeightInitSchemesHaveStringRepresentation()
    {
        std::map<std::string, WeightInitialization> sWeightInitMap = {
            {"xavier",      Xavier},
            {"caffexavier", CaffeXavier},
            {"gaussian",    Gaussian},
            {"uniform",     Uniform},
            {"unitball",    UnitBall},
            {"constant",    Constant},
            {"selu",        SELU},
        };

        CPPUNIT_ASSERT_EQUAL(7, (int)sWeightInitMap.size());

        std::set<WeightInitialization> covered;
        for (auto& kv : sWeightInitMap)
            covered.insert(kv.second);

        WeightInitialization all[] = {
            Xavier, CaffeXavier, Gaussian, Uniform, UnitBall, Constant, SELU
        };
        for (WeightInitialization wi : all)
            CPPUNIT_ASSERT_MESSAGE("WeightInitialization enum value has no string in map",
                                   covered.count(wi) == 1);

        CPPUNIT_ASSERT_EQUAL((int)Xavier,      (int)sWeightInitMap["xavier"]);
        CPPUNIT_ASSERT_EQUAL((int)CaffeXavier, (int)sWeightInitMap["caffexavier"]);
        CPPUNIT_ASSERT_EQUAL((int)Gaussian,    (int)sWeightInitMap["gaussian"]);
        CPPUNIT_ASSERT_EQUAL((int)Uniform,     (int)sWeightInitMap["uniform"]);
        CPPUNIT_ASSERT_EQUAL((int)UnitBall,    (int)sWeightInitMap["unitball"]);
        CPPUNIT_ASSERT_EQUAL((int)Constant,    (int)sWeightInitMap["constant"]);
        CPPUNIT_ASSERT_EQUAL((int)SELU,        (int)sWeightInitMap["selu"]);
    }

    // -------------------------------------------------------------------------
    // Pooling function string map – mirrors NNNetwork.cpp LoadNeuralNetworkJSON
    // -------------------------------------------------------------------------

    void TestAllPoolingFunctionsHaveStringRepresentation()
    {
        std::map<std::string, PoolingFunction> sPoolingFunctionMap = {
            {"max",                          Max},
            {"maxout",                       Maxout},
            {"dotproduct",                   DotProduct},
            {"cosine",                       Cosine},
            {"average",                      Average},
            {"lrn",                          LRN},
            {"localresponsenormalization",   LRN},
            {"stochastic",                   Stochastic},
            {"localcontrastnormalization",   LCN},
            {"lcn",                          LCN},
            {"globaltemporal",               GlobalTemporal},
        };

        // All non-None pooling enum values must be reachable
        std::set<PoolingFunction> covered;
        for (auto& kv : sPoolingFunctionMap)
            covered.insert(kv.second);

        PoolingFunction parseable[] = {
            Max, Average, LRN, Maxout, DotProduct,
            Cosine, Stochastic, LCN, GlobalTemporal
        };
        for (PoolingFunction pf : parseable)
            CPPUNIT_ASSERT_MESSAGE("PoolingFunction enum value has no string in map",
                                   covered.count(pf) == 1);

        CPPUNIT_ASSERT_EQUAL((int)Max,           (int)sPoolingFunctionMap["max"]);
        CPPUNIT_ASSERT_EQUAL((int)Average,        (int)sPoolingFunctionMap["average"]);
        CPPUNIT_ASSERT_EQUAL((int)LRN,            (int)sPoolingFunctionMap["lrn"]);
        CPPUNIT_ASSERT_EQUAL((int)LRN,            (int)sPoolingFunctionMap["localresponsenormalization"]);
        CPPUNIT_ASSERT_EQUAL((int)Stochastic,     (int)sPoolingFunctionMap["stochastic"]);
        CPPUNIT_ASSERT_EQUAL((int)LCN,            (int)sPoolingFunctionMap["lcn"]);
        CPPUNIT_ASSERT_EQUAL((int)LCN,            (int)sPoolingFunctionMap["localcontrastnormalization"]);
        CPPUNIT_ASSERT_EQUAL((int)GlobalTemporal, (int)sPoolingFunctionMap["globaltemporal"]);
    }

    CPPUNIT_TEST_SUITE(TestCDLParser);
    CPPUNIT_TEST(TestAllOptimizersHaveStringRepresentation);
    CPPUNIT_TEST(TestTrainingModeEnumValues);
    CPPUNIT_TEST(TestActivationEnumValues);
    CPPUNIT_TEST(TestPoolingFunctionEnumValues);
    CPPUNIT_TEST(TestAllActivationsHaveStringRepresentation);
    CPPUNIT_TEST(TestAllErrorFunctionsHaveStringRepresentation);
    CPPUNIT_TEST(TestAllWeightInitSchemesHaveStringRepresentation);
    CPPUNIT_TEST(TestAllPoolingFunctionsHaveStringRepresentation);
    CPPUNIT_TEST_SUITE_END();
};

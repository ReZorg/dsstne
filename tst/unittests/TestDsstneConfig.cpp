#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/ui/text/TestRunner.h>
#include <cppunit/TestAssert.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
#include <unistd.h>

#include "DsstneConfig.h"

using dsstne::DsstneConfig;

namespace {

// Write text to a temp file and return its path.
std::string writeTempFile(const std::string& contents)
{
    std::string path = std::string("/tmp/dsstne_config_test_XXXXXX");
    std::vector<char> buf(path.begin(), path.end());
    buf.push_back('\0');
    int fd = mkstemp(buf.data());
    if (fd == -1) {
        throw std::runtime_error("mkstemp failed");
    }
    close(fd);
    path = buf.data();

    std::ofstream out(path);
    out << contents;
    out.close();
    return path;
}

} // namespace

class TestDsstneConfig : public CppUnit::TestFixture
{
public:
    void TestDefaultsAreValid()
    {
        DsstneConfig config;
        CPPUNIT_ASSERT(config.isValid());
        CPPUNIT_ASSERT(config.validate().empty());

        // Spot-check a few documented defaults
        CPPUNIT_ASSERT_EQUAL(-1, config.gpu.deviceId);
        CPPUNIT_ASSERT_EQUAL(std::string("sgd"), config.training.optimizer);
        CPPUNIT_ASSERT_EQUAL((uint32_t)32, config.network.batchSize);
        CPPUNIT_ASSERT_EQUAL((uint32_t)10, config.inference.topK);
        CPPUNIT_ASSERT_EQUAL((uint32_t)10, config.knn.k);
        CPPUNIT_ASSERT_EQUAL(std::string("INFO"), config.logging.level);
    }

    void TestValidateDetectsBadValues()
    {
        DsstneConfig config;

        config.gpu.memoryFraction = 1.5f;
        CPPUNIT_ASSERT(!config.isValid());

        config = DsstneConfig();
        config.training.learningRate = 0.0f;
        CPPUNIT_ASSERT(!config.isValid());

        config = DsstneConfig();
        config.network.batchSize = 0;
        CPPUNIT_ASSERT(!config.isValid());

        config = DsstneConfig();
        config.inference.topK = 0;
        CPPUNIT_ASSERT(!config.isValid());

        config = DsstneConfig();
        config.knn.k = 0;
        CPPUNIT_ASSERT(!config.isValid());
    }

    void TestEnvironmentOverrides()
    {
        setenv("DSSTNE_GPU_DEVICE_ID", "3", 1);
        setenv("DSSTNE_LEARNING_RATE", "0.25", 1);
        setenv("DSSTNE_BATCH_SIZE", "128", 1);
        setenv("DSSTNE_LOG_LEVEL", "DEBUG", 1);
        setenv("DSSTNE_VGPU_MODE", "true", 1);

        DsstneConfig config;
        config.applyEnvironmentOverrides();

        CPPUNIT_ASSERT_EQUAL(3, config.gpu.deviceId);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.25, config.training.learningRate, 1e-6);
        CPPUNIT_ASSERT_EQUAL((uint32_t)128, config.network.batchSize);
        CPPUNIT_ASSERT_EQUAL(std::string("DEBUG"), config.logging.level);
        CPPUNIT_ASSERT(config.gpu.vgpuMode);

        unsetenv("DSSTNE_GPU_DEVICE_ID");
        unsetenv("DSSTNE_LEARNING_RATE");
        unsetenv("DSSTNE_BATCH_SIZE");
        unsetenv("DSSTNE_LOG_LEVEL");
        unsetenv("DSSTNE_VGPU_MODE");
    }

    void TestJsonRoundTrip()
    {
        DsstneConfig config;
        config.network.name = "roundtrip-net";
        config.training.optimizer = "adam";
        config.training.learningRate = 0.5f;
        config.network.batchSize = 64;
        config.inference.topK = 7;
        config.knn.k = 42;
        config.knn.useGpu = false;
        config.logging.level = "WARN";
        config.gpu.deviceId = 2;

        std::string path = writeTempFile("");
        config.saveJson(path);

        DsstneConfig loaded = DsstneConfig::loadJson(path);
        std::remove(path.c_str());

        CPPUNIT_ASSERT_EQUAL(std::string("roundtrip-net"), loaded.network.name);
        CPPUNIT_ASSERT_EQUAL(std::string("adam"), loaded.training.optimizer);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.5, loaded.training.learningRate, 1e-6);
        CPPUNIT_ASSERT_EQUAL((uint32_t)64, loaded.network.batchSize);
        CPPUNIT_ASSERT_EQUAL((uint32_t)7, loaded.inference.topK);
        CPPUNIT_ASSERT_EQUAL((uint32_t)42, loaded.knn.k);
        CPPUNIT_ASSERT(!loaded.knn.useGpu);
        CPPUNIT_ASSERT_EQUAL(std::string("WARN"), loaded.logging.level);
        CPPUNIT_ASSERT_EQUAL(2, loaded.gpu.deviceId);
    }

    void TestJsonPartialLoadKeepsDefaults()
    {
        std::string path = writeTempFile("{\"network\": {\"batchSize\": 256}}");
        DsstneConfig loaded = DsstneConfig::loadJson(path);
        std::remove(path.c_str());

        // Overridden value
        CPPUNIT_ASSERT_EQUAL((uint32_t)256, loaded.network.batchSize);
        // Untouched defaults retained
        CPPUNIT_ASSERT_EQUAL(std::string("sgd"), loaded.training.optimizer);
        CPPUNIT_ASSERT_EQUAL((uint32_t)10, loaded.knn.k);
    }

    void TestLoadJsonThrowsOnMissingFile()
    {
        bool threw = false;
        try {
            DsstneConfig::loadJson("/tmp/dsstne_definitely_missing_config.json");
        } catch (const std::runtime_error&) {
            threw = true;
        }
        CPPUNIT_ASSERT(threw);
    }

    void TestLoadJsonThrowsOnMalformed()
    {
        std::string path = writeTempFile("{ this is not valid json ");
        bool threw = false;
        try {
            DsstneConfig::loadJson(path);
        } catch (const std::runtime_error&) {
            threw = true;
        }
        std::remove(path.c_str());
        CPPUNIT_ASSERT(threw);
    }

    void TestYamlRoundTrip()
    {
        DsstneConfig config;
        config.network.name = "yaml-net";
        config.training.optimizer = "momentum";
        config.training.momentum = 0.75f;
        config.network.batchSize = 96;
        config.inference.topK = 3;
        config.knn.k = 15;
        config.logging.level = "ERROR";
        config.logging.console = false;
        config.gpu.vgpuMode = true;

        std::string path = writeTempFile("");
        config.saveYaml(path);

        DsstneConfig loaded = DsstneConfig::loadYaml(path);
        std::remove(path.c_str());

        CPPUNIT_ASSERT_EQUAL(std::string("yaml-net"), loaded.network.name);
        CPPUNIT_ASSERT_EQUAL(std::string("momentum"), loaded.training.optimizer);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.75, loaded.training.momentum, 1e-6);
        CPPUNIT_ASSERT_EQUAL((uint32_t)96, loaded.network.batchSize);
        CPPUNIT_ASSERT_EQUAL((uint32_t)3, loaded.inference.topK);
        CPPUNIT_ASSERT_EQUAL((uint32_t)15, loaded.knn.k);
        CPPUNIT_ASSERT_EQUAL(std::string("ERROR"), loaded.logging.level);
        CPPUNIT_ASSERT(!loaded.logging.console);
        CPPUNIT_ASSERT(loaded.gpu.vgpuMode);
    }

    void TestLoadYamlThrowsOnMissingFile()
    {
        bool threw = false;
        try {
            DsstneConfig::loadYaml("/tmp/dsstne_definitely_missing_config.yaml");
        } catch (const std::runtime_error&) {
            threw = true;
        }
        CPPUNIT_ASSERT(threw);
    }

    void TestMerge()
    {
        DsstneConfig base;
        base.network.batchSize = 16;
        base.training.optimizer = "sgd";

        DsstneConfig override_cfg;
        override_cfg.network.batchSize = 512;
        override_cfg.training.optimizer = "adam";
        override_cfg.gpu.deviceId = 5;

        base.merge(override_cfg);

        CPPUNIT_ASSERT_EQUAL((uint32_t)512, base.network.batchSize);
        CPPUNIT_ASSERT_EQUAL(std::string("adam"), base.training.optimizer);
        CPPUNIT_ASSERT_EQUAL(5, base.gpu.deviceId);
    }

    void TestParseCommandLine()
    {
        const char* argv_arr[] = {
            "prog",
            "--network.batchSize=128",
            "--training.optimizer=adam",
            "--gpu.deviceId=1",
            "--knn.k=20",
            "--logging.level=TRACE",
        };
        int argc = sizeof(argv_arr) / sizeof(argv_arr[0]);

        DsstneConfig config;
        config.parseCommandLine(argc, const_cast<char**>(argv_arr));

        CPPUNIT_ASSERT_EQUAL((uint32_t)128, config.network.batchSize);
        CPPUNIT_ASSERT_EQUAL(std::string("adam"), config.training.optimizer);
        CPPUNIT_ASSERT_EQUAL(1, config.gpu.deviceId);
        CPPUNIT_ASSERT_EQUAL((uint32_t)20, config.knn.k);
        CPPUNIT_ASSERT_EQUAL(std::string("TRACE"), config.logging.level);
    }

    void TestConfigVersion()
    {
        dsstne::ConfigVersion v(1, 2, 3);
        CPPUNIT_ASSERT_EQUAL(std::string("1.2.3"), v.toString());
        CPPUNIT_ASSERT(v.isCompatibleWith(dsstne::ConfigVersion(1, 9, 9)));
        CPPUNIT_ASSERT(!v.isCompatibleWith(dsstne::ConfigVersion(2, 0, 0)));
    }

    CPPUNIT_TEST_SUITE(TestDsstneConfig);
    CPPUNIT_TEST(TestDefaultsAreValid);
    CPPUNIT_TEST(TestValidateDetectsBadValues);
    CPPUNIT_TEST(TestEnvironmentOverrides);
    CPPUNIT_TEST(TestJsonRoundTrip);
    CPPUNIT_TEST(TestJsonPartialLoadKeepsDefaults);
    CPPUNIT_TEST(TestLoadJsonThrowsOnMissingFile);
    CPPUNIT_TEST(TestLoadJsonThrowsOnMalformed);
    CPPUNIT_TEST(TestYamlRoundTrip);
    CPPUNIT_TEST(TestLoadYamlThrowsOnMissingFile);
    CPPUNIT_TEST(TestMerge);
    CPPUNIT_TEST(TestParseCommandLine);
    CPPUNIT_TEST(TestConfigVersion);
    CPPUNIT_TEST_SUITE_END();
};

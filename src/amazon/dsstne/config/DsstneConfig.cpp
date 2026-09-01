/*
 *  Copyright 2016-2026  Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License").
 *  You may not use this file except in compliance with the License.
 *  A copy of the License is located at
 *
 *  http://aws.amazon.com/apache2.0/
 *
 *  or in the "license" file accompanying this file.
 *  This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
 *  either express or implied.
 *
 *  See the License for the specific language governing permissions and limitations under the License.
 */

#include "DsstneConfig.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cstring>
#include <cmath>

#include <json/json.h>

namespace dsstne {

//=============================================================================
// JSON serialization helpers
//=============================================================================

namespace {

// Convert the config into a JSON value tree.
Json::Value toJsonValue(const DsstneConfig& c) {
    Json::Value root(Json::objectValue);

    root["version"]["major"] = c.version.major;
    root["version"]["minor"] = c.version.minor;
    root["version"]["patch"] = c.version.patch;

    root["gpu"]["deviceId"] = c.gpu.deviceId;
    root["gpu"]["memoryFraction"] = c.gpu.memoryFraction;
    root["gpu"]["allowGrowth"] = c.gpu.allowGrowth;
    root["gpu"]["vgpuMode"] = c.gpu.vgpuMode;
    root["gpu"]["maxConcurrentKernels"] = c.gpu.maxConcurrentKernels;

    root["training"]["optimizer"] = c.training.optimizer;
    root["training"]["learningRate"] = c.training.learningRate;
    root["training"]["momentum"] = c.training.momentum;
    root["training"]["weightDecay"] = c.training.weightDecay;
    root["training"]["l1Regularization"] = c.training.l1Regularization;
    root["training"]["gradientClip"] = c.training.gradientClip;
    root["training"]["shuffleIndices"] = c.training.shuffleIndices;
    root["training"]["checkpointInterval"] = c.training.checkpointInterval;
    root["training"]["checkpointPath"] = c.training.checkpointPath;

    root["network"]["name"] = c.network.name;
    root["network"]["kind"] = c.network.kind;
    root["network"]["errorFunction"] = c.network.errorFunction;
    root["network"]["batchSize"] = c.network.batchSize;
    root["network"]["dataPath"] = c.network.dataPath;
    root["network"]["modelPath"] = c.network.modelPath;

    root["inference"]["topK"] = c.inference.topK;
    root["inference"]["threshold"] = c.inference.threshold;
    root["inference"]["returnScores"] = c.inference.returnScores;
    root["inference"]["returnEmbeddings"] = c.inference.returnEmbeddings;
    root["inference"]["embeddingLayer"] = c.inference.embeddingLayer;

    root["knn"]["k"] = c.knn.k;
    root["knn"]["useGpu"] = c.knn.useGpu;
    root["knn"]["batchSize"] = c.knn.batchSize;
    root["knn"]["metric"] = c.knn.metric;

    root["logging"]["level"] = c.logging.level;
    root["logging"]["file"] = c.logging.file;
    root["logging"]["console"] = c.logging.console;
    root["logging"]["timestamps"] = c.logging.timestamps;

    return root;
}

// Populate the config from a JSON value tree. Missing keys keep their
// existing (default or previously set) values so partial config files work.
void fromJsonValue(DsstneConfig& c, const Json::Value& root) {
    if (root.isMember("version")) {
        const Json::Value& v = root["version"];
        if (v.isMember("major")) c.version.major = v["major"].asUInt();
        if (v.isMember("minor")) c.version.minor = v["minor"].asUInt();
        if (v.isMember("patch")) c.version.patch = v["patch"].asUInt();
    }
    if (root.isMember("gpu")) {
        const Json::Value& v = root["gpu"];
        if (v.isMember("deviceId")) c.gpu.deviceId = v["deviceId"].asInt();
        if (v.isMember("memoryFraction")) c.gpu.memoryFraction = v["memoryFraction"].asFloat();
        if (v.isMember("allowGrowth")) c.gpu.allowGrowth = v["allowGrowth"].asBool();
        if (v.isMember("vgpuMode")) c.gpu.vgpuMode = v["vgpuMode"].asBool();
        if (v.isMember("maxConcurrentKernels")) c.gpu.maxConcurrentKernels = v["maxConcurrentKernels"].asUInt();
    }
    if (root.isMember("training")) {
        const Json::Value& v = root["training"];
        if (v.isMember("optimizer")) c.training.optimizer = v["optimizer"].asString();
        if (v.isMember("learningRate")) c.training.learningRate = v["learningRate"].asFloat();
        if (v.isMember("momentum")) c.training.momentum = v["momentum"].asFloat();
        if (v.isMember("weightDecay")) c.training.weightDecay = v["weightDecay"].asFloat();
        if (v.isMember("l1Regularization")) c.training.l1Regularization = v["l1Regularization"].asFloat();
        if (v.isMember("gradientClip")) c.training.gradientClip = v["gradientClip"].asFloat();
        if (v.isMember("shuffleIndices")) c.training.shuffleIndices = v["shuffleIndices"].asBool();
        if (v.isMember("checkpointInterval")) c.training.checkpointInterval = v["checkpointInterval"].asUInt();
        if (v.isMember("checkpointPath")) c.training.checkpointPath = v["checkpointPath"].asString();
    }
    if (root.isMember("network")) {
        const Json::Value& v = root["network"];
        if (v.isMember("name")) c.network.name = v["name"].asString();
        if (v.isMember("kind")) c.network.kind = v["kind"].asString();
        if (v.isMember("errorFunction")) c.network.errorFunction = v["errorFunction"].asString();
        if (v.isMember("batchSize")) c.network.batchSize = v["batchSize"].asUInt();
        if (v.isMember("dataPath")) c.network.dataPath = v["dataPath"].asString();
        if (v.isMember("modelPath")) c.network.modelPath = v["modelPath"].asString();
    }
    if (root.isMember("inference")) {
        const Json::Value& v = root["inference"];
        if (v.isMember("topK")) c.inference.topK = v["topK"].asUInt();
        if (v.isMember("threshold")) c.inference.threshold = v["threshold"].asFloat();
        if (v.isMember("returnScores")) c.inference.returnScores = v["returnScores"].asBool();
        if (v.isMember("returnEmbeddings")) c.inference.returnEmbeddings = v["returnEmbeddings"].asBool();
        if (v.isMember("embeddingLayer")) c.inference.embeddingLayer = v["embeddingLayer"].asString();
    }
    if (root.isMember("knn")) {
        const Json::Value& v = root["knn"];
        if (v.isMember("k")) c.knn.k = v["k"].asUInt();
        if (v.isMember("useGpu")) c.knn.useGpu = v["useGpu"].asBool();
        if (v.isMember("batchSize")) c.knn.batchSize = v["batchSize"].asUInt();
        if (v.isMember("metric")) c.knn.metric = v["metric"].asString();
    }
    if (root.isMember("logging")) {
        const Json::Value& v = root["logging"];
        if (v.isMember("level")) c.logging.level = v["level"].asString();
        if (v.isMember("file")) c.logging.file = v["file"].asString();
        if (v.isMember("console")) c.logging.console = v["console"].asBool();
        if (v.isMember("timestamps")) c.logging.timestamps = v["timestamps"].asBool();
    }
}

//=============================================================================
// Minimal YAML helpers (flat "section: { key: value }" subset)
//=============================================================================

// Quote a scalar for YAML output if it could be misparsed.
std::string yamlScalar(const std::string& s) {
    if (s.empty()) return "\"\"";
    // Quote if it contains characters that would confuse a simple parser.
    if (s.find_first_of(":#\n\"'") != std::string::npos ||
        s.front() == ' ' || s.back() == ' ') {
        std::string out = "\"";
        for (char ch : s) {
            if (ch == '"' || ch == '\\') out += '\\';
            out += ch;
        }
        out += '"';
        return out;
    }
    return s;
}

std::string yamlBool(bool b) { return b ? "true" : "false"; }

// Strip surrounding whitespace.
std::string trim(const std::string& s) {
    const char* ws = " \t\r\n";
    size_t b = s.find_first_not_of(ws);
    if (b == std::string::npos) return "";
    size_t e = s.find_last_not_of(ws);
    return s.substr(b, e - b + 1);
}

// Remove surrounding double quotes if present, unescaping \" and \\ so that
// values emitted by yamlScalar round-trip correctly.
std::string unquote(const std::string& s) {
    if (s.size() >= 2 && s.front() == '"' && s.back() == '"') {
        const size_t last = s.size() - 1;  // index of the closing quote
        std::string out;
        out.reserve(s.size() - 2);
        for (size_t i = 1; i < last; ++i) {
            // An escape is valid only when the escaped char is still inside the
            // quoted body (strictly before the closing quote).
            if (s[i] == '\\' && i + 1 < last && (s[i + 1] == '"' || s[i + 1] == '\\')) {
                out += s[i + 1];
                ++i;
            } else {
                out += s[i];
            }
        }
        return out;
    }
    return s;
}

// Set a config field addressed by "section.key" from a YAML string scalar.
void setScalar(DsstneConfig& c, const std::string& section, const std::string& key,
               const std::string& rawVal) {
    const std::string val = unquote(trim(rawVal));
    try {
        if (section == "version") {
            if (key == "major") c.version.major = static_cast<uint32_t>(std::stoul(val));
            else if (key == "minor") c.version.minor = static_cast<uint32_t>(std::stoul(val));
            else if (key == "patch") c.version.patch = static_cast<uint32_t>(std::stoul(val));
        } else if (section == "gpu") {
            if (key == "deviceId") c.gpu.deviceId = std::stoi(val);
            else if (key == "memoryFraction") c.gpu.memoryFraction = std::stof(val);
            else if (key == "allowGrowth") c.gpu.allowGrowth = (val == "true" || val == "1");
            else if (key == "vgpuMode") c.gpu.vgpuMode = (val == "true" || val == "1");
            else if (key == "maxConcurrentKernels") c.gpu.maxConcurrentKernels = static_cast<uint32_t>(std::stoul(val));
        } else if (section == "training") {
            if (key == "optimizer") c.training.optimizer = val;
            else if (key == "learningRate") c.training.learningRate = std::stof(val);
            else if (key == "momentum") c.training.momentum = std::stof(val);
            else if (key == "weightDecay") c.training.weightDecay = std::stof(val);
            else if (key == "l1Regularization") c.training.l1Regularization = std::stof(val);
            else if (key == "gradientClip") c.training.gradientClip = std::stof(val);
            else if (key == "shuffleIndices") c.training.shuffleIndices = (val == "true" || val == "1");
            else if (key == "checkpointInterval") c.training.checkpointInterval = static_cast<uint32_t>(std::stoul(val));
            else if (key == "checkpointPath") c.training.checkpointPath = val;
        } else if (section == "network") {
            if (key == "name") c.network.name = val;
            else if (key == "kind") c.network.kind = val;
            else if (key == "errorFunction") c.network.errorFunction = val;
            else if (key == "batchSize") c.network.batchSize = static_cast<uint32_t>(std::stoul(val));
            else if (key == "dataPath") c.network.dataPath = val;
            else if (key == "modelPath") c.network.modelPath = val;
        } else if (section == "inference") {
            if (key == "topK") c.inference.topK = static_cast<uint32_t>(std::stoul(val));
            else if (key == "threshold") c.inference.threshold = std::stof(val);
            else if (key == "returnScores") c.inference.returnScores = (val == "true" || val == "1");
            else if (key == "returnEmbeddings") c.inference.returnEmbeddings = (val == "true" || val == "1");
            else if (key == "embeddingLayer") c.inference.embeddingLayer = val;
        } else if (section == "knn") {
            if (key == "k") c.knn.k = static_cast<uint32_t>(std::stoul(val));
            else if (key == "useGpu") c.knn.useGpu = (val == "true" || val == "1");
            else if (key == "batchSize") c.knn.batchSize = static_cast<uint32_t>(std::stoul(val));
            else if (key == "metric") c.knn.metric = val;
        } else if (section == "logging") {
            if (key == "level") c.logging.level = val;
            else if (key == "file") c.logging.file = val;
            else if (key == "console") c.logging.console = (val == "true" || val == "1");
            else if (key == "timestamps") c.logging.timestamps = (val == "true" || val == "1");
        }
    } catch (const std::exception&) {
        throw std::runtime_error("DsstneConfig: invalid YAML value for " + section + "." + key + ": '" + rawVal + "'");
    }
}

} // anonymous namespace

//=============================================================================
// JSON load / save
//=============================================================================

DsstneConfig DsstneConfig::loadJson(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("DsstneConfig: unable to open JSON config file: " + path);
    }
    std::stringstream ss;
    ss << in.rdbuf();
    const std::string text = ss.str();

    Json::Value root;
    Json::CharReaderBuilder builder;
    std::string errs;
    std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
    if (!reader->parse(text.data(), text.data() + text.size(), &root, &errs)) {
        throw std::runtime_error("DsstneConfig: failed to parse JSON config " + path + ": " + errs);
    }

    DsstneConfig config;
    fromJsonValue(config, root);
    return config;
}

void DsstneConfig::saveJson(const std::string& path) const {
    Json::Value root = toJsonValue(*this);
    Json::StreamWriterBuilder builder;
    builder["indentation"] = "  ";
    std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());

    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("DsstneConfig: unable to write JSON config file: " + path);
    }
    writer->write(root, &out);
    out << "\n";
}

//=============================================================================
// YAML load / save
//=============================================================================

void DsstneConfig::saveYaml(const std::string& path) const {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("DsstneConfig: unable to write YAML config file: " + path);
    }

    out << "version:\n"
        << "  major: " << version.major << "\n"
        << "  minor: " << version.minor << "\n"
        << "  patch: " << version.patch << "\n";

    out << "gpu:\n"
        << "  deviceId: " << gpu.deviceId << "\n"
        << "  memoryFraction: " << gpu.memoryFraction << "\n"
        << "  allowGrowth: " << yamlBool(gpu.allowGrowth) << "\n"
        << "  vgpuMode: " << yamlBool(gpu.vgpuMode) << "\n"
        << "  maxConcurrentKernels: " << gpu.maxConcurrentKernels << "\n";

    out << "training:\n"
        << "  optimizer: " << yamlScalar(training.optimizer) << "\n"
        << "  learningRate: " << training.learningRate << "\n"
        << "  momentum: " << training.momentum << "\n"
        << "  weightDecay: " << training.weightDecay << "\n"
        << "  l1Regularization: " << training.l1Regularization << "\n"
        << "  gradientClip: " << training.gradientClip << "\n"
        << "  shuffleIndices: " << yamlBool(training.shuffleIndices) << "\n"
        << "  checkpointInterval: " << training.checkpointInterval << "\n"
        << "  checkpointPath: " << yamlScalar(training.checkpointPath) << "\n";

    out << "network:\n"
        << "  name: " << yamlScalar(network.name) << "\n"
        << "  kind: " << yamlScalar(network.kind) << "\n"
        << "  errorFunction: " << yamlScalar(network.errorFunction) << "\n"
        << "  batchSize: " << network.batchSize << "\n"
        << "  dataPath: " << yamlScalar(network.dataPath) << "\n"
        << "  modelPath: " << yamlScalar(network.modelPath) << "\n";

    out << "inference:\n"
        << "  topK: " << inference.topK << "\n"
        << "  threshold: " << inference.threshold << "\n"
        << "  returnScores: " << yamlBool(inference.returnScores) << "\n"
        << "  returnEmbeddings: " << yamlBool(inference.returnEmbeddings) << "\n"
        << "  embeddingLayer: " << yamlScalar(inference.embeddingLayer) << "\n";

    out << "knn:\n"
        << "  k: " << knn.k << "\n"
        << "  useGpu: " << yamlBool(knn.useGpu) << "\n"
        << "  batchSize: " << knn.batchSize << "\n"
        << "  metric: " << yamlScalar(knn.metric) << "\n";

    out << "logging:\n"
        << "  level: " << yamlScalar(logging.level) << "\n"
        << "  file: " << yamlScalar(logging.file) << "\n"
        << "  console: " << yamlBool(logging.console) << "\n"
        << "  timestamps: " << yamlBool(logging.timestamps) << "\n";
}

DsstneConfig DsstneConfig::loadYaml(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("DsstneConfig: unable to open YAML config file: " + path);
    }

    DsstneConfig config;
    std::string line;
    std::string section;
    while (std::getline(in, line)) {
        // Strip comments (everything after a '#' that is not inside a quoted
        // string), so values like checkpointPath: "a/#b" are not truncated.
        bool inQuotes = false;
        for (size_t i = 0; i < line.size(); ++i) {
            if (line[i] == '\\' && inQuotes) { ++i; continue; }  // skip escaped char
            if (line[i] == '"') inQuotes = !inQuotes;
            else if (line[i] == '#' && !inQuotes) { line.resize(i); break; }
        }
        if (trim(line).empty()) continue;

        const bool indented = (line[0] == ' ' || line[0] == '\t');
        const std::string content = trim(line);
        const size_t colon = content.find(':');
        if (colon == std::string::npos) continue;

        if (!indented) {
            // Top-level section header (e.g. "training:").
            section = trim(content.substr(0, colon));
        } else {
            const std::string key = trim(content.substr(0, colon));
            const std::string value = trim(content.substr(colon + 1));
            if (!section.empty() && !key.empty()) {
                setScalar(config, section, key, value);
            }
        }
    }
    return config;
}

//=============================================================================
// Command-line parsing
//=============================================================================

void DsstneConfig::parseCommandLine(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg.rfind("--", 0) != 0) continue;  // Only handle --key=value

        const size_t eq = arg.find('=');
        if (eq == std::string::npos) continue;

        const std::string key = arg.substr(2, eq - 2);
        const std::string value = arg.substr(eq + 1);
        const size_t dot = key.find('.');
        if (dot == std::string::npos) continue;

        const std::string section = key.substr(0, dot);
        const std::string field = key.substr(dot + 1);
        setScalar(*this, section, field, value);
    }
}

//=============================================================================
// Merge
//=============================================================================

void DsstneConfig::merge(const DsstneConfig& other) {
    // Field-level precedence: for each field, the value from `other` is applied
    // only when it differs from the default-constructed value. This lets a
    // partially-specified `other` override just the fields it sets without
    // clobbering customizations in `*this` that `other` left at defaults.
    //
    // Floats are compared with a relative epsilon so that values surviving a
    // serialization round-trip (which may differ by an ULP) are not mistaken
    // for intentional overrides.
    const DsstneConfig dflt;  // defaults for comparison
    const auto floatDiffers = [](float a, float b) {
        const float scale = std::max(std::fabs(a), std::fabs(b));
        return std::fabs(a - b) > 1e-6f * (scale > 1.0f ? scale : 1.0f);
    };

    // version
    if (other.version.major != dflt.version.major) version.major = other.version.major;
    if (other.version.minor != dflt.version.minor) version.minor = other.version.minor;
    if (other.version.patch != dflt.version.patch) version.patch = other.version.patch;

    // gpu
    if (other.gpu.deviceId != dflt.gpu.deviceId) gpu.deviceId = other.gpu.deviceId;
    if (floatDiffers(other.gpu.memoryFraction, dflt.gpu.memoryFraction)) gpu.memoryFraction = other.gpu.memoryFraction;
    if (other.gpu.allowGrowth != dflt.gpu.allowGrowth) gpu.allowGrowth = other.gpu.allowGrowth;
    if (other.gpu.vgpuMode != dflt.gpu.vgpuMode) gpu.vgpuMode = other.gpu.vgpuMode;
    if (other.gpu.maxConcurrentKernels != dflt.gpu.maxConcurrentKernels) gpu.maxConcurrentKernels = other.gpu.maxConcurrentKernels;

    // training
    if (other.training.optimizer != dflt.training.optimizer) training.optimizer = other.training.optimizer;
    if (floatDiffers(other.training.learningRate, dflt.training.learningRate)) training.learningRate = other.training.learningRate;
    if (floatDiffers(other.training.momentum, dflt.training.momentum)) training.momentum = other.training.momentum;
    if (floatDiffers(other.training.weightDecay, dflt.training.weightDecay)) training.weightDecay = other.training.weightDecay;
    if (floatDiffers(other.training.l1Regularization, dflt.training.l1Regularization)) training.l1Regularization = other.training.l1Regularization;
    if (floatDiffers(other.training.gradientClip, dflt.training.gradientClip)) training.gradientClip = other.training.gradientClip;
    if (other.training.shuffleIndices != dflt.training.shuffleIndices) training.shuffleIndices = other.training.shuffleIndices;
    if (other.training.checkpointInterval != dflt.training.checkpointInterval) training.checkpointInterval = other.training.checkpointInterval;
    if (other.training.checkpointPath != dflt.training.checkpointPath) training.checkpointPath = other.training.checkpointPath;

    // network
    if (other.network.name != dflt.network.name) network.name = other.network.name;
    if (other.network.kind != dflt.network.kind) network.kind = other.network.kind;
    if (other.network.errorFunction != dflt.network.errorFunction) network.errorFunction = other.network.errorFunction;
    if (other.network.batchSize != dflt.network.batchSize) network.batchSize = other.network.batchSize;
    if (other.network.dataPath != dflt.network.dataPath) network.dataPath = other.network.dataPath;
    if (other.network.modelPath != dflt.network.modelPath) network.modelPath = other.network.modelPath;

    // inference
    if (other.inference.topK != dflt.inference.topK) inference.topK = other.inference.topK;
    if (floatDiffers(other.inference.threshold, dflt.inference.threshold)) inference.threshold = other.inference.threshold;
    if (other.inference.returnScores != dflt.inference.returnScores) inference.returnScores = other.inference.returnScores;
    if (other.inference.returnEmbeddings != dflt.inference.returnEmbeddings) inference.returnEmbeddings = other.inference.returnEmbeddings;
    if (other.inference.embeddingLayer != dflt.inference.embeddingLayer) inference.embeddingLayer = other.inference.embeddingLayer;

    // knn
    if (other.knn.k != dflt.knn.k) knn.k = other.knn.k;
    if (other.knn.useGpu != dflt.knn.useGpu) knn.useGpu = other.knn.useGpu;
    if (other.knn.batchSize != dflt.knn.batchSize) knn.batchSize = other.knn.batchSize;
    if (other.knn.metric != dflt.knn.metric) knn.metric = other.knn.metric;

    // logging
    if (other.logging.level != dflt.logging.level) logging.level = other.logging.level;
    if (other.logging.file != dflt.logging.file) logging.file = other.logging.file;
    if (other.logging.console != dflt.logging.console) logging.console = other.logging.console;
    if (other.logging.timestamps != dflt.logging.timestamps) logging.timestamps = other.logging.timestamps;
}

} // namespace dsstne

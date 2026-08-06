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

#ifndef DSSTNE_LOGGER_H
#define DSSTNE_LOGGER_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <mutex>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <memory>
#include <map>

namespace dsstne {

/**
 * @brief Log severity levels
 */
enum class LogLevel {
    TRACE = 0,
    DEBUG = 1,
    INFO = 2,
    WARN = 3,
    ERROR = 4,
    FATAL = 5,
    OFF = 6
};

/**
 * @brief Convert log level to string
 */
inline std::string LogLevelToString(LogLevel level) {
    switch (level) {
        case LogLevel::TRACE: return "TRACE";
        case LogLevel::DEBUG: return "DEBUG";
        case LogLevel::INFO:  return "INFO ";
        case LogLevel::WARN:  return "WARN ";
        case LogLevel::ERROR: return "ERROR";
        case LogLevel::FATAL: return "FATAL";
        case LogLevel::OFF:   return "OFF  ";
        default: return "UNKNOWN";
    }
}

/**
 * @brief Parse log level from string
 */
inline LogLevel LogLevelFromString(const std::string& str) {
    if (str == "TRACE" || str == "trace") return LogLevel::TRACE;
    if (str == "DEBUG" || str == "debug") return LogLevel::DEBUG;
    if (str == "INFO" || str == "info") return LogLevel::INFO;
    if (str == "WARN" || str == "warn" || str == "WARNING" || str == "warning") return LogLevel::WARN;
    if (str == "ERROR" || str == "error") return LogLevel::ERROR;
    if (str == "FATAL" || str == "fatal") return LogLevel::FATAL;
    if (str == "OFF" || str == "off") return LogLevel::OFF;
    return LogLevel::INFO; // default
}

/**
 * @brief Centralized logger for DSSTNE
 * 
 * Thread-safe singleton logger with configurable log levels and output destinations.
 * Supports console and file logging.
 */
class Logger {
public:
    /**
     * @brief Get the singleton logger instance
     */
    static Logger& instance() {
        static Logger logger;
        return logger;
    }

    /**
     * @brief Set the minimum log level
     */
    void setLevel(LogLevel level) {
        std::lock_guard<std::mutex> lock(_mutex);
        _level = level;
    }

    /**
     * @brief Get the current log level
     */
    LogLevel getLevel() const {
        return _level;
    }

    /**
     * @brief Check if a log level is enabled
     */
    bool isEnabled(LogLevel level) const {
        return level >= _level;
    }

    /**
     * @brief Enable/disable console output
     */
    void setConsoleEnabled(bool enabled) {
        std::lock_guard<std::mutex> lock(_mutex);
        _consoleEnabled = enabled;
    }

    /**
     * @brief Set log file path (enables file logging)
     */
    bool setLogFile(const std::string& path) {
        std::lock_guard<std::mutex> lock(_mutex);
        if (_fileStream.is_open()) {
            _fileStream.close();
        }
        _fileStream.open(path, std::ios::app);
        _fileEnabled = _fileStream.is_open();
        return _fileEnabled;
    }

    /**
     * @brief Close log file
     */
    void closeLogFile() {
        std::lock_guard<std::mutex> lock(_mutex);
        if (_fileStream.is_open()) {
            _fileStream.close();
        }
        _fileEnabled = false;
    }

    /**
     * @brief Log a message at the specified level
     */
    void log(LogLevel level, const std::string& message, 
             const std::string& file = "", int line = 0,
             const std::string& function = "") {
        if (!isEnabled(level)) return;

        std::lock_guard<std::mutex> lock(_mutex);
        std::string formattedMessage = formatMessage(level, message, file, line, function);

        if (_consoleEnabled) {
            std::ostream& out = (level >= LogLevel::ERROR) ? std::cerr : std::cout;
            out << formattedMessage << std::endl;
        }

        if (_fileEnabled && _fileStream.is_open()) {
            _fileStream << formattedMessage << std::endl;
            _fileStream.flush();
        }
    }

    /**
     * @brief Log a message with format arguments (printf-style)
     */
    template<typename... Args>
    void logf(LogLevel level, const char* format, Args... args) {
        if (!isEnabled(level)) return;
        
        char buffer[4096];
        snprintf(buffer, sizeof(buffer), format, args...);
        log(level, std::string(buffer));
    }

    // Convenience methods for each log level
    void trace(const std::string& msg) { log(LogLevel::TRACE, msg); }
    void debug(const std::string& msg) { log(LogLevel::DEBUG, msg); }
    void info(const std::string& msg)  { log(LogLevel::INFO, msg); }
    void warn(const std::string& msg)  { log(LogLevel::WARN, msg); }
    void error(const std::string& msg) { log(LogLevel::ERROR, msg); }
    void fatal(const std::string& msg) { log(LogLevel::FATAL, msg); }

private:
    Logger() : _level(LogLevel::INFO), _consoleEnabled(true), _fileEnabled(false) {
        // Check for environment variable to set log level
        const char* envLevel = std::getenv("DSSTNE_LOG_LEVEL");
        if (envLevel) {
            _level = LogLevelFromString(envLevel);
        }

        // Check for environment variable to set log file
        const char* envFile = std::getenv("DSSTNE_LOG_FILE");
        if (envFile) {
            setLogFile(envFile);
        }
    }

    ~Logger() {
        closeLogFile();
    }

    // Non-copyable
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    std::string formatMessage(LogLevel level, const std::string& message,
                              const std::string& file, int line,
                              const std::string& function) {
        std::ostringstream oss;
        
        // Timestamp
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;
        
        oss << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S")
            << '.' << std::setfill('0') << std::setw(3) << ms.count();

        // Log level
        oss << " [" << LogLevelToString(level) << "]";

        // Source location (if provided)
        if (!file.empty()) {
            // Extract just the filename
            size_t pos = file.find_last_of("/\\");
            std::string filename = (pos != std::string::npos) ? file.substr(pos + 1) : file;
            oss << " " << filename << ":" << line;
            if (!function.empty()) {
                oss << " (" << function << ")";
            }
        }

        // Message
        oss << " - " << message;

        return oss.str();
    }

    LogLevel _level;
    bool _consoleEnabled;
    bool _fileEnabled;
    std::ofstream _fileStream;
    std::mutex _mutex;
};

// Global convenience function
inline Logger& getLogger() {
    return Logger::instance();
}

// Logging macros with source location
#define LOG_TRACE(msg) \
    dsstne::getLogger().log(dsstne::LogLevel::TRACE, msg, __FILE__, __LINE__, __func__)

#define LOG_DEBUG(msg) \
    dsstne::getLogger().log(dsstne::LogLevel::DEBUG, msg, __FILE__, __LINE__, __func__)

#define LOG_INFO(msg) \
    dsstne::getLogger().log(dsstne::LogLevel::INFO, msg, __FILE__, __LINE__, __func__)

#define LOG_WARN(msg) \
    dsstne::getLogger().log(dsstne::LogLevel::WARN, msg, __FILE__, __LINE__, __func__)

#define LOG_ERROR(msg) \
    dsstne::getLogger().log(dsstne::LogLevel::ERROR, msg, __FILE__, __LINE__, __func__)

#define LOG_FATAL(msg) \
    dsstne::getLogger().log(dsstne::LogLevel::FATAL, msg, __FILE__, __LINE__, __func__)

// Conditional logging (only evaluates message if level is enabled)
#define LOG_TRACE_IF(condition, msg) \
    do { if ((condition) && dsstne::getLogger().isEnabled(dsstne::LogLevel::TRACE)) LOG_TRACE(msg); } while(0)

#define LOG_DEBUG_IF(condition, msg) \
    do { if ((condition) && dsstne::getLogger().isEnabled(dsstne::LogLevel::DEBUG)) LOG_DEBUG(msg); } while(0)

// Printf-style logging
#define LOGF_INFO(fmt, ...) \
    dsstne::getLogger().logf(dsstne::LogLevel::INFO, fmt, ##__VA_ARGS__)

#define LOGF_DEBUG(fmt, ...) \
    dsstne::getLogger().logf(dsstne::LogLevel::DEBUG, fmt, ##__VA_ARGS__)

#define LOGF_ERROR(fmt, ...) \
    dsstne::getLogger().logf(dsstne::LogLevel::ERROR, fmt, ##__VA_ARGS__)

} // namespace dsstne

#endif // DSSTNE_LOGGER_H

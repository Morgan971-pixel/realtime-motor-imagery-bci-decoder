// Every rejection the contract defines. Messages match the Rust core's, so tests can share them.
#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

namespace mi {

/// Base of every contract rejection.
class Error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

/// model.json breaks a load rule. `rule` is one of: json, keys, type, shape, value.
class ModelError : public Error {
 public:
  ModelError(std::string rule, const std::string& message)
      : Error("invalid model (" + rule + "): " + message), rule_(std::move(rule)) {}
  const std::string& rule() const noexcept { return rule_; }

 private:
  std::string rule_;
};

/// A stim value breaks the input contract (step 4).
class StimError : public Error {
 public:
  StimError(std::size_t index, const std::string& reason)
      : Error("invalid stim at sample " + std::to_string(index) + ": " + reason), index_(index) {}
  std::size_t index() const noexcept { return index_; }

 private:
  std::size_t index_;
};

/// An EEG value is NaN or infinite.
class SampleError : public Error {
 public:
  SampleError(std::size_t index, const std::string& reason)
      : Error("invalid EEG sample at sample " + std::to_string(index) + ": " + reason), index_(index) {}
  std::size_t index() const noexcept { return index_; }

 private:
  std::size_t index_;
};

/// A recording file does not follow the CSV dialect, or cannot be read.
class CsvError : public Error {
 public:
  explicit CsvError(const std::string& message) : Error("invalid CSV: " + message) {}
};

}  // namespace mi

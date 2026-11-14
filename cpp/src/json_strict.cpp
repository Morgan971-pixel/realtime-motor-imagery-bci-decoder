#include "mi_core/json_strict.hpp"

#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "mi_core/error.hpp"

namespace mi {
namespace {

using json = nlohmann::json;

/// Builds the document like nlohmann's own DOM parser, but tracks the keys of every open object.
class StrictSax : public nlohmann::json_sax<json> {
 public:
  json root;

  bool null() override { return put(nullptr); }
  bool boolean(bool v) override { return put(v); }
  bool number_integer(number_integer_t v) override { return put(v); }
  bool number_unsigned(number_unsigned_t v) override { return put(v); }
  bool number_float(number_float_t v, const string_t&) override { return put(v); }
  bool string(string_t& v) override { return put(v); }
  bool binary(binary_t&) override { throw ModelError("json", "binary values are not JSON"); }

  bool start_object(std::size_t) override { return open(json::object()); }
  bool key(string_t& k) override {
    if (!frames_.back().keys.insert(k).second) throw ModelError("keys", "repeated key \"" + k + "\"");
    key_ = k;
    return true;
  }
  bool end_object() override { return close(); }
  bool start_array(std::size_t) override { return open(json::array()); }
  bool end_array() override { return close(); }

  bool parse_error(std::size_t, const std::string&, const nlohmann::detail::exception& ex) override {
    throw ModelError("json", ex.what());
  }

 private:
  struct Frame {
    json* value;
    std::unordered_set<std::string> keys;
  };
  std::vector<Frame> frames_;
  std::string key_;

  /// Insert a value into the open container (or make it the root); returns where it now lives.
  json* insert(json v) {
    if (frames_.empty()) {
      root = std::move(v);
      return &root;
    }
    json* parent = frames_.back().value;
    if (parent->is_object()) return &((*parent)[key_] = std::move(v));
    parent->push_back(std::move(v));
    return &parent->back();
  }
  bool put(json v) {
    insert(std::move(v));
    return true;
  }
  bool open(json container) {
    frames_.push_back(Frame{insert(std::move(container)), {}});
    return true;
  }
  bool close() {
    frames_.pop_back();
    return true;
  }
};

}  // namespace

nlohmann::json parse_strict(std::string_view text) {
  // nlohmann's lexer skips a UTF-8 byte-order mark; JSON (and the Rust and Python loaders) do not
  if (text.starts_with("\xEF\xBB\xBF")) throw ModelError("json", "byte-order mark before the document");
  StrictSax sax;
  json::sax_parse(text, &sax);
  return std::move(sax.root);
}

}  // namespace mi

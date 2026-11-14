// JSON parsing for model.json that refuses repeated keys (nlohmann keeps the last one silently).
#pragma once

#include <nlohmann/json.hpp>
#include <string_view>

namespace mi {

/// Parse JSON text. Throws ModelError with rule "json" for invalid JSON or a number no double
/// can hold, and rule "keys" for a key that appears twice in one object.
nlohmann::json parse_strict(std::string_view text);

}  // namespace mi

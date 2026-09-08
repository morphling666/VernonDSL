#include "compiler_json.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <vector>

namespace vernon::compiler {
namespace {

llvm::json::Value canonicalized(const llvm::json::Value &value) {
    if (const llvm::json::Object *object = value.getAsObject()) {
        std::vector<std::string> keys;
        keys.reserve(object->size());
        for (const auto &[key, unused] : *object) {
            (void)unused;
            keys.push_back(key.str());
        }
        std::sort(keys.begin(), keys.end());
        llvm::json::Object result;
        for (const std::string &key : keys)
            result[key] = canonicalized(*object->get(key));
        return result;
    }
    if (const llvm::json::Array *array = value.getAsArray()) {
        llvm::json::Array result;
        result.reserve(array->size());
        for (const llvm::json::Value &element : *array)
            result.emplace_back(canonicalized(element));
        return result;
    }
    return value;
}

} // namespace

llvm::json::Array copyJsonArray(const llvm::json::Array &source) {
    llvm::json::Array result;
    result.reserve(source.size());
    for (const llvm::json::Value &value : source)
        result.emplace_back(value);
    return result;
}

llvm::json::Object copyJsonObject(const llvm::json::Object &source) {
    llvm::json::Object result;
    for (const auto &[key, value] : source)
        result[key] = value;
    return result;
}

const llvm::json::Object *findJsonObjectByIntegerId(const llvm::json::Array &objects, int64_t id) {
    for (const llvm::json::Value &value : objects)
        if (const llvm::json::Object *object = value.getAsObject(); object && object->getInteger("id") == id)
            return object;
    return nullptr;
}

std::string canonicalJsonSha256(const llvm::json::Value &value) {
    std::string bytes;
    llvm::raw_string_ostream stream(bytes);
    stream << canonicalized(value);
    llvm::SHA256 hash;
    hash.update(bytes);
    return llvm::toHex(hash.final(), true);
}

} // namespace vernon::compiler

#include "compiler_program_builtin.h"

#include "VernonVersions.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/SHA256.h"

#include <functional>
#include <sstream>
#include <vector>

namespace vernon::compiler {
namespace {

std::vector<std::string> tupleMembers(llvm::StringRef type) {
    type = type.trim();
    if (!type.consume_front("tuple<") || !type.consume_back(">"))
        return {};
    std::vector<std::string> members;
    size_t begin = 0;
    unsigned depth = 0;
    for (size_t index = 0; index <= type.size(); ++index) {
        const char character = index < type.size() ? type[index] : ',';
        if (character == '<' || character == '[' || character == '(')
            ++depth;
        else if (character == '>' || character == ']' || character == ')')
            --depth;
        else if (character == ',' && depth == 0) {
            members.push_back(type.slice(begin, index).trim().str());
            begin = index + 1;
        }
    }
    return members;
}

std::string builtinIdentity(llvm::StringRef operation, llvm::StringRef elementType, uint32_t rank,
                            llvm::ArrayRef<std::string> leafDtypes) {
    if (leafDtypes.size() == 1 && leafDtypes.front() == elementType)
        return elementType.str();
    llvm::SHA256 hash;
    std::string payload = operation.str();
    payload.push_back('\0');
    payload += elementType.str();
    payload.push_back('\0');
    payload += std::to_string(rank);
    hash.update(payload);
    return llvm::toHex(hash.final(), true).substr(0, 16);
}

} // namespace

bool buildProgramBuiltinMlir(llvm::StringRef operation, llvm::StringRef elementType, uint32_t rank,
                             llvm::ArrayRef<std::string> leafDtypes, std::string &entry, std::string &mlir,
                             std::string &error) {
    if ((operation != "add" && operation != "copy") || elementType.empty() || leafDtypes.empty()) {
        error = "Program built-in operation, element type, and ABI leaves must be valid";
        return false;
    }
    entry = "_program_" + operation.str() + "_" + builtinIdentity(operation, elementType, rank, leafDtypes) + "_rank" +
            std::to_string(rank);
    std::ostringstream shapeStream;
    shapeStream << '[';
    for (uint32_t axis = 0; axis < rank; ++axis) {
        if (axis)
            shapeStream << ", ";
        shapeStream << "-1";
    }
    shapeStream << ']';
    const std::string shape = shapeStream.str();
    std::ostringstream leavesStream;
    for (size_t index = 0; index < leafDtypes.size(); ++index) {
        if (index)
            leavesStream << ", ";
        leavesStream << '"' << leafDtypes[index] << '"';
    }
    const std::vector<std::string> parameters = operation == "add" ? std::vector<std::string>{"output", "left", "right"}
                                                                   : std::vector<std::string>{"output", "source"};
    std::vector<std::string> arguments;
    for (size_t index = 0; index < parameters.size(); ++index) {
        const std::string access = index == 0 ? "write" : "read";
        std::ostringstream argument;
        argument << "%arg" << index << ": !vernon.tensor_view<" << elementType.str() << ", " << shape << ", \""
                 << access << "\", \"device\"> {vernon.source_name = \"" << parameters[index]
                 << "\", vernon.element_abi_leaf_dtypes = [" << leavesStream.str()
                 << "], vernon.interface = \"resource\", vernon.set = 0 : i64, vernon.binding = " << index << " : i64}";
        arguments.push_back(argument.str());
    }
    if (rank)
        arguments.push_back("%arg" + std::to_string(arguments.size()) +
                            ": tensor<3xi32> {vernon.interface = \"input\", "
                            "vernon.builtin = \"global_invocation_id\", vernon.source_name = \"gid\", "
                            "vernon.dtype = \"u32\", vernon.abi_leaf_dtypes = [\"u32\"], "
                            "vernon.element_abi_leaf_dtypes = [\"u32\"]}");

    std::vector<std::string> lines;
    uint32_t nextValue = 0;
    auto emit = [&](const std::string &expression) {
        const std::string result = "%" + std::to_string(nextValue++);
        lines.push_back("    " + result + " = " + expression);
        return result;
    };
    std::vector<std::string> indices;
    if (rank) {
        const std::string zero = emit("arith.constant 0 : i32");
        const std::string zeroIndex = emit("arith.index_cast " + zero + " : i32 to index");
        std::string linear =
            emit("tensor.extract %arg" + std::to_string(arguments.size() - 1) + "[" + zeroIndex + "] : tensor<3xi32>");
        std::vector<std::string> coordinates(rank);
        if (rank > 1) {
            const std::string outputType =
                "!vernon.tensor_view<" + elementType.str() + ", " + shape + ", \"write\", \"device\">";
            const std::string runtimeShape =
                emit("\"vernon.get_shape\"(%arg0) : (" + outputType + ") -> tensor<" + std::to_string(rank) + "xi32>");
            for (uint32_t axis = rank - 1; axis > 0; --axis) {
                const std::string axisValue = emit("arith.constant " + std::to_string(axis) + " : i32");
                const std::string axisIndex = emit("arith.index_cast " + axisValue + " : i32 to index");
                const std::string extent = emit("tensor.extract " + runtimeShape + "[" + axisIndex + "] : tensor<" +
                                                std::to_string(rank) + "xi32>");
                coordinates[axis] = emit("arith.remui " + linear + ", " + extent + " : i32");
                linear = emit("arith.divui " + linear + ", " + extent + " : i32");
            }
        }
        coordinates[0] = linear;
        for (const std::string &coordinate : coordinates)
            indices.push_back(emit("arith.index_cast " + coordinate + " : i32 to index"));
    }
    std::string indexOperands;
    std::string indexTypes;
    for (const std::string &index : indices) {
        indexOperands += ", " + index;
        indexTypes += ", index";
    }
    const auto load = [&](size_t argument, llvm::StringRef access) {
        const std::string view =
            "!vernon.tensor_view<" + elementType.str() + ", " + shape + ", \"" + access.str() + "\", \"device\">";
        return emit("\"vernon.load\"(%arg" + std::to_string(argument) + indexOperands + ") : (" + view + indexTypes +
                    ") -> " + elementType.str());
    };
    const std::string left = load(1, "read");
    std::string result = left;
    if (operation == "add") {
        const std::string right = load(2, "read");
        std::function<std::string(const std::string &, const std::string &, llvm::StringRef)> addValues;
        addValues = [&](const std::string &lhs, const std::string &rhs, llvm::StringRef type) {
            const std::vector<std::string> members = tupleMembers(type);
            if (members.empty())
                return emit("arith.addf " + lhs + ", " + rhs + " : " + type.str());
            std::vector<std::string> values;
            for (size_t index = 0; index < members.size(); ++index) {
                const std::string leftMember =
                    emit("\"vernon.tuple_get\"(" + lhs + ") {index = " + std::to_string(index) + " : i64} : (" +
                         type.str() + ") -> " + members[index]);
                const std::string rightMember =
                    emit("\"vernon.tuple_get\"(" + rhs + ") {index = " + std::to_string(index) + " : i64} : (" +
                         type.str() + ") -> " + members[index]);
                values.push_back(addValues(leftMember, rightMember, members[index]));
            }
            std::string operands;
            std::string types;
            for (size_t index = 0; index < values.size(); ++index) {
                if (index) {
                    operands += ", ";
                    types += ", ";
                }
                operands += values[index];
                types += members[index];
            }
            return emit("\"vernon.tuple_create\"(" + operands + ") : (" + types + ") -> " + type.str());
        };
        result = addValues(left, right, elementType);
    }
    const std::string outputType =
        "!vernon.tensor_view<" + elementType.str() + ", " + shape + ", \"write\", \"device\">";
    lines.push_back("    \"vernon.store\"(" + result + ", %arg0" + indexOperands + ") : (" + elementType.str() + ", " +
                    outputType + indexTypes + ") -> ()");
    lines.push_back("    func.return");
    const std::string effects =
        operation == "add"
            ? "[{kind = \"read\", owner = \"left\", region = \"unknown\"}, {kind = \"read\", owner = \"right\", "
              "region = \"unknown\"}, {kind = \"write\", owner = \"output\", region = \"unknown\"}]"
            : "[{kind = \"read\", owner = \"source\", region = \"unknown\"}, {kind = \"write\", owner = \"output\", "
              "region = \"unknown\"}]";
    std::ostringstream output;
    output << "module attributes {vernon.frontend = \"program\", vernon.compiler_contract_version = "
           << VERNON_COMPILER_CONTRACT_VERSION << " : i64, vernon.pipeline_version = " << VERNON_PIPELINE_VERSION
           << " : i64} {\n  func.func @" << entry << '(';
    for (size_t index = 0; index < arguments.size(); ++index) {
        if (index)
            output << ", ";
        output << arguments[index];
    }
    output << ") attributes {vernon.entry, vernon.stage = \"compute\", vernon.storage_effects = " << effects
           << ", vernon.workgroup_size = array<i32: 1, 1, 1>} {\n";
    for (const std::string &line : lines)
        output << line << '\n';
    output << "  }\n}\n";
    mlir = output.str();
    return true;
}

} // namespace vernon::compiler

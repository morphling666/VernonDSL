#include "VernonCompiler.h"
#include "compiler_dispatch.h"
#include "compiler_frontend.h"
#include "compiler_internal.h"

#include <memory>
#include <new>
#include <string>
#include <string_view>
#include <vector>

struct VernonCompilerContext {
    vernon::compiler::CompilerFrontend *frontend{};
};

struct VernonCompileResult {
    VernonStatus status{VERNON_STATUS_INTERNAL_ERROR};
    std::string diagnostics;
    std::vector<vernon::compiler::Artifact> artifacts;
    std::string reflection;
    vernon::compiler::CpuExecutionStatePtr cpuExecution;
};

namespace {

VernonStringView viewOf(const std::string &value) { return VernonStringView{value.data(), value.size()}; }

std::unique_ptr<VernonCompileResult> validate(VernonCompilerContext *context, const char *source, size_t sourceSize) {
    auto result = std::make_unique<VernonCompileResult>();
    if (!context || (!source && sourceSize != 0)) {
        result->status = VERNON_STATUS_INVALID_ARGUMENT;
        result->diagnostics = "context and source must be valid";
        return result;
    }
    result->status = vernon::compiler::validateMlir(*context->frontend, source, sourceSize, result->artifacts,
                                                    result->reflection, result->diagnostics);
    return result;
}

} // namespace

extern "C" {

VernonCompilerContext *vernonCompilerCreate(void) {
    std::unique_ptr<VernonCompilerContext> context(new (std::nothrow) VernonCompilerContext());
    if (!context)
        return nullptr;
    context->frontend = vernon::compiler::createCompilerFrontend();
    return context->frontend ? context.release() : nullptr;
}

void vernonCompilerDestroy(VernonCompilerContext *context) {
    if (!context)
        return;
    vernon::compiler::destroyCompilerFrontend(context->frontend);
    delete context;
}

VernonTargetCapabilities vernonCompilerGetTargetCapabilities(const VernonCompilerContext *context,
                                                             VernonTarget target) {
    if (!context || target < VERNON_TARGET_CPU || target > VERNON_TARGET_CUDA)
        return VernonTargetCapabilities{0, 0, 0, 0};
    return vernon::compiler::targetCapabilities(target);
}

VernonCompileResult *vernonCompilerValidateMlir(VernonCompilerContext *context, const char *source, size_t sourceSize) {
    return validate(context, source, sourceSize).release();
}

VernonCompileResult *vernonCompilerCompileMlir(VernonCompilerContext *context, const char *source, size_t sourceSize,
                                               VernonTarget target) {
    return vernonCompilerCompileMlirWithOptions(context, source, sourceSize, target, nullptr);
}

VernonCompileResult *vernonCompilerCompileMlirWithOptions(VernonCompilerContext *context, const char *source,
                                                          size_t sourceSize, VernonTarget target,
                                                          const VernonCompileOptions *options) {
    auto result = validate(context, source, sourceSize);
    if (result->status != VERNON_STATUS_OK)
        return result.release();

    if (target < VERNON_TARGET_CPU || target > VERNON_TARGET_CUDA) {
        result->status = VERNON_STATUS_INVALID_ARGUMENT;
        result->diagnostics = "unknown compilation target";
        result->artifacts.clear();
        return result.release();
    }

    vernon::compiler::CompileOptions parsedOptions;
    result->status = vernon::compiler::parseCompileOptions(options, target, parsedOptions, result->diagnostics);
    if (result->status != VERNON_STATUS_OK) {
        result->artifacts.clear();
        return result.release();
    }
    result->status = vernon::compiler::compileTarget(*context->frontend, source, sourceSize, target, parsedOptions,
                                                     result->artifacts, result->reflection, result->diagnostics,
                                                     result->cpuExecution);
    return result.release();
}

VernonCompileResult *vernonCompilerLinkHostObject(VernonCompilerContext *context, const void *object,
                                                  size_t objectSize) {
    auto result = std::make_unique<VernonCompileResult>();
    if (!context || !object || objectSize == 0) {
        result->status = VERNON_STATUS_INVALID_ARGUMENT;
        result->diagnostics = "host object linking requires a compiler context and object bytes";
        return result.release();
    }
    vernon::compiler::Artifact artifact;
    if (!vernon::compiler::linkCpuHostObject(object, objectSize, artifact, result->diagnostics)) {
        result->status = VERNON_STATUS_INTERNAL_ERROR;
        return result.release();
    }
    result->artifacts.push_back(std::move(artifact));
    result->status = VERNON_STATUS_OK;
    return result.release();
}

void vernonCompileResultDestroy(VernonCompileResult *result) { delete result; }

VernonStatus vernonCompileResultGetStatus(const VernonCompileResult *result) {
    return result ? result->status : VERNON_STATUS_INVALID_ARGUMENT;
}

VernonStringView vernonCompileResultGetDiagnostics(const VernonCompileResult *result) {
    return result ? viewOf(result->diagnostics) : VernonStringView{nullptr, 0};
}

VernonStringView vernonCompileResultGetArtifact(const VernonCompileResult *result) {
    return result && !result->artifacts.empty() ? viewOf(result->artifacts.front().data) : VernonStringView{nullptr, 0};
}

size_t vernonCompileResultGetArtifactCount(const VernonCompileResult *result) {
    return result ? result->artifacts.size() : 0;
}

VernonStringView vernonCompileResultGetArtifactName(const VernonCompileResult *result, size_t index) {
    return result && index < result->artifacts.size() ? viewOf(result->artifacts[index].name)
                                                      : VernonStringView{nullptr, 0};
}

VernonStringView vernonCompileResultGetArtifactData(const VernonCompileResult *result, size_t index) {
    return result && index < result->artifacts.size() ? viewOf(result->artifacts[index].data)
                                                      : VernonStringView{nullptr, 0};
}

VernonStringView vernonCompileResultGetReflection(const VernonCompileResult *result) {
    return result ? viewOf(result->reflection) : VernonStringView{nullptr, 0};
}

VernonCpuEntryPoint vernonCompileResultGetCpuEntry(const VernonCompileResult *result, const char *entryName,
                                                   size_t entryNameSize) {
    if (!result || result->status != VERNON_STATUS_OK || (!entryName && entryNameSize != 0))
        return nullptr;
    return vernon::compiler::findCompiledCpuEntry(result->cpuExecution.get(),
                                                  std::string_view(entryName ? entryName : "", entryNameSize));
}

} // extern "C"

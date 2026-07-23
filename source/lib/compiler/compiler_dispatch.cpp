#include "compiler_dispatch.h"

#include "compiler_artifacts.h"
#include "compiler_cpu.h"
#include "compiler_cuda.h"
#include "compiler_frontend.h"
#include "compiler_spirv.h"
#include "compiler_spirv_cross.h"

#include <cstring>
#include <string>

namespace vernon::compiler {
namespace {

std::string copyStringView(VernonStringView value) {
    return value.data && value.size ? std::string(value.data, value.size) : std::string();
}

} // namespace

VernonTargetCapabilities targetCapabilities(VernonTarget target) {
    if (target == VERNON_TARGET_CPU || target == VERNON_TARGET_VULKAN)
        return VernonTargetCapabilities{1, 1, 1, 0};
    if (target == VERNON_TARGET_CUDA)
        return VernonTargetCapabilities{1, 0, 1, 0};
    if (target == VERNON_TARGET_OPENGL || target == VERNON_TARGET_OPENGL_ES || target == VERNON_TARGET_METAL)
        return VernonTargetCapabilities{1, 1, 1, 0};
    // Do not advertise an IR-only path as a usable target; availability means
    // the complete lowering and artifact pipeline is linked.
    return VernonTargetCapabilities{0, 0, 0, 0};
}

VernonStatus parseCompileOptions(const VernonCompileOptions *source, VernonTarget target, CompileOptions &options,
                                 std::string &diagnostics) {
    if (!source)
        return VERNON_STATUS_OK;
    constexpr size_t requiredSize = offsetof(VernonCompileOptions, glsl_version) + sizeof(uint32_t);
    if (source->struct_size < requiredSize) {
        diagnostics = "compile options structure is too small";
        return VERNON_STATUS_INVALID_ARGUMENT;
    }
    options.glslVersion = source->glsl_version;
    if (options.glslVersion != 0 && target != VERNON_TARGET_OPENGL && target != VERNON_TARGET_OPENGL_ES) {
        diagnostics = "GLSL version is valid only for OpenGL and OpenGL ES targets";
        return VERNON_STATUS_INVALID_ARGUMENT;
    }
    if (options.glslVersion != 0 && (options.glslVersion < 100 || options.glslVersion > 999)) {
        diagnostics = "GLSL version must be a three-digit version number";
        return VERNON_STATUS_INVALID_ARGUMENT;
    }
    auto readCpuOption = [&](size_t offset, std::string &destination) {
        if (source->struct_size < offset + sizeof(VernonStringView))
            return true;
        VernonStringView value{};
        std::memcpy(&value, reinterpret_cast<const char *>(source) + offset, sizeof(value));
        if (value.size != 0 && !value.data) {
            diagnostics = "CPU compile option has null data";
            return false;
        }
        destination = copyStringView(value);
        return true;
    };
    if (!readCpuOption(offsetof(VernonCompileOptions, cpu_target_triple), options.cpu.targetTriple) ||
        !readCpuOption(offsetof(VernonCompileOptions, cpu_name), options.cpu.cpu) ||
        !readCpuOption(offsetof(VernonCompileOptions, cpu_features), options.cpu.features))
        return VERNON_STATUS_INVALID_ARGUMENT;
    if (target != VERNON_TARGET_CPU &&
        (!options.cpu.targetTriple.empty() || !options.cpu.cpu.empty() || !options.cpu.features.empty())) {
        diagnostics = "CPU code generation options are valid only for the CPU target";
        return VERNON_STATUS_INVALID_ARGUMENT;
    }
    return VERNON_STATUS_OK;
}

VernonStatus compileTarget(CompilerFrontend &frontend, const char *source, size_t sourceSize, VernonTarget target,
                           const CompileOptions &options, std::vector<Artifact> &artifacts, std::string &reflection,
                           std::string &diagnostics, CpuExecutionStatePtr &cpuExecution) {
    diagnostics.clear();
    mlir::MLIRContext &context = compilerMlirContext(frontend);
    if (target == VERNON_TARGET_CPU) {
        if (!compileCpu(context, source, sourceSize, options.cpu, artifacts, reflection, diagnostics, cpuExecution)) {
            artifacts.clear();
            return VERNON_STATUS_INTERNAL_ERROR;
        }
        addArtifactTable(reflection, artifacts, target, options.glslVersion, options.cpu.targetTriple, options.cpu.cpu,
                         options.cpu.features);
        return VERNON_STATUS_OK;
    }
    if (target == VERNON_TARGET_VULKAN || target == VERNON_TARGET_OPENGL || target == VERNON_TARGET_OPENGL_ES ||
        target == VERNON_TARGET_METAL) {
        if (!compileSpirv(context, source, sourceSize, target, artifacts, diagnostics)) {
            artifacts.clear();
            return VERNON_STATUS_INTERNAL_ERROR;
        }
        if (target != VERNON_TARGET_VULKAN && !crossCompileSpirv(artifacts, diagnostics, target, options.glslVersion)) {
            artifacts.clear();
            return VERNON_STATUS_INTERNAL_ERROR;
        }
        addArtifactTable(reflection, artifacts, target, options.glslVersion);
        return VERNON_STATUS_OK;
    }
    if (target == VERNON_TARGET_CUDA) {
        if (!compileCuda(context, source, sourceSize, artifacts, diagnostics)) {
            artifacts.clear();
            return VERNON_STATUS_INTERNAL_ERROR;
        }
        addArtifactTable(reflection, artifacts, target, options.glslVersion);
        return VERNON_STATUS_OK;
    }
    artifacts.clear();
    diagnostics = "the requested target lowering pipeline is not available";
    return VERNON_STATUS_UNSUPPORTED_TARGET;
}

bool linkCpuHostObject(const void *object, size_t objectSize, Artifact &artifact, std::string &diagnostics) {
    return linkHostObject(object, objectSize, artifact, diagnostics);
}

VernonCpuEntryPoint findCompiledCpuEntry(const CpuExecutionState *execution, std::string_view entry) {
    return findCpuEntry(execution, entry);
}

} // namespace vernon::compiler

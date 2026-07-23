#include "graphics_opengl_encoder.h"
#include "backend_opengl.h"
#include "runtime_state.h"
#include "tensor_bridge.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <optional>
#include <vector>

namespace vernon::runtime {
namespace {

VernonStatus fail(std::string &error, const std::string &message,
                  VernonStatus status = VERNON_STATUS_INVALID_ARGUMENT) {
    error = message;
    return status;
}

GlEnum topologyMode(VernonPrimitiveTopology topology) {
    if (topology == VERNON_TOPOLOGY_LINE_LIST)
        return kLines;
    if (topology == VERNON_TOPOLOGY_POINT_LIST)
        return kPoints;
    return kTriangles;
}

} // namespace

VernonStatus encodeAndSubmitOpenGLGraphics(const OpenGLGraphicsState &state, const Variant &variant,
                                           const VernonPipelineInvocation &invocation,
                                           const PlannedGraphicsInvocation &plan, std::string &error) {
    OpenGLDriver &gl = openGLState(*state.context).driver;
    gl.useProgram(state.program);
    gl.bindVertexArray(state.vertexArray);
    const bool hasViewport = invocation.viewport[2] && invocation.viewport[3];
    for (const Parameter &parameter : variant.parameters) {
        const VernonPipelineArgument &argument = *plan.arguments.at(parameter.slot);
        for (const ParameterUse &use : parameter.uses) {
            if (use.stage != "vertex" && use.stage != "fragment")
                continue;
            if (argument.kind == VERNON_PIPELINE_TEXTURE) {
                if (use.descriptorSet != 0 || use.binding == UINT32_MAX)
                    return fail(error, "OpenGL sampled texture requires set zero and a binding");
                const VernonDeviceTexture *texture = argument.texture.texture;
                const GlEnum target = texture->dimension == VERNON_TEXTURE_2D   ? kTexture2D
                                      : texture->dimension == VERNON_TEXTURE_3D ? kTexture3D
                                                                                : kTextureCubeMap;
                gl.activeTexture(kTexture0 + use.binding);
                gl.bindTexture(target, openGLTextureState(*texture).name);
                const std::string uniformName =
                    use.uniformName.empty() ? "main_arg_" + std::to_string(use.index) : use.uniformName;
                const GlInt location = gl.getUniformLocation(state.program, uniformName.c_str());
                if (location >= 0)
                    gl.uniform1i(location, static_cast<GlInt>(use.binding));
                continue;
            }
            if (argument.kind == VERNON_PIPELINE_SAMPLER) {
                for (const SampledTextureBinding &binding : use.sampledTextureBindings) {
                    if (binding.descriptorSet != 0 || binding.binding == UINT32_MAX)
                        return fail(error, "OpenGL sampled texture requires set zero and a binding");
                    gl.bindSampler(binding.binding, openGLSamplerState(*argument.sampler).name);
                }
                continue;
            }
            if (use.interfaceKind == "uniform") {
                if (argument.kind != VERNON_PIPELINE_TENSOR || argument.tensor.storage != VERNON_TENSOR_HOST ||
                    argument.tensor.dtype != VERNON_DATA_F32)
                    return fail(error, "graphics uniform is invalid");
                const GlInt location = gl.getUniformLocation(state.program, use.uniformName.c_str());
                if (location < 0)
                    return fail(error, "graphics uniform location is missing");
                const std::optional<size_t> elementCount = tensorElementCount(argument.tensor);
                if (!elementCount)
                    return fail(error, "graphics uniform is invalid");
                const uint32_t count = static_cast<uint32_t>(*elementCount);
                const float *data = reinterpret_cast<const float *>(hostTensorData(argument.tensor));
                std::optional<std::vector<uint8_t>> packed;
                if (count <= 4 && !isRowMajorContiguous(argument.tensor)) {
                    packed = packTensorRowMajor(argument.tensor);
                    if (!packed)
                        return fail(error, "failed to pack graphics uniform");
                    data = reinterpret_cast<const float *>(packed->data());
                }
                if (count == 1)
                    gl.uniform1fv(location, 1, data);
                else if (count == 2)
                    gl.uniform2fv(location, 1, data);
                else if (count == 3)
                    gl.uniform3fv(location, 1, data);
                else if (count == 4)
                    gl.uniform4fv(location, 1, data);
                else if (count == 9 || count == 16) {
                    const uint32_t dimension = count == 9 ? 3 : 4;
                    if (argument.tensor.rank != 2 || argument.tensor.shape[0] != dimension ||
                        argument.tensor.shape[1] != dimension)
                        return fail(error, "graphics matrix uniform shape is unsupported");
                    const bool columnMajor = argument.tensor.byte_strides[0] == sizeof(float) &&
                                             argument.tensor.byte_strides[1] == dimension * sizeof(float);
                    const bool rowMajor = argument.tensor.byte_strides[1] == sizeof(float) &&
                                          argument.tensor.byte_strides[0] == dimension * sizeof(float);
                    GlBoolean transpose = 0;
                    std::vector<float> matrix;
                    if (!columnMajor && !(rowMajor && state.context->backend != VERNON_RUNTIME_OPENGL_ES)) {
                        matrix.resize(count);
                        const uint8_t *source = hostTensorData(argument.tensor);
                        for (uint32_t column = 0; column < dimension; ++column)
                            for (uint32_t row = 0; row < dimension; ++row)
                                std::memcpy(&matrix[column * dimension + row],
                                            source + row * argument.tensor.byte_strides[0] +
                                                column * argument.tensor.byte_strides[1],
                                            sizeof(float));
                        data = matrix.data();
                    } else if (rowMajor) {
                        transpose = 1;
                    }
                    if (dimension == 3)
                        gl.uniformMatrix3fv(location, 1, transpose, data);
                    else
                        gl.uniformMatrix4fv(location, 1, transpose, data);
                } else {
                    return fail(error, "graphics uniform size is unsupported");
                }
                continue;
            }
            if (use.interfaceKind != "input" && use.interfaceKind != "instance")
                continue;
            const auto planned = std::find_if(plan.vertexInputs.begin(), plan.vertexInputs.end(),
                                              [&](const PlannedVertexInput &input) { return input.use == &use; });
            if (planned == plan.vertexInputs.end())
                return fail(error, "planned graphics Tensor input is missing", VERNON_STATUS_INTERNAL_ERROR);
            const VernonTensorView &tensor = *planned->tensor;
            gl.bindBuffer(kArrayBuffer, openGLBufferState(*tensor.buffer).name);
            if (planned->components <= 4) {
                gl.enableVertexAttribArray(use.location);
                gl.vertexAttribPointer(use.location, static_cast<GlInt>(planned->components), kFloat, 0,
                                       static_cast<GlSize>(tensor.byte_strides[0]),
                                       reinterpret_cast<const void *>(tensor.byte_offset));
                gl.vertexAttribDivisor(use.location, planned->instanced ? std::max(use.divisor, 1u) : 0);
            } else {
                const uint32_t columns = static_cast<uint32_t>(tensor.shape[1]);
                const uint32_t rows = static_cast<uint32_t>(tensor.shape[2]);
                for (uint32_t column = 0; column < columns; ++column) {
                    gl.enableVertexAttribArray(use.location + column);
                    gl.vertexAttribPointer(
                        use.location + column, static_cast<GlInt>(rows), kFloat, 0,
                        static_cast<GlSize>(tensor.byte_strides[0]),
                        reinterpret_cast<const void *>(tensor.byte_offset + column * tensor.byte_strides[1]));
                    gl.vertexAttribDivisor(use.location + column, planned->instanced ? std::max(use.divisor, 1u) : 0);
                }
            }
        }
    }
    for (const Parameter &parameter : variant.internalParameters) {
        for (const ParameterUse &use : parameter.uses) {
            if (use.stage != "vertex" && use.stage != "fragment")
                continue;
            if (parameter.source == "implicit_sampler") {
                for (const SampledTextureBinding &binding : use.sampledTextureBindings) {
                    if (binding.descriptorSet != 0 || binding.binding == UINT32_MAX)
                        return fail(error, "OpenGL implicit sampler requires set zero and a binding");
                    const auto found = plan.sampledResources.find({binding.descriptorSet, binding.binding});
                    if (found == plan.sampledResources.end())
                        return fail(error, "planned sampler binding is missing", VERNON_STATUS_INTERNAL_ERROR);
                    // Sampler object zero selects the context's texture sampling state.
                    gl.bindSampler(binding.binding,
                                   found->second.sampler ? openGLSamplerState(*found->second.sampler).name : 0);
                }
                continue;
            }
            if (parameter.systemValue == "resolution") {
                if (use.interfaceKind != "uniform")
                    return fail(error, "OpenGL resolution system value must be a uniform");
                const GlInt location = gl.getUniformLocation(state.program, use.uniformName.c_str());
                if (location < 0)
                    return fail(error, "resolution system uniform location is missing");
                gl.uniform2fv(location, 1, plan.resolution.data());
            }
        }
    }

    gl.bindFramebuffer(kFramebuffer, state.framebuffer);
    for (const VernonColorAttachment *attachment : plan.attachments)
        gl.framebufferTexture2D(kFramebuffer, kColorAttachment0 + attachment->location, kTexture2D,
                                openGLTextureState(*attachment->texture).name, 0);
    std::vector<GlEnum> drawBuffers(plan.maximumAttachmentLocation + 1, kNone);
    for (const VernonColorAttachment *attachment : plan.attachments)
        drawBuffers[attachment->location] = kColorAttachment0 + attachment->location;
    gl.drawBuffers(static_cast<GlSize>(drawBuffers.size()), drawBuffers.data());
    if (gl.checkFramebufferStatus(kFramebuffer) != kFramebufferComplete)
        return fail(error, "OpenGL framebuffer is incomplete", VERNON_STATUS_INTERNAL_ERROR);
    constexpr float clearColor[4] = {0.0F, 0.0F, 0.0F, 0.0F};
    for (const VernonColorAttachment *attachment : plan.attachments)
        gl.clearBufferfv(kColor, static_cast<GlInt>(attachment->location), clearColor);
    gl.viewport(static_cast<GlInt>(hasViewport ? invocation.viewport[0] : 0),
                static_cast<GlInt>(hasViewport ? invocation.viewport[1] : 0),
                static_cast<GlSize>(hasViewport ? invocation.viewport[2] : plan.attachmentWidth),
                static_cast<GlSize>(hasViewport ? invocation.viewport[3] : plan.attachmentHeight));

    const GlEnum mode = topologyMode(invocation.topology);
    if (plan.indexBinding) {
        const VernonIndexBinding &index = *plan.indexBinding;
        gl.bindBuffer(kElementArrayBuffer, openGLBufferState(*index.buffer).name);
        gl.drawElementsInstanced(mode, static_cast<GlSize>(index.index_count), kUnsignedInt,
                                 reinterpret_cast<const void *>(index.offset), static_cast<GlSize>(plan.instanceCount));
    } else if (plan.instanceCount == 1) {
        gl.drawArrays(mode, 0, static_cast<GlSize>(plan.vertexCount));
    } else {
        gl.drawArraysInstanced(mode, 0, static_cast<GlSize>(plan.vertexCount), static_cast<GlSize>(plan.instanceCount));
    }
    return VERNON_STATUS_OK;
}

} // namespace vernon::runtime

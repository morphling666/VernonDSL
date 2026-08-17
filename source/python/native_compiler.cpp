#include "native_compiler.h"

#include "VernonExecutionGraph.h"
#include "native_runtime.h"

void bindNativeCompiler(nb::module_ &module) {
    module.def("target_available", &targetAvailable, nb::arg("target"));
    module.def("target_capabilities", &targetCapabilities, nb::arg("target"));
    nb::enum_<VernonStatus>(module, "Status")
        .value("OK", VERNON_STATUS_OK)
        .value("INVALID_ARGUMENT", VERNON_STATUS_INVALID_ARGUMENT)
        .value("PARSE_ERROR", VERNON_STATUS_PARSE_ERROR)
        .value("VERIFICATION_ERROR", VERNON_STATUS_VERIFICATION_ERROR)
        .value("UNSUPPORTED_TARGET", VERNON_STATUS_UNSUPPORTED_TARGET)
        .value("INTERNAL_ERROR", VERNON_STATUS_INTERNAL_ERROR);
    nb::enum_<VernonRuntimeBackend>(module, "RuntimeBackend")
        .value("CPU", VERNON_RUNTIME_CPU)
        .value("CUDA", VERNON_RUNTIME_CUDA)
        .value("VULKAN", VERNON_RUNTIME_VULKAN)
        .value("OPENGL", VERNON_RUNTIME_OPENGL)
        .value("OPENGL_ES", VERNON_RUNTIME_OPENGL_ES)
        .value("DIRECTX12", VERNON_RUNTIME_DIRECTX12)
        .value("METAL", VERNON_RUNTIME_METAL);
    nb::enum_<VernonRhiBackend>(module, "RhiBackend")
        .value("CUDA", VERNON_RHI_BACKEND_CUDA)
        .value("VULKAN", VERNON_RHI_BACKEND_VULKAN)
        .value("DIRECTX12", VERNON_RHI_BACKEND_DIRECTX12)
        .value("OPENGL", VERNON_RHI_BACKEND_OPENGL)
        .value("OPENGL_ES", VERNON_RHI_BACKEND_OPENGL_ES)
        .value("METAL", VERNON_RHI_BACKEND_METAL);
    nb::enum_<VernonPrimitiveTopology>(module, "PrimitiveTopology")
        .value("TRIANGLE_LIST", VERNON_TOPOLOGY_TRIANGLE_LIST)
        .value("LINE_LIST", VERNON_TOPOLOGY_LINE_LIST)
        .value("POINT_LIST", VERNON_TOPOLOGY_POINT_LIST);
    nb::enum_<VernonTextureFormat>(module, "TextureFormat")
        .value("RGBA8_UNORM", VERNON_TEXTURE_RGBA8_UNORM)
        .value("RGBA8_SRGB", VERNON_TEXTURE_RGBA8_SRGB)
        .value("RGBA16_FLOAT", VERNON_TEXTURE_RGBA16_FLOAT)
        .value("RGBA32_FLOAT", VERNON_TEXTURE_RGBA32_FLOAT)
        .value("R8_UNORM", VERNON_TEXTURE_R8_UNORM)
        .value("R16_FLOAT", VERNON_TEXTURE_R16_FLOAT)
        .value("R32_FLOAT", VERNON_TEXTURE_R32_FLOAT)
        .value("RG8_UNORM", VERNON_TEXTURE_RG8_UNORM)
        .value("RGB8_UNORM", VERNON_TEXTURE_RGB8_UNORM)
        .value("R11G11B10_FLOAT", VERNON_TEXTURE_R11G11B10_FLOAT)
        .value("D32_FLOAT", VERNON_TEXTURE_D32_FLOAT)
        .value("D32_FLOAT_S8_UINT", VERNON_TEXTURE_D32_FLOAT_S8_UINT);
    nb::enum_<VernonTextureDimension>(module, "TextureDimension")
        .value("TEXTURE_2D", VERNON_TEXTURE_2D)
        .value("TEXTURE_3D", VERNON_TEXTURE_3D)
        .value("CUBE", VERNON_TEXTURE_CUBE);
    nb::enum_<VernonRhiSamplerAddressMode>(module, "SamplerAddressMode")
        .value("REPEAT", VERNON_RHI_ADDRESS_REPEAT)
        .value("CLAMP_TO_EDGE", VERNON_RHI_ADDRESS_CLAMP_TO_EDGE)
        .value("MIRRORED_REPEAT", VERNON_RHI_ADDRESS_MIRRORED_REPEAT);
    module.attr("IMAGE_COLOR_ATTACHMENT") = static_cast<uint32_t>(VERNON_RHI_IMAGE_COLOR_ATTACHMENT);
    module.attr("IMAGE_DEPTH_STENCIL_ATTACHMENT") = static_cast<uint32_t>(VERNON_RHI_IMAGE_DEPTH_STENCIL_ATTACHMENT);
    module.attr("IMAGE_TRANSFER_SOURCE") = static_cast<uint32_t>(VERNON_RHI_IMAGE_TRANSFER_SOURCE);
    module.attr("IMAGE_TRANSFER_DESTINATION") = static_cast<uint32_t>(VERNON_RHI_IMAGE_TRANSFER_DESTINATION);
    module.attr("IMAGE_SAMPLED") = static_cast<uint32_t>(VERNON_RHI_IMAGE_SAMPLED);
    module.attr("IMAGE_STORAGE") = static_cast<uint32_t>(VERNON_RHI_IMAGE_STORAGE);
    module.attr("IMAGE_ASPECT_COLOR") = static_cast<uint32_t>(VERNON_RHI_IMAGE_ASPECT_COLOR);
    module.attr("IMAGE_ASPECT_DEPTH") = static_cast<uint32_t>(VERNON_RHI_IMAGE_ASPECT_DEPTH);
    module.attr("IMAGE_ASPECT_STENCIL") = static_cast<uint32_t>(VERNON_RHI_IMAGE_ASPECT_STENCIL);
    module.def("_plan_value_abi", &planValueAbi, nb::arg("module"), nb::arg("logical_dtypes"));
    nb::class_<StructuredVjp>(module, "_StructuredVjp")
        .def_prop_ro("tape_bytes", &StructuredVjp::tapeBytes)
        .def_prop_ro("active_operation_count", &StructuredVjp::activeOperationCount)
        .def_prop_ro("recomputation_cost", &StructuredVjp::recomputationCost)
        .def_prop_ro("derivative_rules", &StructuredVjp::derivativeRules)
        .def_prop_ro("required_primal_paths", &StructuredVjp::requiredPrimalPaths)
        .def_prop_ro("source_kind_counts", &StructuredVjp::sourceKindCounts)
        .def_prop_ro("cost_components", &StructuredVjp::costComponents)
        .def_prop_ro("selected_policy", &StructuredVjp::selectedPolicy)
        .def_prop_ro("whole_dispatch_retention_permitted", &StructuredVjp::wholeDispatchRetentionPermitted)
        .def("profiles", &StructuredVjp::profiles, nb::arg("identity"));
    module.def("_build_structured_vjp", &buildStructuredVjp, nb::arg("module"), nb::arg("entry"), nb::arg("wrt_paths"),
               nb::arg("output_paths"), nb::arg("forward_symbol"), nb::arg("backward_symbol"));
    nb::class_<Compiler>(module, "Compiler")
        .def(nb::init<>())
        .def("compile_program_result", &compileProgramResult, nb::arg("mlir"), nb::arg("target"),
             nb::arg("options") = nb::dict());
    module.def("_compile_cpu_program_results", &compileCpuProgramResults, nb::arg("modules"),
               nb::arg("options") = nb::dict());
    nb::class_<CompiledProgram>(module, "CompiledProgram")
        .def_prop_ro("ok", &CompiledProgram::ok)
        .def_prop_ro("status", &CompiledProgram::status)
        .def_prop_ro("diagnostics", &CompiledProgram::diagnostics)
        .def_prop_ro("artifacts", &CompiledProgram::artifacts)
        .def_prop_ro("reflection", &CompiledProgram::reflection)
        .def_prop_ro("target", [](const CompiledProgram &value) { return value.target; })
        .def("has_cpu_entry", &CompiledProgram::hasCpuEntry);
    nb::class_<RhiHost>(module, "RhiHost")
        .def(nb::init<VernonRhiBackend, uint32_t>(), nb::arg("backend"), nb::arg("device_index") = 0)
        .def_static("create_external_opengl", &RhiHost::createExternalOpenGL, nb::arg("backend"), nb::arg("user_data"),
                    nb::arg("make_current"), nb::arg("get_proc_address"), nb::arg("api_major"), nb::arg("api_minor"))
        .def("create_buffer", &RhiHost::createBuffer)
        .def("create_image", &RhiHost::createImage, nb::arg("width"), nb::arg("height"),
             nb::arg("format") = VERNON_TEXTURE_RGBA8_UNORM, nb::arg("dimension") = VERNON_TEXTURE_2D,
             nb::arg("depth") = 1, nb::arg("mip_levels") = 1, nb::arg("usage") = 0)
        .def("create_attachment_image", &RhiHost::createAttachmentImage)
        .def("create_sampler", &RhiHost::createSampler, nb::arg("address") = VERNON_RHI_ADDRESS_REPEAT)
        .def("create_runtime", &createRhiRuntime, nb::keep_alive<0, 1>())
        .def("create_execution_graph", [](RhiHost &host) { return createNativeExecutionGraph(host.state); });
    nb::class_<RhiBuffer>(module, "RhiBuffer")
        .def_prop_ro("size", [](const RhiBuffer &value) { return value.size; })
        .def("upload", &RhiBuffer::upload, nb::arg("data"), nb::arg("offset") = 0)
        .def("upload_ranges", &RhiBuffer::uploadRanges, nb::arg("ranges"))
        .def("download", &RhiBuffer::download);
    nb::class_<RhiImage>(module, "RhiImage")
        .def_prop_ro("width", [](const RhiImage &value) { return value.width; })
        .def_prop_ro("height", [](const RhiImage &value) { return value.height; })
        .def_prop_ro("depth", [](const RhiImage &value) { return value.depth; })
        .def_prop_ro("mip_levels", [](const RhiImage &value) { return value.mipLevels; })
        .def("upload", &RhiImage::upload, nb::arg("data"), nb::arg("mip_level") = 0, nb::arg("offset_x") = 0,
             nb::arg("offset_y") = 0, nb::arg("offset_z") = 0, nb::arg("width") = 0, nb::arg("height") = 0,
             nb::arg("depth") = 0)
        .def("download", &RhiImage::download, nb::arg("mip_level") = 0, nb::arg("offset_x") = 0,
             nb::arg("offset_y") = 0, nb::arg("offset_z") = 0, nb::arg("width") = 0, nb::arg("height") = 0,
             nb::arg("depth") = 0)
        .def("generate_mipmaps", &RhiImage::generateMipmaps)
        .def(
            "create_view",
            [](RhiImage &image, const nb::object &format, const nb::object &dimension, uint32_t baseMipLevel,
               uint32_t mipLevelCount, uint32_t baseArrayLayer, uint32_t arrayLayerCount, uint32_t aspects) {
                const VernonTextureFormat viewFormat =
                    format.is_none() ? image.format : nb::cast<VernonTextureFormat>(format);
                const VernonTextureDimension viewDimension =
                    dimension.is_none() ? image.dimension : nb::cast<VernonTextureDimension>(dimension);
                if (baseMipLevel >= image.mipLevels || baseArrayLayer >= image.layers)
                    throw std::invalid_argument("RHI image view base subresource is out of range");
                mipLevelCount = mipLevelCount ? mipLevelCount : image.mipLevels - baseMipLevel;
                arrayLayerCount = arrayLayerCount ? arrayLayerCount : image.layers - baseArrayLayer;
                if (!aspects)
                    aspects = image.format == VERNON_TEXTURE_D32_FLOAT ? VERNON_RHI_IMAGE_ASPECT_DEPTH
                              : image.format == VERNON_TEXTURE_D32_FLOAT_S8_UINT
                                  ? VERNON_RHI_IMAGE_ASPECT_DEPTH | VERNON_RHI_IMAGE_ASPECT_STENCIL
                                  : VERNON_RHI_IMAGE_ASPECT_COLOR;
                return std::make_unique<RhiImageView>(&image, viewFormat, viewDimension, baseMipLevel, mipLevelCount,
                                                      baseArrayLayer, arrayLayerCount, aspects);
            },
            nb::arg("format") = nb::none(), nb::arg("dimension") = nb::none(), nb::arg("base_mip_level") = 0,
            nb::arg("mip_level_count") = 0, nb::arg("base_array_layer") = 0, nb::arg("array_layer_count") = 0,
            nb::arg("aspects") = 0, nb::keep_alive<0, 1>());
    nb::class_<RhiImageView>(module, "RhiImageView")
        .def_prop_ro("width", [](const RhiImageView &value) { return value.width; })
        .def_prop_ro("height", [](const RhiImageView &value) { return value.height; })
        .def_prop_ro("base_mip_level", [](const RhiImageView &value) { return value.baseMipLevel; })
        .def_prop_ro("mip_level_count", [](const RhiImageView &value) { return value.mipLevelCount; })
        .def_prop_ro("base_array_layer", [](const RhiImageView &value) { return value.baseArrayLayer; })
        .def_prop_ro("array_layer_count", [](const RhiImageView &value) { return value.arrayLayerCount; })
        .def_prop_ro("aspects", [](const RhiImageView &value) { return value.aspects; });
    nb::class_<RhiSampler>(module, "RhiSampler");
    bindNativeExecutionGraph(module);
    nb::class_<Runtime>(module, "Runtime")
        .def(nb::init<VernonRuntimeBackend>(), nb::arg("backend"))
        .def("load", &Runtime::load, nb::keep_alive<0, 1>())
        .def("load_cpu_entry", &Runtime::loadCpuEntry, nb::keep_alive<0, 1>())
        .def("load_autodiff", &Runtime::loadAutodiff, nb::keep_alive<0, 1>())
        .def("load_pipeline", &Runtime::loadPipeline, nb::keep_alive<0, 1>())
        .def("load_pipeline_asset", &Runtime::loadPipelineAsset, nb::keep_alive<0, 1>())
        .def("create_execution_graph", &Runtime::createExecutionGraph);
    nb::class_<PipelineParameterMetadata>(module, "PipelineParameter")
        .def_ro("slot", &PipelineParameterMetadata::slot)
        .def_ro("name", &PipelineParameterMetadata::name)
        .def_prop_ro("kind", [](const PipelineParameterMetadata &value) { return static_cast<uint32_t>(value.kind); })
        .def_ro("element_byte_size", &PipelineParameterMetadata::elementByteSize)
        .def_ro("element_alignment", &PipelineParameterMetadata::elementAlignment)
        .def_ro("layout_hash", &PipelineParameterMetadata::layoutHash)
        .def_prop_ro("element_leaves",
                     [](const PipelineParameterMetadata &value) {
                         nb::list leaves;
                         for (const VernonValueLeafView &leaf : value.elementLeaves)
                             leaves.append(nb::make_tuple(leaf.dtype, leaf.scalar_count, leaf.byte_offset));
                         return leaves;
                     })
        .def_prop_ro("access",
                     [](const PipelineParameterMetadata &value) { return static_cast<uint32_t>(value.access); })
        .def_ro("shape", &PipelineParameterMetadata::shape);
    nb::class_<PipelineOutputMetadata>(module, "PipelineOutput")
        .def_ro("name", &PipelineOutputMetadata::name)
        .def_prop_ro("kind", [](const PipelineOutputMetadata &value) { return static_cast<uint32_t>(value.kind); })
        .def_prop_ro("dtype", [](const PipelineOutputMetadata &value) { return static_cast<uint32_t>(value.dtype); })
        .def_prop_ro("access", [](const PipelineOutputMetadata &value) { return static_cast<uint32_t>(value.access); })
        .def_ro("shape", &PipelineOutputMetadata::shape)
        .def_ro("location", &PipelineOutputMetadata::location);
    nb::class_<PythonRuntimeSubmission>(module, "Submission")
        .def("wait", &PythonRuntimeSubmission::wait, nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro("state", &PythonRuntimeSubmission::state);
    nb::class_<PreparedPipelineArgument>(module, "_PreparedPipelineArgument");
    nb::class_<PipelineInvocationBuilder>(module, "PipelineInvocationBuilder")
        .def("prepare_host_tensor", &PipelineInvocationBuilder::prepareHostTensor, nb::arg("parameter"),
             nb::arg("array"))
        .def("prepare_rhi_tensor", &PipelineInvocationBuilder::prepareRhiTensor, nb::arg("parameter"),
             nb::arg("buffer"), nb::arg("access"), nb::arg("shape"), nb::arg("strides"), nb::arg("offset") = 0)
        .def("prepare_rhi_texture", &PipelineInvocationBuilder::prepareRhiTexture, nb::arg("parameter"),
             nb::arg("texture"))
        .def("prepare_rhi_sampler", &PipelineInvocationBuilder::prepareRhiSampler, nb::arg("parameter"),
             nb::arg("sampler"))
        .def("prepared_argument", &PipelineInvocationBuilder::preparedArgument, nb::arg("argument"),
             nb::rv_policy::reference_internal, nb::keep_alive<1, 2>())
        .def("host_tensor", &PipelineInvocationBuilder::hostTensor, nb::arg("parameter"), nb::arg("array"),
             nb::rv_policy::reference_internal)
        .def("rhi_tensor", &PipelineInvocationBuilder::rhiTensor, nb::arg("parameter"), nb::arg("buffer"),
             nb::arg("access"), nb::arg("shape"), nb::arg("strides"), nb::arg("offset") = 0,
             nb::rv_policy::reference_internal, nb::keep_alive<1, 3>())
        .def("rhi_texture", &PipelineInvocationBuilder::rhiTexture, nb::arg("parameter"), nb::arg("texture"),
             nb::rv_policy::reference_internal, nb::keep_alive<1, 3>())
        .def("rhi_sampler", &PipelineInvocationBuilder::rhiSampler, nb::arg("parameter"), nb::arg("sampler"),
             nb::rv_policy::reference_internal, nb::keep_alive<1, 3>())
        .def("rhi_color_attachment", &PipelineInvocationBuilder::rhiColorAttachment, nb::arg("location"),
             nb::arg("texture"), nb::arg("load_operation"), nb::arg("store_operation"), nb::arg("clear_color"),
             nb::rv_policy::reference_internal, nb::keep_alive<1, 3>())
        .def("rhi_depth_attachment", &PipelineInvocationBuilder::rhiDepthAttachment, nb::arg("texture"),
             nb::arg("load_operation"), nb::arg("store_operation"), nb::arg("clear_depth"),
             nb::rv_policy::reference_internal, nb::keep_alive<1, 2>())
        .def("rhi_index_binding", &PipelineInvocationBuilder::rhiIndexBinding, nb::arg("buffer"), nb::arg("count"),
             nb::arg("offset") = 0, nb::rv_policy::reference_internal, nb::keep_alive<1, 2>())
        .def("topology", &PipelineInvocationBuilder::setTopology, nb::arg("topology"),
             nb::rv_policy::reference_internal)
        .def("counts", &PipelineInvocationBuilder::counts, nb::arg("vertex_count") = 0, nb::arg("instance_count") = 0,
             nb::rv_policy::reference_internal)
        .def("grid", &PipelineInvocationBuilder::grid, nb::arg("x"), nb::arg("y"), nb::arg("z"),
             nb::rv_policy::reference_internal)
        .def("viewport", &PipelineInvocationBuilder::setViewport, nb::arg("x"), nb::arg("y"), nb::arg("width"),
             nb::arg("height"), nb::rv_policy::reference_internal)
        .def("scissor", &PipelineInvocationBuilder::setScissor, nb::arg("x"), nb::arg("y"), nb::arg("width"),
             nb::arg("height"), nb::rv_policy::reference_internal)
        .def("encode", [](PipelineInvocationBuilder &builder,
                          const vernon::execution::GraphicsEncoder &encoder) { builder.encode(encoder); })
        .def("encode", [](PipelineInvocationBuilder &builder,
                          const vernon::execution::ComputeEncoder &encoder) { builder.encode(encoder); })
        .def("submit", [](PipelineInvocationBuilder &builder) { return builder.submit(); });
    nb::class_<PythonPullback>(module, "Pullback")
        .def("__call__", &PythonPullback::apply, nb::arg("cotangent") = nb::none())
        .def("apply_logical", &PythonPullback::applyLogical, nb::arg("cotangent"))
        .def("apply_grouped", &PythonPullback::applyGrouped, nb::arg("cotangent").none(), nb::arg("gradient_groups"),
             nb::arg("cotangent_groups"), nb::arg("carrier_shape"), nb::arg("logical"))
        .def_prop_ro("logical_residual_bytes", &PythonPullback::logicalResidualBytes)
        .def_prop_ro("resident_bytes", &PythonPullback::residentBytes)
        .def_prop_ro("allocated_bytes", &PythonPullback::allocatedBytes)
        .def_prop_ro("peak_temporary_bytes", &PythonPullback::peakTemporaryBytes)
        .def_prop_ro("submission_count", &PythonPullback::submissionCount)
        .def_prop_ro("wait_count", &PythonPullback::waitCount)
        .def_prop_ro("readback_count", &PythonPullback::readbackCount)
        .def_prop_ro("atomic_publication_count", &PythonPullback::atomicPublicationCount)
        .def_prop_ro("temporary_allocation_traffic_bytes", &PythonPullback::temporaryAllocationTrafficBytes)
        .def_prop_ro("device_wait_nanoseconds", &PythonPullback::deviceWaitNanoseconds);
    nb::class_<LoadedPipeline>(module, "LoadedPipeline")
        .def("invocation_builder", &LoadedPipeline::invocationBuilder, nb::keep_alive<0, 1>())
        .def(
            "submit",
            [](LoadedPipeline &pipeline, uint32_t x, uint32_t y, uint32_t z, const nb::list &values) {
                return pipeline.submitCompute(x, y, z, values);
            },
            nb::arg("x"), nb::arg("y"), nb::arg("z"), nb::arg("values"))
        .def(
            "submit",
            [](LoadedPipeline &pipeline, PipelineInvocationBuilder &builder) {
                if (builder.pipeline != pipeline.pipeline)
                    throw std::invalid_argument("invocation builder belongs to another pipeline");
                return builder.submit();
            },
            nb::arg("builder"))
        .def(
            "vjp",
            [](LoadedPipeline &pipeline, const nb::dict &bindings, const nb::tuple &grid) {
                if (grid.size() != 3)
                    throw std::invalid_argument("autodiff grid must contain three dimensions");
                const auto dimension = [&](size_t index) {
                    if (PyBool_Check(grid[index].ptr()))
                        throw std::invalid_argument("autodiff grid dimensions must be positive integers");
                    const uint64_t value = nb::cast<uint64_t>(grid[index]);
                    if (!value || value > UINT32_MAX)
                        throw std::invalid_argument("autodiff grid dimensions must be positive uint32 values");
                    return static_cast<uint32_t>(value);
                };
                const uint32_t x = dimension(0);
                const uint32_t y = dimension(1);
                const uint32_t z = dimension(2);
                return pipeline.vjp(x, y, z, bindings, nb::cast(&pipeline, nb::rv_policy::reference));
            },
            nb::arg("bindings"), nb::arg("grid"))
        .def(
            "vjp_encode",
            [](LoadedPipeline &pipeline, PipelineInvocationBuilder &builder, vernon::execution::ComputeEncoder &encoder,
               const nb::dict &bindings, const nb::tuple &grid) {
                if (grid.size() != 3)
                    throw std::invalid_argument("autodiff grid must contain three dimensions");
                const auto dimension = [&](size_t index) {
                    if (PyBool_Check(grid[index].ptr()))
                        throw std::invalid_argument("autodiff grid dimensions must be positive integers");
                    const uint64_t value = nb::cast<uint64_t>(grid[index]);
                    if (!value || value > UINT32_MAX)
                        throw std::invalid_argument("autodiff grid dimensions must be positive uint32 values");
                    return static_cast<uint32_t>(value);
                };
                const VernonRhiCommandEncoder native = encoder.native();
                return pipeline.vjp(dimension(0), dimension(1), dimension(2), bindings,
                                    nb::cast(&pipeline, nb::rv_policy::reference), &builder, &native);
            },
            nb::arg("builder"), nb::arg("encoder"), nb::arg("bindings"), nb::arg("grid"))
        .def_prop_ro("derivative_groups", &LoadedPipeline::derivativeGroups)
        .def_prop_ro("workgroup_size", &LoadedPipeline::workgroupSize)
        .def_prop_ro("read_footprints", &LoadedPipeline::readFootprints)
        .def_prop_ro("write_footprints", &LoadedPipeline::writeFootprints)
        .def_prop_ro("parameters", &LoadedPipeline::parameters)
        .def_prop_ro("outputs", &LoadedPipeline::outputs);
    module.attr("DATA_BOOL") = static_cast<uint32_t>(VERNON_DATA_BOOL);
    module.attr("DATA_I32") = static_cast<uint32_t>(VERNON_DATA_I32);
    module.attr("DATA_U32") = static_cast<uint32_t>(VERNON_DATA_U32);
    module.attr("DATA_F16") = static_cast<uint32_t>(VERNON_DATA_F16);
    module.attr("DATA_F32") = static_cast<uint32_t>(VERNON_DATA_F32);
    module.attr("DATA_F64") = static_cast<uint32_t>(VERNON_DATA_F64);
    module.attr("ACCESS_READ") = static_cast<uint32_t>(VERNON_ACCESS_READ);
    module.attr("ACCESS_WRITE") = static_cast<uint32_t>(VERNON_ACCESS_WRITE);
    module.attr("ACCESS_READ_WRITE") = static_cast<uint32_t>(VERNON_ACCESS_READ_WRITE);
    module.attr("PIPELINE_TENSOR") = static_cast<uint32_t>(VERNON_PIPELINE_TENSOR);
    module.attr("PIPELINE_IMAGE") = static_cast<uint32_t>(VERNON_PIPELINE_IMAGE);
    module.attr("PIPELINE_SAMPLER") = static_cast<uint32_t>(VERNON_PIPELINE_SAMPLER);
    module.attr("TOPOLOGY_TRIANGLE_LIST") = static_cast<uint32_t>(VERNON_TOPOLOGY_TRIANGLE_LIST);
    module.attr("ATTACHMENT_CLEAR") = static_cast<uint32_t>(VERNON_RHI_LOAD_CLEAR);
    module.attr("ATTACHMENT_PRESERVE") = static_cast<uint32_t>(VERNON_RHI_LOAD_PRESERVE);
    module.attr("ATTACHMENT_DISCARD") = static_cast<uint32_t>(VERNON_RHI_LOAD_DISCARD);
    module.attr("ATTACHMENT_STORE") = static_cast<uint32_t>(VERNON_RHI_STORE_PRESERVE);
    module.attr("ATTACHMENT_DONT_CARE") = static_cast<uint32_t>(VERNON_RHI_STORE_DISCARD);
    module.attr("GRAPH_READ") = static_cast<uint32_t>(vernon::execution::AccessMode::Read);
    module.attr("GRAPH_WRITE") = static_cast<uint32_t>(vernon::execution::AccessMode::Write);
    module.attr("GRAPH_READ_WRITE") = static_cast<uint32_t>(vernon::execution::AccessMode::ReadWrite);
    module.attr("GRAPH_SHADER_READ") = static_cast<uint32_t>(VERNON_RHI_STATE_SHADER_READ);
    module.attr("GRAPH_SHADER_WRITE") = static_cast<uint32_t>(VERNON_RHI_STATE_SHADER_WRITE);
    module.attr("GRAPH_STAGE_COMPUTE") = static_cast<uint32_t>(VERNON_RHI_STAGE_COMPUTE);
    module.attr("GRAPH_STAGE_VERTEX") = static_cast<uint32_t>(VERNON_RHI_STAGE_VERTEX);
    module.attr("GRAPH_STAGE_FRAGMENT") = static_cast<uint32_t>(VERNON_RHI_STAGE_FRAGMENT);
    module.attr("GRAPH_PASS_NEVER_CULL") = static_cast<uint32_t>(vernon::execution::PassNeverCull);
    module.attr("GRAPH_PASS_NO_MERGE") = static_cast<uint32_t>(vernon::execution::PassNoMerge);
    module.attr("GRAPH_PASS_SIDE_EFFECT") = static_cast<uint32_t>(vernon::execution::PassSideEffect);
    module.attr("TOPOLOGY_LINE_LIST") = static_cast<uint32_t>(VERNON_TOPOLOGY_LINE_LIST);
    module.attr("TOPOLOGY_POINT_LIST") = static_cast<uint32_t>(VERNON_TOPOLOGY_POINT_LIST);
    module.def("runtime_available",
               [](VernonRuntimeBackend backend) { return vernonRuntimeGetCapabilities(backend).available != 0; });
}

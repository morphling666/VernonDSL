#ifndef VERNON_PYTHON_NATIVE_PIPELINE_AUTODIFF_H
#define VERNON_PYTHON_NATIVE_PIPELINE_AUTODIFF_H

#include "native_pipeline.h"
#include "runtime/autodiff/runtime_direct_autodiff.h"

#include <deque>
#include <unordered_map>

struct PythonAdViewDescriptor {
    uintptr_t allocationBegin{};
    size_t allocationSize{};
    size_t byteOffset{};
    VernonDataType dtype{};
    bool writable{};
    std::vector<uint64_t> shape;
    std::vector<int64_t> strides;

    VernonTensorView tensorView() const {
        VernonTensorView tensor{};
        tensor.struct_size = sizeof(VernonTensorView);
        tensor.storage = VERNON_TENSOR_HOST;
        tensor.host_data = reinterpret_cast<const void *>(allocationBegin);
        tensor.element_layout = vernonRuntimeGetScalarValueLayout(dtype);
        tensor.access = writable ? VERNON_ACCESS_READ_WRITE : VERNON_ACCESS_READ;
        tensor.rank = static_cast<uint32_t>(shape.size());
        tensor.shape = shape.data();
        tensor.byte_strides = strides.data();
        tensor.byte_offset = byteOffset;
        tensor.byte_size = allocationSize;
        return tensor;
    }
};

PythonAdViewDescriptor validatePythonAdOriginalView(const std::string &path, VernonDataType dtype,
                                                    const std::vector<uint64_t> &expectedShape, const nb::object &array,
                                                    bool writable);

bool pythonAdViewsOverlap(const PythonAdViewDescriptor &left, const PythonAdViewDescriptor &right);

struct PythonAdValue {
    std::string path;
    nb::object source;
    nb::object array;
    std::vector<uint64_t> shape;
    bool writable{};
    PythonAdViewDescriptor originalView;
    VernonAdValue value{};

    PythonAdValue(std::string path, VernonDataType dtype, std::vector<uint64_t> shape, const nb::object &source,
                  uint32_t access = VERNON_ACCESS_READ)
        : path(std::move(path)), source(nb::module_::import_("numpy").attr("asarray")(source)), array(this->source),
          shape(std::move(shape)), writable(access != VERNON_ACCESS_READ) {
        originalView = validatePythonAdOriginalView(this->path, dtype, this->shape, this->source, writable);
        if (!nb::cast<bool>(array.attr("flags").attr("c_contiguous"))) {
            nb::object numpy = nb::module_::import_("numpy");
            array =
                access == VERNON_ACCESS_WRITE
                    ? numpy.attr("empty")(this->shape, nb::arg("dtype") = numpy.attr("dtype")(numpyDtypeName(dtype)))
                    : numpy.attr("ascontiguousarray")(array);
        }
        size_t scalarCount = 1;
        for (uint64_t extent : this->shape) {
            if (scalarCount && extent > std::numeric_limits<size_t>::max() / scalarCount)
                throw std::invalid_argument("Python autodiff Value shape overflows");
            scalarCount *= static_cast<size_t>(extent);
        }
        const size_t scalarSize = autodiffDtypeSize(dtype);
        if (!scalarSize || (scalarCount && scalarSize > std::numeric_limits<size_t>::max() / scalarCount))
            throw std::invalid_argument("Python autodiff Value byte size overflows");
        if (this->shape.size() > std::numeric_limits<uint32_t>::max())
            throw std::invalid_argument("Python autodiff Value rank overflows");
        value.struct_size = sizeof(value);
        value.path = {this->path.data(), this->path.size()};
        value.dtype = dtype;
        value.data = reinterpret_cast<void *>(nb::cast<uintptr_t>(array.attr("ctypes").attr("data")));
        value.size = scalarCount * scalarSize;
        value.rank = static_cast<uint32_t>(this->shape.size());
        value.shape = this->shape.empty() ? nullptr : this->shape.data();
    }

    PythonAdValue(PythonAdValue &&other) noexcept
        : path(std::move(other.path)), source(std::move(other.source)), array(std::move(other.array)),
          shape(std::move(other.shape)), writable(other.writable), originalView(std::move(other.originalView)),
          value(other.value) {
        value.path = {path.data(), path.size()};
        value.shape = shape.empty() ? nullptr : shape.data();
    }
    void commit() {
        if (writable && source.ptr() != array.ptr())
            nb::module_::import_("numpy").attr("copyto")(source, array);
    }
    PythonAdValue(const PythonAdValue &) = delete;
    PythonAdValue &operator=(const PythonAdValue &) = delete;
};

struct PythonAdMetadata {
    std::string path;
    std::string binding;
    VernonDataType dtype{};
    std::vector<uint64_t> shape;
};

PythonAdMetadata adInputLeafMetadata(VernonLoadedPipeline *pipeline, const PipelineParameterMetadata &parameter,
                                     size_t leafIndex, VernonPipelineValueLeafView *reflected = nullptr);

nb::object resolveAdInputLeaf(const nb::dict &bindings, const PipelineParameterMetadata &parameter,
                              const VernonPipelineValueLeafView &leaf);

struct PythonPullback {
    PythonPullback(VernonRuntimeContext *runtime, VernonPullback *handle, std::vector<PythonAdMetadata> gradients,
                   std::vector<PythonAdMetadata> cotangents, bool hasCarrierDimensions, nb::object pipelineOwner,
                   nb::dict bindings)
        : runtime(runtime), handle(handle), gradients(std::move(gradients)), cotangents(std::move(cotangents)),
          hasCarrierDimensions(hasCarrierDimensions), pipelineOwner(std::move(pipelineOwner)),
          bindings(std::move(bindings)) {}
    ~PythonPullback() { vernonPullbackDestroy(handle); }

    nb::dict apply(const nb::object &cotangent) { return applyImpl(cotangent, false); }
    nb::dict applyLogical(const nb::object &cotangent) { return applyImpl(cotangent, true); }
    nb::dict applyGrouped(const nb::object &cotangent, const nb::object &gradientGroups,
                          const nb::object &cotangentGroups, const nb::object &carrierShape, bool logical);
    size_t logicalResidualBytes() const { return memoryUsage().logicalResidualBytes; }
    size_t residentBytes() const { return memoryUsage().residentBytes; }
    size_t allocatedBytes() const { return memoryUsage().allocatedBytes; }
    size_t tapeContextLimitBytes() const { return vernon::runtime::autodiffHostTapeContextLimit(runtime); }

private:
    vernon::runtime::AutodiffPullbackMemoryUsage memoryUsage() const {
        return vernon::runtime::autodiffPullbackMemoryUsage(handle);
    }

    nb::dict applyImpl(const nb::object &cotangent, bool logicalCotangent) {
        std::deque<PythonAdValue> gradientValues;
        std::vector<VernonAdValue> gradientViews;
        nb::dict result;
        for (const PythonAdMetadata &gradient : gradients) {
            nb::object zeros = nb::module_::import_("numpy").attr("zeros")(
                gradient.shape, nb::module_::import_("numpy").attr(numpyDtypeName(gradient.dtype)));
            gradientValues.emplace_back(gradient.path, gradient.dtype, gradient.shape, zeros);
            gradientViews.push_back(gradientValues.back().value);
        }
        VernonAdValueSet gradientSet{sizeof(VernonAdValueSet), gradientViews.data(), gradientViews.size(), {}};

        std::deque<PythonAdValue> seeds;
        std::vector<VernonAdValue> seedViews;
        VernonAdValueSet seedSet{};
        const VernonAdValueSet *seedView = nullptr;
        if (!cotangent.is_none()) {
            const auto seedMetadata = [&](const PythonAdMetadata &metadata) {
                PythonAdMetadata logical = metadata;
                if (logicalCotangent && hasCarrierDimensions) {
                    if (logical.shape.size() < 3)
                        throw std::runtime_error("autodiff cotangent has no invocation carrier dimensions");
                    logical.shape.erase(logical.shape.begin(), logical.shape.begin() + 3);
                }
                return logical;
            };
            if (cotangents.size() == 1) {
                const PythonAdMetadata metadata = seedMetadata(cotangents.front());
                seeds.emplace_back(metadata.path, metadata.dtype, metadata.shape, cotangent);
            } else {
                if (!nb::isinstance<nb::dict>(cotangent))
                    throw std::invalid_argument("aggregate pullback cotangent must be a leaf-path dictionary");
                nb::dict values = nb::cast<nb::dict>(cotangent);
                if (values.size() != cotangents.size())
                    throw std::invalid_argument("aggregate pullback requires every output cotangent leaf");
                for (const PythonAdMetadata &carriedMetadata : cotangents) {
                    const PythonAdMetadata metadata = seedMetadata(carriedMetadata);
                    nb::str path(metadata.path.c_str());
                    if (!values.contains(path))
                        throw std::invalid_argument("missing output cotangent leaf '" + metadata.path + "'");
                    seeds.emplace_back(metadata.path, metadata.dtype, metadata.shape,
                                       nb::borrow<nb::object>(values[path]));
                }
            }
            for (PythonAdValue &seed : seeds)
                seedViews.push_back(seed.value);
            seedSet = {sizeof(VernonAdValueSet), seedViews.data(), seedViews.size(), {}};
            seedView = &seedSet;
        }
        if (vernonPullbackApply(handle, seedView, &gradientSet) != VERNON_STATUS_OK)
            throw std::runtime_error("pullback application failed: " +
                                     nativeStringView(vernonRuntimeGetLastError(runtime)));
        std::unordered_map<PyObject *, nb::object> gradientsByOwner;
        for (size_t index = 0; index < gradientValues.size(); ++index) {
            PythonAdValue &gradient = gradientValues[index];
            const PythonAdMetadata &metadata = gradients[index];
            nb::str path(gradient.path.c_str());
            nb::str bindingPath(metadata.binding.c_str());
            nb::object binding =
                bindings.contains(bindingPath) ? nb::borrow<nb::object>(bindings[bindingPath]) : nb::none();
            if (binding.is_none() || !nb::hasattr(binding, "_materialize_gradient")) {
                result[path] = gradient.array;
                continue;
            }
            nb::object owner = nb::hasattr(binding, "owner") ? binding.attr("owner") : binding;
            auto existing = gradientsByOwner.find(owner.ptr());
            nb::object materialized =
                existing == gradientsByOwner.end()
                    ? binding.attr("_materialize_gradient")(gradient.array, path)
                    : binding.attr("_materialize_gradient")(gradient.array, path, existing->second);
            if (existing == gradientsByOwner.end())
                gradientsByOwner.emplace(owner.ptr(), materialized);
            result[path] = std::move(materialized);
        }
        return result;
    }

    VernonRuntimeContext *runtime{};
    VernonPullback *handle{};
    std::vector<PythonAdMetadata> gradients;
    std::vector<PythonAdMetadata> cotangents;
    bool hasCarrierDimensions{};
    nb::object pipelineOwner;
    nb::dict bindings;
};

struct LoadedPipeline {
    LoadedPipeline(Runtime *owner, VernonRuntimeContext *runtime, VernonPipelineBundle *bundle,
                   VernonLoadedPipeline *pipeline, std::vector<SharedCompileResult> retainedResults = {})
        : owner(owner), runtime(runtime), bundle(bundle), pipeline(pipeline),
          retainedResults(std::move(retainedResults)) {}
    ~LoadedPipeline() {
        vernonRuntimeLoadedPipelineDestroy(pipeline);
        vernonRuntimePipelineBundleDestroy(bundle);
    }

    std::unique_ptr<PipelineInvocationBuilder> invocationBuilder() {
        return std::make_unique<PipelineInvocationBuilder>(owner, runtime, pipeline);
    }

    std::array<uint32_t, 3> workgroupSize() const {
        const VernonLaunchSize size = vernon::runtime::autodiffWorkgroupSize(pipeline);
        return {size.x, size.y, size.z};
    }

    nb::tuple vjp(uint32_t gridX, uint32_t gridY, uint32_t gridZ, const nb::dict &bindings, nb::object pipelineOwner) {
        if (!gridX || !gridY || !gridZ)
            throw std::invalid_argument("autodiff grid dimensions must be positive");
        const std::vector<PipelineParameterMetadata> metadata = parameters();
        if (bindings.size() != metadata.size())
            throw std::invalid_argument("autodiff bindings do not match pipeline parameters");
        std::deque<PythonAdValue> inputValues;
        std::vector<VernonAdValue> inputViews;
        std::vector<PythonAdMetadata> inputLeafMetadata;
        for (const PipelineParameterMetadata &parameter : metadata) {
            for (size_t leafIndex = 0; leafIndex < parameter.elementLeaves.size(); ++leafIndex) {
                VernonPipelineValueLeafView reflected{};
                PythonAdMetadata leaf = adInputLeafMetadata(pipeline, parameter, leafIndex, &reflected);
                nb::object value = resolveAdInputLeaf(bindings, parameter, reflected);
                const std::vector<uint64_t> actualShape = nb::cast<std::vector<uint64_t>>(value.attr("shape"));
                if (actualShape.size() == leaf.shape.size())
                    for (size_t dimension = 0; dimension < leaf.shape.size(); ++dimension)
                        if (!leaf.shape[dimension])
                            leaf.shape[dimension] = actualShape[dimension];
                inputValues.emplace_back(leaf.path, leaf.dtype, leaf.shape, value, parameter.access);
                inputViews.push_back(inputValues.back().value);
                inputLeafMetadata.push_back(std::move(leaf));
            }
        }
        for (size_t left = 0; left < inputValues.size(); ++left)
            for (size_t right = left + 1; right < inputValues.size(); ++right) {
                const PythonAdValue &leftValue = inputValues[left];
                const PythonAdValue &rightValue = inputValues[right];
                if (pythonAdViewsOverlap(leftValue.originalView, rightValue.originalView))
                    throw std::invalid_argument("Python autodiff Values '" + leftValue.path + "' and '" +
                                                rightValue.path + "' have incompatible overlapping views");
            }
        VernonAdValueSet inputSet{sizeof(VernonAdValueSet), inputViews.data(), inputViews.size(), {}};

        const VernonLaunchSize workgroup = vernon::runtime::autodiffWorkgroupSize(pipeline);
        if (gridX > UINT32_MAX / workgroup.x || gridY > UINT32_MAX / workgroup.y || gridZ > UINT32_MAX / workgroup.z)
            throw std::invalid_argument("autodiff invocation extent overflows");
        const uint32_t extentX = gridX * workgroup.x;
        const uint32_t extentY = gridY * workgroup.y;
        const uint32_t extentZ = gridZ * workgroup.z;
        if (extentX > SIZE_MAX / extentY || static_cast<size_t>(extentX) * extentY > SIZE_MAX / extentZ)
            throw std::invalid_argument("autodiff grid size overflows");
        const size_t carrierCount = static_cast<size_t>(extentX) * extentY * extentZ;
        const size_t outputCount = vernonRuntimeLoadedPipelineGetAdOutputCount(pipeline);
        const size_t cotangentCount = vernonRuntimeLoadedPipelineGetAdCotangentCount(pipeline);
        if (!outputCount || !cotangentCount)
            throw std::runtime_error("pipeline has no consistent autodiff output/cotangent signature");
        auto prependCarrierDimensions = [&](std::vector<uint64_t> &shape) {
            if (carrierCount > 1)
                shape.insert(shape.begin(), {extentZ, extentY, extentX});
        };
        const bool storageObjectives = vernon::runtime::hasAutodiffStorageObjectives(pipeline);
        auto materializeMetadata = [&](const VernonAdValueMetadataView &value) {
            const std::string path = nativeStringView(value.path);
            PythonAdMetadata result{path, path, value.dtype, {}};
            if (value.rank)
                result.shape.assign(value.shape, value.shape + value.rank);
            prependCarrierDimensions(result.shape);
            return result;
        };
        std::vector<PythonAdMetadata> outputMetadata;
        std::vector<PythonAdMetadata> cotangentMetadata;
        outputMetadata.reserve(outputCount);
        cotangentMetadata.reserve(cotangentCount);
        std::deque<PythonAdValue> outputValues;
        std::vector<VernonAdValue> outputViews;
        nb::dict aggregateOutput;
        for (size_t index = 0; index < outputCount; ++index) {
            VernonAdValueMetadataView outputValue{};
            outputValue.struct_size = sizeof(outputValue);
            if (vernonRuntimeLoadedPipelineGetAdOutputByIndex(pipeline, index, &outputValue) != VERNON_STATUS_OK)
                throw std::runtime_error("pipeline has no consistent autodiff output/cotangent signature");
            outputMetadata.push_back(materializeMetadata(outputValue));
            const PythonAdMetadata &metadata = outputMetadata.back();
            if (storageObjectives)
                continue;
            nb::object zeros = nb::module_::import_("numpy").attr("zeros")(
                metadata.shape, nb::module_::import_("numpy").attr(numpyDtypeName(metadata.dtype)));
            outputValues.emplace_back(metadata.path, metadata.dtype, metadata.shape, zeros);
            outputViews.push_back(outputValues.back().value);
            aggregateOutput[nb::str(metadata.path.c_str())] = outputValues.back().array;
        }
        for (size_t index = 0; index < cotangentCount; ++index) {
            VernonAdValueMetadataView cotangentValue{};
            cotangentValue.struct_size = sizeof(cotangentValue);
            if (vernonRuntimeLoadedPipelineGetAdCotangentByIndex(pipeline, index, &cotangentValue) != VERNON_STATUS_OK)
                throw std::runtime_error("pipeline has no consistent autodiff output/cotangent signature");
            PythonAdMetadata cotangent = materializeMetadata(cotangentValue);
            const auto input =
                std::find_if(inputLeafMetadata.begin(), inputLeafMetadata.end(),
                             [&](const PythonAdMetadata &candidate) { return candidate.path == cotangent.path; });
            if (input != inputLeafMetadata.end()) {
                cotangent.binding = input->binding;
                cotangent.shape = input->shape;
                prependCarrierDimensions(cotangent.shape);
            }
            cotangentMetadata.push_back(std::move(cotangent));
        }
        if (storageObjectives) {
            outputValues.clear();
            outputViews.clear();
            aggregateOutput.clear();
        }
        VernonAdValueSet outputSet{sizeof(VernonAdValueSet), outputViews.data(), outputViews.size(), {}};
        VernonPullback *pullback = nullptr;
        if (vernonAdPipelineForward(pipeline, {gridX, gridY, gridZ}, &inputSet, &outputSet, &pullback) !=
            VERNON_STATUS_OK)
            throw std::runtime_error("autodiff forward invocation failed: " +
                                     nativeStringView(vernonRuntimeGetLastError(runtime)));
        for (PythonAdValue &input : inputValues)
            input.commit();

        std::vector<PythonAdMetadata> gradients;
        const size_t gradientCount = vernonRuntimeLoadedPipelineGetAdGradientCount(pipeline);
        gradients.reserve(gradientCount);
        for (size_t index = 0; index < gradientCount; ++index) {
            VernonAdValueMetadataView gradient{};
            gradient.struct_size = sizeof(gradient);
            if (vernonRuntimeLoadedPipelineGetAdGradientByIndex(pipeline, index, &gradient) != VERNON_STATUS_OK) {
                vernonPullbackDestroy(pullback);
                throw std::runtime_error("cannot read autodiff gradient reflection");
            }
            const std::string name = nativeStringView(gradient.path);
            const auto reflectedInput =
                std::find_if(inputLeafMetadata.begin(), inputLeafMetadata.end(),
                             [&](const PythonAdMetadata &candidate) { return candidate.path == name; });
            if (reflectedInput == inputLeafMetadata.end()) {
                vernonPullbackDestroy(pullback);
                throw std::runtime_error("autodiff gradient path does not match an input Value leaf");
            }
            PythonAdMetadata inputLeaf = *reflectedInput;
            if (autodiffTangentDtype(inputLeaf.dtype) != gradient.dtype) {
                vernonPullbackDestroy(pullback);
                throw std::runtime_error("autodiff gradient dtype does not match its input Value leaf tangent type");
            }
            inputLeaf.dtype = gradient.dtype;
            gradients.push_back(std::move(inputLeaf));
        }
        nb::object output = storageObjectives          ? nb::none()
                            : outputValues.size() == 1 ? outputValues.front().array
                                                       : nb::borrow<nb::object>(aggregateOutput);
        return nb::make_tuple(output, std::make_unique<PythonPullback>(runtime, pullback, std::move(gradients),
                                                                       std::move(cotangentMetadata), carrierCount > 1,
                                                                       std::move(pipelineOwner), nb::dict(bindings)));
    }

    static size_t dataTypeSize(VernonDataType type) {
        switch (type) {
        case VERNON_DATA_BOOL:
        case VERNON_DATA_U8:
            return 1;
        case VERNON_DATA_F16:
            return 2;
        case VERNON_DATA_I32:
        case VERNON_DATA_U32:
        case VERNON_DATA_F32:
            return 4;
        case VERNON_DATA_F64:
            return 8;
        }
        throw std::invalid_argument("unsupported compute pipeline data type");
    }

    std::unique_ptr<PythonRuntimeSubmission> submitCompute(uint32_t x, uint32_t y, uint32_t z, const nb::list &values) {
        const std::vector<PipelineParameterMetadata> metadata = parameters();
        if (values.size() != metadata.size())
            throw std::invalid_argument("compute pipeline value count does not match reflection");
        std::deque<std::string> scalarStorage;
        std::deque<std::vector<uint64_t>> shapes;
        std::deque<std::vector<int64_t>> strides;
        std::vector<VernonPipelineArgument> arguments;
        arguments.reserve(metadata.size());
        for (size_t index = 0; index < metadata.size(); ++index) {
            const PipelineParameterMetadata &parameter = metadata[index];
            if (parameter.kind != VERNON_PIPELINE_TENSOR)
                throw std::invalid_argument("direct compute pipelines accept only Tensor/value parameters");
            shapes.push_back(parameter.shape);
            strides.emplace_back(parameter.shape.size());
            size_t stride = parameter.elementByteSize;
            for (size_t dimension = parameter.shape.size(); dimension-- != 0;) {
                strides.back()[dimension] = static_cast<int64_t>(stride);
                stride *= static_cast<size_t>(parameter.shape[dimension]);
            }
            VernonPipelineArgument argument{};
            argument.slot = parameter.slot;
            argument.kind = VERNON_PIPELINE_TENSOR;
            argument.tensor.struct_size = sizeof(VernonTensorView);
            argument.tensor.element_layout = {
                sizeof(VernonValueLayoutView),  parameter.elementByteSize,
                parameter.elementAlignment,     {parameter.layoutHash.data(), parameter.layoutHash.size()},
                parameter.elementLeaves.data(), parameter.elementLeaves.size(),
            };
            argument.tensor.access = parameter.access;
            argument.tensor.rank = static_cast<uint32_t>(shapes.back().size());
            argument.tensor.shape = shapes.back().empty() ? nullptr : shapes.back().data();
            argument.tensor.byte_strides = strides.back().empty() ? nullptr : strides.back().data();
            nb::handle value = values[index];
            if (nb::isinstance<RhiBuffer>(value)) {
                RhiBuffer *buffer = nb::cast<RhiBuffer *>(value);
                if (!buffer || runtimeRhiHost(owner) != buffer->host.get())
                    throw std::invalid_argument("compute pipeline RHI buffer belongs to another device");
                argument.tensor.storage = VERNON_TENSOR_RHI_RESOURCE;
                if (vernonRuntimeReferenceRhiBuffer(runtime, buffer->handle, 0, buffer->size,
                                                    &argument.tensor.resource) != VERNON_STATUS_OK)
                    throw std::invalid_argument("compute pipeline RHI buffer is invalid");
                argument.tensor.byte_size = buffer->size;
            } else if (nb::isinstance<nb::bytes>(value)) {
                nb::bytes bytes = nb::borrow<nb::bytes>(value);
                scalarStorage.emplace_back(bytes.c_str(), bytes.size());
                argument.tensor.storage = VERNON_TENSOR_HOST;
                argument.tensor.host_data = scalarStorage.back().data();
                argument.tensor.byte_size = scalarStorage.back().size();
            } else {
                throw std::invalid_argument("compute pipeline values must be RhiBuffer or bytes");
            }
            arguments.push_back(argument);
        }
        VernonPipelineInvocation invocation{};
        invocation.struct_size = sizeof(invocation);
        invocation.abi_version = VERNON_PIPELINE_VERSION;
        invocation.arguments = arguments.data();
        invocation.argument_count = arguments.size();
        invocation.compute_grid = {x, y, z};
        VernonSubmission *submission{};
        if (vernonRuntimePipelineSubmit(pipeline, &invocation, &submission) != VERNON_STATUS_OK)
            throw std::runtime_error("compute pipeline submission failed: " +
                                     nativeStringView(vernonRuntimeGetLastError(runtime)));
        return std::make_unique<PythonRuntimeSubmission>(submission);
    }

    nb::list derivativeGroups() const {
        nb::list result;
        const size_t groupCount = vernonRuntimeLoadedPipelineGetAdDerivativeGroupCount(pipeline);
        for (size_t groupIndex = 0; groupIndex < groupCount; ++groupIndex) {
            VernonAdDerivativeGroupView group{};
            group.struct_size = sizeof(group);
            if (vernonRuntimeLoadedPipelineGetAdDerivativeGroupByIndex(pipeline, groupIndex, &group) !=
                VERNON_STATUS_OK)
                throw std::runtime_error("cannot read autodiff derivative group metadata");
            nb::list leaves;
            for (size_t leafIndex = 0; leafIndex < group.leaf_count; ++leafIndex) {
                VernonStringView leafPath{};
                if (vernonRuntimeLoadedPipelineGetAdDerivativeGroupLeaf(pipeline, groupIndex, leafIndex, &leafPath) !=
                    VERNON_STATUS_OK)
                    throw std::runtime_error("cannot read autodiff derivative group leaf");
                leaves.append(nativeStringView(leafPath));
            }
            result.append(nb::make_tuple(group.role == VERNON_AD_DERIVATIVE_GRADIENT ? "gradient" : "cotangent",
                                         nativeStringView(group.declared_path), leaves));
        }
        return result;
    }

    std::vector<PipelineParameterMetadata> parameters() const {
        std::vector<PipelineParameterMetadata> result;
        const size_t count = vernonRuntimeLoadedPipelineGetParameterCount(pipeline);
        result.reserve(count);
        for (size_t index = 0; index < count; ++index) {
            VernonPipelineParameterView view{};
            if (vernonRuntimeLoadedPipelineGetParameterByIndex(pipeline, index, &view) != VERNON_STATUS_OK)
                throw std::runtime_error("cannot read loaded pipeline parameter");
            result.push_back(parameterMetadata(view));
        }
        return result;
    }

    std::vector<PipelineOutputMetadata> outputs() const {
        std::vector<PipelineOutputMetadata> result;
        const size_t count = vernonRuntimeLoadedPipelineGetOutputCount(pipeline);
        result.reserve(count);
        for (size_t index = 0; index < count; ++index) {
            VernonPipelineOutputView view{};
            if (vernonRuntimeLoadedPipelineGetOutputByIndex(pipeline, index, &view) != VERNON_STATUS_OK)
                throw std::runtime_error("cannot read loaded pipeline output");
            result.push_back(outputMetadata(view));
        }
        return result;
    }

    Runtime *owner{};
    VernonRuntimeContext *runtime{};
    VernonPipelineBundle *bundle{};
    VernonLoadedPipeline *pipeline{};
    // ORC entry pointers are valid only while their compile result owns the JIT.
    std::vector<SharedCompileResult> retainedResults;
};

#endif

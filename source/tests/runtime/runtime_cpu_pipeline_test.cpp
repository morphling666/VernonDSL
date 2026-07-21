#include "runtime/content_hash.h"
#include "vernon-c/Runtime.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <string>

#ifndef VERNON_CPU_BUNDLE_PATH
#error VERNON_CPU_BUNDLE_PATH must name the CPU bundle test fixture
#endif

#ifndef VERNON_RUNTIME_TEST_OS
#error VERNON_RUNTIME_TEST_OS must name the host operating system
#endif

#ifndef VERNON_RUNTIME_TEST_ARCH
#error VERNON_RUNTIME_TEST_ARCH must name the host architecture
#endif

namespace {

std::string readFile(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>());
}

void replaceOnce(std::string &text, const std::string &from,
                 const std::string &to) {
  const size_t position = text.find(from);
  ASSERT_TRUE(position != std::string::npos);
  text.replace(position, from.size(), to);
}

VernonPipelineBundle *loadWithDirectory(VernonRuntimeContext *runtime,
                                        const std::string &bundle,
                                        const std::string &directory) {
  VernonPipelineBundleLoadOptions options{};
  options.struct_size = sizeof(options);
  options.bundle_directory = directory.c_str();
  return vernonRuntimeLoadPipelineBundleWithOptions(runtime, bundle.data(),
                                                    bundle.size(), &options);
}

std::string schema2Bundle(const std::string &legacy) {
  nlohmann::json root = nlohmann::json::parse(legacy);
  root.erase("pipeline_bundle_schema_version");
  root["schema_version"] = 2;
  root["type"] = "pipeline";
  for (auto &[_, stage] : root["stage_artifacts"].items()) {
    stage["artifact"]["format"] = stage["format"];
    stage["artifact"]["storage"] = "external";
    stage.erase("format");
  }
  root.erase("content_hash");
  const std::string canonical = root.dump(-1, ' ', false);
  root["content_hash"] =
      vernon::runtime::sha256Hex(canonical.data(), canonical.size());
  return root.dump(-1, ' ', false);
}

VernonStatus staticallyLinkedFill(const VernonCpuInvocation *invocation) {
  uintptr_t address = 0;
  uint32_t gid[3] = {0, 0, 0};
  if (!invocation || invocation->arguments_size < 20)
    return VERNON_STATUS_INVALID_ARGUMENT;
  std::memcpy(&address, invocation->arguments, sizeof(address));
  std::memcpy(gid,
              static_cast<const unsigned char *>(invocation->arguments) + 8,
              sizeof(gid));
  float *values = reinterpret_cast<float *>(address);
  values[gid[2] * 6 + gid[1] * 3 + gid[0]] =
      static_cast<float>(gid[0] + 10 * gid[1] + 100 * gid[2]);
  return VERNON_STATUS_OK;
}

} // namespace

TEST(RuntimeCpuPipeline, LoadsValidatesAndInvokesBundles) {
  const std::filesystem::path directory = VERNON_CPU_BUNDLE_PATH;
  const std::string directoryUtf8 = directory.u8string();
  const std::string bundle = readFile(directory / "pipeline.bundle");
  ASSERT_TRUE(!bundle.empty());

  VernonRuntimeBackend target = VERNON_RUNTIME_CUDA;
  ASSERT_TRUE(vernonRuntimePipelineBundleInspectTarget(
                  bundle.data(), bundle.size(), &target) == VERNON_STATUS_OK);
  ASSERT_TRUE(target == VERNON_RUNTIME_CPU);

  VernonRuntimeContext *runtime = vernonRuntimeCreate(VERNON_RUNTIME_CPU, 0);
  ASSERT_TRUE(runtime);

  ASSERT_TRUE(
      !vernonRuntimeLoadPipelineBundle(runtime, bundle.data(), bundle.size()));

  const std::string schema2 = schema2Bundle(bundle);
  target = VERNON_RUNTIME_CUDA;
  ASSERT_TRUE(vernonRuntimePipelineBundleInspectTarget(
                  schema2.data(), schema2.size(), &target) == VERNON_STATUS_OK);
  ASSERT_TRUE(target == VERNON_RUNTIME_CPU);
  VernonPipelineBundle *schema2Loaded =
      loadWithDirectory(runtime, schema2, directoryUtf8);
  ASSERT_TRUE(schema2Loaded);
  vernonRuntimePipelineBundleDestroy(schema2Loaded);

  const std::string objectBytes = "test relocatable object";
  const std::filesystem::path objectPath = directory / "test_static.o";
  {
    std::ofstream objectOutput(objectPath, std::ios::binary);
    objectOutput << objectBytes;
  }
  nlohmann::json objectBundle = nlohmann::json::parse(schema2);
  nlohmann::json &objectStage = objectBundle["stage_artifacts"]["fill"];
  objectStage["artifact"]["format"] = "relocatable_object";
  objectStage["artifact"]["path"] = objectPath.filename().string();
  objectStage["artifact"]["size"] = objectBytes.size();
  objectStage["artifact"]["sha256"] =
      vernon::runtime::sha256Hex(objectBytes.data(), objectBytes.size());
  objectStage["format"] = "relocatable_object";
  objectStage["target_triple"] = "test-host-triple";
  objectStage["object_format"] = "elf";
  objectBundle.erase("content_hash");
  std::string objectCanonical = objectBundle.dump(-1, ' ', false);
  objectBundle["content_hash"] = vernon::runtime::sha256Hex(
      objectCanonical.data(), objectCanonical.size());
  const std::string objectManifest = objectBundle.dump(-1, ' ', false);
  ASSERT_TRUE(vernonRuntimeRegisterStaticCpuEntry(
                  {"vernon_test_fill", std::strlen("vernon_test_fill")},
                  staticallyLinkedFill) == VERNON_STATUS_OK);
  VernonPipelineBundle *objectLoaded =
      loadWithDirectory(runtime, objectManifest, directoryUtf8);
  ASSERT_TRUE(objectLoaded);
  VernonLoadedPipeline *objectPipeline =
      vernonRuntimeResolvePipeline(objectLoaded, {nullptr, 0});
  ASSERT_TRUE(objectPipeline);
  vernonRuntimeLoadedPipelineDestroy(objectPipeline);
  vernonRuntimePipelineBundleDestroy(objectLoaded);

  nlohmann::json invalidSchema2 = nlohmann::json::parse(schema2);
  invalidSchema2["stage_artifacts"]["fill"]["cpu_invocation_abi_version"] = 2;
  invalidSchema2.erase("content_hash");
  std::string canonical = invalidSchema2.dump(-1, ' ', false);
  invalidSchema2["content_hash"] =
      vernon::runtime::sha256Hex(canonical.data(), canonical.size());
  canonical = invalidSchema2.dump(-1, ' ', false);
  ASSERT_TRUE(!loadWithDirectory(runtime, canonical, directoryUtf8));

  VernonPipelineBundleLoadOptions shortOptions{};
  shortOptions.struct_size = sizeof(shortOptions) - 1;
  shortOptions.bundle_directory = directoryUtf8.c_str();
  ASSERT_TRUE(!vernonRuntimeLoadPipelineBundleWithOptions(
      runtime, bundle.data(), bundle.size(), &shortOptions));

  std::string invalid = bundle;
  replaceOnce(invalid, R"("steps": [{"kind": "dispatch", "stage": "fill"}])",
              R"("steps": [{"kind": "dispatch", "stage": "fill"},)"
              R"({"kind": "barrier"}])");
  ASSERT_TRUE(!loadWithDirectory(runtime, invalid, directoryUtf8));

  invalid = bundle;
  replaceOnce(invalid, R"("steps": [{"kind": "dispatch", "stage": "fill"}])",
              R"("steps": [{"kind": "draw", "vertex": "fill", )"
              R"("fragment": "fill"}])");
  ASSERT_TRUE(!loadWithDirectory(runtime, invalid, directoryUtf8));

  invalid = bundle;
  const std::string hashMarker = R"("sha256": ")";
  size_t position = invalid.find(hashMarker);
  ASSERT_TRUE(position != std::string::npos);
  position += hashMarker.size();
  invalid[position] = invalid[position] == '0' ? '1' : '0';
  ASSERT_TRUE(!loadWithDirectory(runtime, invalid, directoryUtf8));

  invalid = bundle;
  const std::string libraryMarker = R"("path": ")";
  position = invalid.find(libraryMarker);
  ASSERT_TRUE(position != std::string::npos);
  position += libraryMarker.size();
  invalid.insert(position, "../");
  ASSERT_TRUE(!loadWithDirectory(runtime, invalid, directoryUtf8));

  invalid = bundle;
  replaceOnce(invalid,
              std::string(R"("operating_system": ")") + VERNON_RUNTIME_TEST_OS +
                  "\"",
              R"("operating_system": "unsupported")");
  ASSERT_TRUE(!loadWithDirectory(runtime, invalid, directoryUtf8));

  invalid = bundle;
  replaceOnce(invalid,
              std::string(R"("architecture": ")") + VERNON_RUNTIME_TEST_ARCH +
                  "\"",
              R"("architecture": "unsupported")");
  ASSERT_TRUE(!loadWithDirectory(runtime, invalid, directoryUtf8));

  VernonPipelineBundle *loaded = vernonRuntimeLoadPipelineBundleFromDirectory(
      runtime, directoryUtf8.c_str());
  ASSERT_TRUE(loaded);
  const VernonStringView id = vernonRuntimePipelineBundleGetId(loaded);
  ASSERT_TRUE(id.size == std::strlen("cpu/fill"));
  ASSERT_TRUE(std::memcmp(id.data, "cpu/fill", id.size) == 0);

  VernonLoadedPipeline *pipeline =
      vernonRuntimeResolvePipeline(loaded, {nullptr, 0});
  ASSERT_TRUE(pipeline);
  ASSERT_TRUE(vernonRuntimeLoadedPipelineGetParameterCount(pipeline) == 1);
  VernonPipelineParameterView parameter{};
  ASSERT_TRUE(vernonRuntimeLoadedPipelineGetParameterByIndex(
                  pipeline, 0, &parameter) == VERNON_STATUS_OK);
  ASSERT_TRUE(parameter.slot == 0 && parameter.kind == VERNON_PIPELINE_TENSOR &&
              parameter.dtype == VERNON_DATA_F32 &&
              parameter.access == VERNON_ACCESS_WRITE && parameter.rank == 1 &&
              parameter.static_shape[0] == 12);
  ASSERT_TRUE(vernonRuntimeLoadedPipelineFindParameter(
                  pipeline, {"output", std::strlen("output")}, &parameter) ==
              VERNON_STATUS_OK);
  ASSERT_TRUE(vernonRuntimeLoadedPipelineGetOutputCount(pipeline) == 1);
  VernonPipelineOutputView outputView{};
  ASSERT_TRUE(vernonRuntimeLoadedPipelineFindOutput(
                  pipeline, {"result", std::strlen("result")}, &outputView) ==
              VERNON_STATUS_OK);
  ASSERT_TRUE(outputView.kind == VERNON_PIPELINE_TENSOR &&
              outputView.dtype == VERNON_DATA_F32 && outputView.rank == 1 &&
              outputView.static_shape[0] == 12 && outputView.location == 0);

  VernonDeviceBuffer *buffer =
      vernonRuntimeBufferAllocate(runtime, 12 * sizeof(float), alignof(float));
  ASSERT_TRUE(buffer);
  const uint64_t shape[] = {12};
  const uint64_t strides[] = {sizeof(float)};
  VernonPipelineArgument argument{};
  argument.slot = 0;
  argument.kind = VERNON_PIPELINE_TENSOR;
  argument.tensor = {
      buffer, VERNON_DATA_F32, VERNON_ACCESS_WRITE, 1, shape, strides, 0};
  VernonPipelineInvocation invocation{};
  invocation.struct_size = sizeof(invocation);
  invocation.abi_version = VERNON_PIPELINE_INVOCATION_ABI_VERSION;
  invocation.arguments = &argument;
  invocation.argument_count = 1;
  invocation.compute_grid = {3, 2, 2};
  ASSERT_TRUE(vernonRuntimePipelineInvoke(pipeline, &invocation) ==
              VERNON_STATUS_OK);

  float output[12]{};
  ASSERT_TRUE(vernonRuntimeCopyToHost(buffer, 0, output, sizeof(output)) ==
              VERNON_STATUS_OK);
  ASSERT_TRUE(output[0] == 0.0f && output[2] == 2.0f);
  ASSERT_TRUE(output[3] == 10.0f && output[11] == 112.0f);

  ASSERT_TRUE(vernonRuntimeBufferFree(buffer) == VERNON_STATUS_OK);
  vernonRuntimeLoadedPipelineDestroy(pipeline);
  vernonRuntimePipelineBundleDestroy(loaded);
  ASSERT_TRUE(vernonRuntimeDestroy(runtime) == VERNON_STATUS_OK);
}

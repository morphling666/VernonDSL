#include "VernonRuntime.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace {

#if defined(VERNON_HAS_CUDA_RUNTIME)
using CudaDevice = int;
using CudaDevicePointer = unsigned long long;
using CudaContext = struct CudaContextOpaque *;
using CudaModule = struct CudaModuleOpaque *;
using CudaFunction = struct CudaFunctionOpaque *;
using CudaStream = struct CudaStreamOpaque *;
using CudaResult = int;

constexpr CudaResult kCudaSuccess = 0;

struct CudaDriver {
  using Init = CudaResult (*)(unsigned);
  using DeviceGet = CudaResult (*)(CudaDevice *, int);
  using PrimaryContextRetain = CudaResult (*)(CudaContext *, CudaDevice);
  using PrimaryContextRelease = CudaResult (*)(CudaDevice);
  using ContextSetCurrent = CudaResult (*)(CudaContext);
  using ContextSynchronize = CudaResult (*)();
  using ErrorName = CudaResult (*)(CudaResult, const char **);
  using ErrorString = CudaResult (*)(CudaResult, const char **);
  using MemoryAllocate = CudaResult (*)(CudaDevicePointer *, size_t);
  using MemoryFree = CudaResult (*)(CudaDevicePointer);
  using CopyHostToDevice = CudaResult (*)(CudaDevicePointer, const void *,
                                          size_t);
  using CopyDeviceToHost = CudaResult (*)(void *, CudaDevicePointer, size_t);
  using ModuleLoadData = CudaResult (*)(CudaModule *, const void *, unsigned,
                                        int *, void **);
  using ModuleGetFunction = CudaResult (*)(CudaFunction *, CudaModule,
                                           const char *);
  using ModuleUnload = CudaResult (*)(CudaModule);
  using LaunchKernel = CudaResult (*)(CudaFunction, unsigned, unsigned,
                                      unsigned, unsigned, unsigned, unsigned,
                                      unsigned, CudaStream, void **, void **);

  bool load() {
    std::lock_guard<std::mutex> guard(mutex);
    if (attempted)
      return available;
    attempted = true;
#if defined(_WIN32)
    constexpr const char *libraryName = "nvcuda.dll";
#else
    constexpr const char *libraryName = "libcuda.so.1";
#endif
    library =
        llvm::sys::DynamicLibrary::getPermanentLibrary(libraryName, &error);
    if (!library.isValid())
      return false;

#define VERNON_LOAD_CUDA(member, symbol)                                       \
  member =                                                                     \
      reinterpret_cast<decltype(member)>(library.getAddressOfSymbol(symbol));  \
  if (!member) {                                                               \
    error = std::string("CUDA Driver is missing symbol ") + symbol;            \
    return false;                                                              \
  }
    VERNON_LOAD_CUDA(init, "cuInit");
    VERNON_LOAD_CUDA(deviceGet, "cuDeviceGet");
    VERNON_LOAD_CUDA(primaryContextRetain, "cuDevicePrimaryCtxRetain");
    VERNON_LOAD_CUDA(primaryContextRelease, "cuDevicePrimaryCtxRelease");
    VERNON_LOAD_CUDA(contextSetCurrent, "cuCtxSetCurrent");
    VERNON_LOAD_CUDA(contextSynchronize, "cuCtxSynchronize");
    VERNON_LOAD_CUDA(errorName, "cuGetErrorName");
    VERNON_LOAD_CUDA(errorString, "cuGetErrorString");
    memoryAllocate = reinterpret_cast<MemoryAllocate>(
        library.getAddressOfSymbol("cuMemAlloc_v2"));
    if (!memoryAllocate)
      memoryAllocate = reinterpret_cast<MemoryAllocate>(
          library.getAddressOfSymbol("cuMemAlloc"));
    if (!memoryAllocate) {
      error = "CUDA Driver is missing symbol cuMemAlloc_v2";
      return false;
    }
    memoryFree = reinterpret_cast<MemoryFree>(
        library.getAddressOfSymbol("cuMemFree_v2"));
    if (!memoryFree)
      memoryFree =
          reinterpret_cast<MemoryFree>(library.getAddressOfSymbol("cuMemFree"));
    if (!memoryFree) {
      error = "CUDA Driver is missing symbol cuMemFree_v2";
      return false;
    }
    copyHostToDevice = reinterpret_cast<CopyHostToDevice>(
        library.getAddressOfSymbol("cuMemcpyHtoD_v2"));
    if (!copyHostToDevice)
      copyHostToDevice = reinterpret_cast<CopyHostToDevice>(
          library.getAddressOfSymbol("cuMemcpyHtoD"));
    if (!copyHostToDevice) {
      error = "CUDA Driver is missing symbol cuMemcpyHtoD_v2";
      return false;
    }
    copyDeviceToHost = reinterpret_cast<CopyDeviceToHost>(
        library.getAddressOfSymbol("cuMemcpyDtoH_v2"));
    if (!copyDeviceToHost)
      copyDeviceToHost = reinterpret_cast<CopyDeviceToHost>(
          library.getAddressOfSymbol("cuMemcpyDtoH"));
    if (!copyDeviceToHost) {
      error = "CUDA Driver is missing symbol cuMemcpyDtoH_v2";
      return false;
    }
    VERNON_LOAD_CUDA(moduleLoadData, "cuModuleLoadDataEx");
    VERNON_LOAD_CUDA(moduleGetFunction, "cuModuleGetFunction");
    VERNON_LOAD_CUDA(moduleUnload, "cuModuleUnload");
    VERNON_LOAD_CUDA(launchKernel, "cuLaunchKernel");
#undef VERNON_LOAD_CUDA
    available = true;
    return true;
  }

  llvm::sys::DynamicLibrary library;
  std::mutex mutex;
  std::string error;
  bool attempted{false};
  bool available{false};
  Init init{};
  DeviceGet deviceGet{};
  PrimaryContextRetain primaryContextRetain{};
  PrimaryContextRelease primaryContextRelease{};
  ContextSetCurrent contextSetCurrent{};
  ContextSynchronize contextSynchronize{};
  ErrorName errorName{};
  ErrorString errorString{};
  MemoryAllocate memoryAllocate{};
  MemoryFree memoryFree{};
  CopyHostToDevice copyHostToDevice{};
  CopyDeviceToHost copyDeviceToHost{};
  ModuleLoadData moduleLoadData{};
  ModuleGetFunction moduleGetFunction{};
  ModuleUnload moduleUnload{};
  LaunchKernel launchKernel{};
};

CudaDriver &cudaDriver() {
  static CudaDriver driver;
  return driver;
}
#endif

struct ReflectedArgument {
  std::string kind;
  std::string builtin;
  size_t cpuOffset{0};
  size_t cpuSize{0};
  size_t tensorBytes{0};
  size_t tensorElements{0};
  size_t tensorElementSize{0};
  size_t alignment{1};
};

struct ReflectedEntry {
  std::vector<ReflectedArgument> arguments;
  size_t cpuArgumentsSize{0};
  uint32_t workgroup[3]{1, 1, 1};
};

bool parseReflection(llvm::StringRef text, llvm::StringRef selected,
                     ReflectedEntry &output, std::string &error) {
  llvm::Expected<llvm::json::Value> parsed = llvm::json::parse(text);
  if (!parsed) {
    error = "invalid reflection JSON";
    return false;
  }
  llvm::json::Object *root = parsed->getAsObject();
  if (!root || root->getInteger("gpu_launch_abi_version").value_or(0) != 1) {
    error = "unsupported GPU launch ABI version";
    return false;
  }
  llvm::json::Array *entries = root->getArray("entries");
  if (!entries) {
    error = "reflection has no entries";
    return false;
  }
  for (llvm::json::Value &value : *entries) {
    llvm::json::Object *entry = value.getAsObject();
    if (!entry || entry->getString("name").value_or("") != selected)
      continue;
    output.cpuArgumentsSize = static_cast<size_t>(
        entry->getInteger("cpu_arguments_size").value_or(0));
    if (llvm::json::Array *workgroup = entry->getArray("workgroup_size")) {
      if (workgroup->size() == 3) {
        for (size_t index = 0; index < 3; ++index)
          output.workgroup[index] = static_cast<uint32_t>(
              (*workgroup)[index].getAsInteger().value_or(1));
      }
    }
    llvm::json::Array *arguments = entry->getArray("arguments");
    if (!arguments) {
      error = "entry has no argument reflection";
      return false;
    }
    for (llvm::json::Value &argumentValue : *arguments) {
      llvm::json::Object *argument = argumentValue.getAsObject();
      if (!argument) {
        error = "invalid argument reflection";
        return false;
      }
      ReflectedArgument reflected;
      reflected.kind = argument->getString("kind").value_or("scalar").str();
      reflected.builtin = argument->getString("builtin").value_or("").str();
      reflected.cpuOffset =
          static_cast<size_t>(argument->getInteger("cpu_offset").value_or(0));
      reflected.cpuSize =
          static_cast<size_t>(argument->getInteger("cpu_size").value_or(0));
      reflected.alignment =
          static_cast<size_t>(argument->getInteger("alignment").value_or(1));
      llvm::StringRef dtype = argument->getString("dtype").value_or("");
      reflected.tensorElementSize = dtype == "f64"    ? 8
                                    : dtype == "f16"  ? 2
                                    : dtype == "bool" ? 1
                                                      : 4;
      if (llvm::json::Array *shape = argument->getArray("shape")) {
        size_t elements = 1;
        for (llvm::json::Value &dimension : *shape)
          elements *= static_cast<size_t>(dimension.getAsInteger().value_or(0));
        reflected.tensorElements = elements;
        reflected.tensorBytes = elements * reflected.tensorElementSize;
      }
      output.arguments.push_back(std::move(reflected));
    }
    return true;
  }
  error = "selected entry is absent from reflection";
  return false;
}

std::string readFile(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>());
}

} // namespace

struct VernonRuntimeContext {
  VernonRuntimeBackend backend{VERNON_RUNTIME_CPU};
  std::string error;
  size_t liveBuffers{0};
  size_t liveKernels{0};
#if defined(VERNON_HAS_CUDA_RUNTIME)
  CudaDevice device{};
  CudaContext cudaContext{};
#endif
};

struct VernonDeviceBuffer {
  VernonRuntimeContext *context{};
  size_t size{};
  size_t alignment{};
  std::vector<unsigned char> host;
#if defined(VERNON_HAS_CUDA_RUNTIME)
  CudaDevicePointer device{};
#endif
};

struct VernonLoadedKernel {
  VernonRuntimeContext *context{};
  ReflectedEntry reflection;
  VernonCpuEntryPoint cpuEntry{};
  std::unique_ptr<llvm::orc::LLJIT> cpuJit;
#if defined(VERNON_HAS_CUDA_RUNTIME)
  CudaModule cudaModule{};
  CudaFunction cudaFunction{};
#endif
};

namespace {

VernonStatus fail(VernonRuntimeContext *context, std::string message,
                  VernonStatus status = VERNON_STATUS_INVALID_ARGUMENT) {
  if (context)
    context->error = std::move(message);
  return status;
}

#if defined(VERNON_HAS_CUDA_RUNTIME)
VernonStatus cudaFail(VernonRuntimeContext *context, CudaResult result,
                      const char *operation) {
  if (result == kCudaSuccess)
    return VERNON_STATUS_OK;
  const char *name = nullptr;
  const char *description = nullptr;
  CudaDriver &driver = cudaDriver();
  driver.errorName(result, &name);
  driver.errorString(result, &description);
  return fail(context,
              std::string(operation) +
                  " failed: " + (name ? name : "CUDA_ERROR") + " (" +
                  (description ? description : "unknown") + ")",
              VERNON_STATUS_INTERNAL_ERROR);
}
#endif

} // namespace

extern "C" {

VernonRuntimeCapabilities
vernonRuntimeGetCapabilities(VernonRuntimeBackend backend) {
  static std::string diagnostic;
  diagnostic.clear();
  VernonRuntimeCapabilities capabilities{};
  capabilities.supports_compute = 1;
  if (backend == VERNON_RUNTIME_CPU) {
    capabilities.available = 1;
  } else if (backend == VERNON_RUNTIME_CUDA) {
#if defined(VERNON_HAS_CUDA_RUNTIME)
    CudaDriver &driver = cudaDriver();
    if (!driver.load()) {
      diagnostic = driver.error.empty() ? "CUDA Driver API is unavailable"
                                        : driver.error;
    } else {
      CudaDevice device{};
      CudaResult status = driver.init(0);
      if (status == kCudaSuccess)
        status = driver.deviceGet(&device, 0);
      capabilities.available = status == kCudaSuccess;
      if (!capabilities.available)
        diagnostic = "CUDA Driver API loaded but no usable device was found";
    }
#else
    diagnostic = "VernonDSLRuntime was built without CUDA Driver support";
#endif
  } else {
    diagnostic = "unknown runtime backend";
  }
  capabilities.diagnostic = {diagnostic.data(), diagnostic.size()};
  return capabilities;
}

VernonRuntimeContext *vernonRuntimeCreate(VernonRuntimeBackend backend,
                                          uint32_t deviceIndex) {
  static std::once_flag nativeTargetInitialization;
  std::call_once(nativeTargetInitialization, [] {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();
  });
  auto context = std::make_unique<VernonRuntimeContext>();
  context->backend = backend;
  if (backend == VERNON_RUNTIME_CPU)
    return context.release();
  if (backend != VERNON_RUNTIME_CUDA)
    return nullptr;
#if defined(VERNON_HAS_CUDA_RUNTIME)
  CudaDriver &driver = cudaDriver();
  if (!driver.load() || driver.init(0) != kCudaSuccess ||
      driver.deviceGet(&context->device, static_cast<int>(deviceIndex)) !=
          kCudaSuccess ||
      driver.primaryContextRetain(&context->cudaContext, context->device) !=
          kCudaSuccess)
    return nullptr;
  if (driver.contextSetCurrent(context->cudaContext) != kCudaSuccess) {
    driver.primaryContextRelease(context->device);
    return nullptr;
  }
  return context.release();
#else
  (void)deviceIndex;
  return nullptr;
#endif
}

VernonStatus vernonRuntimeDestroy(VernonRuntimeContext *context) {
  if (!context)
    return VERNON_STATUS_OK;
  if (context->liveBuffers || context->liveKernels)
    return fail(context, "runtime context still owns live handles");
#if defined(VERNON_HAS_CUDA_RUNTIME)
  if (context->backend == VERNON_RUNTIME_CUDA) {
    cudaDriver().contextSynchronize();
    cudaDriver().primaryContextRelease(context->device);
  }
#endif
  delete context;
  return VERNON_STATUS_OK;
}

VernonStringView
vernonRuntimeGetLastError(const VernonRuntimeContext *context) {
  if (!context)
    return {nullptr, 0};
  return {context->error.data(), context->error.size()};
}

VernonDeviceBuffer *vernonRuntimeBufferAllocate(VernonRuntimeContext *context,
                                                size_t size, size_t alignment) {
  if (!context || !size || !alignment || (alignment & (alignment - 1))) {
    fail(context, "buffer size and power-of-two alignment are required");
    return nullptr;
  }
  auto buffer = std::make_unique<VernonDeviceBuffer>();
  buffer->context = context;
  buffer->size = size;
  buffer->alignment = alignment;
  if (context->backend == VERNON_RUNTIME_CPU) {
    buffer->host.resize(size);
  } else {
#if defined(VERNON_HAS_CUDA_RUNTIME)
    if (cudaFail(context, cudaDriver().memoryAllocate(&buffer->device, size),
                 "cuMemAlloc") != VERNON_STATUS_OK)
      return nullptr;
#else
    return nullptr;
#endif
  }
  ++context->liveBuffers;
  return buffer.release();
}

VernonStatus vernonRuntimeBufferFree(VernonDeviceBuffer *buffer) {
  if (!buffer)
    return VERNON_STATUS_OK;
  VernonRuntimeContext *context = buffer->context;
#if defined(VERNON_HAS_CUDA_RUNTIME)
  if (context->backend == VERNON_RUNTIME_CUDA && buffer->device)
    if (cudaFail(context, cudaDriver().memoryFree(buffer->device),
                 "cuMemFree") != VERNON_STATUS_OK)
      return VERNON_STATUS_INTERNAL_ERROR;
#endif
  --context->liveBuffers;
  delete buffer;
  return VERNON_STATUS_OK;
}

VernonStatus vernonRuntimeCopyFromHost(VernonDeviceBuffer *buffer,
                                       size_t offset, const void *source,
                                       size_t size) {
  if (!buffer || !source || offset > buffer->size ||
      size > buffer->size - offset)
    return fail(buffer ? buffer->context : nullptr,
                "invalid host upload range");
  if (buffer->context->backend == VERNON_RUNTIME_CPU)
    std::memcpy(buffer->host.data() + offset, source, size);
#if defined(VERNON_HAS_CUDA_RUNTIME)
  else
    return cudaFail(
        buffer->context,
        cudaDriver().copyHostToDevice(buffer->device + offset, source, size),
        "cuMemcpyHtoD");
#endif
  return VERNON_STATUS_OK;
}

VernonStatus vernonRuntimeCopyToHost(const VernonDeviceBuffer *buffer,
                                     size_t offset, void *destination,
                                     size_t size) {
  if (!buffer || !destination || offset > buffer->size ||
      size > buffer->size - offset)
    return fail(buffer ? buffer->context : nullptr,
                "invalid host download range");
  if (buffer->context->backend == VERNON_RUNTIME_CPU)
    std::memcpy(destination, buffer->host.data() + offset, size);
#if defined(VERNON_HAS_CUDA_RUNTIME)
  else
    return cudaFail(buffer->context,
                    cudaDriver().copyDeviceToHost(
                        destination, buffer->device + offset, size),
                    "cuMemcpyDtoH");
#endif
  return VERNON_STATUS_OK;
}

VernonLoadedKernel *vernonRuntimeLoadCpuEntry(VernonRuntimeContext *context,
                                              VernonCpuEntryPoint entryPoint,
                                              const char *reflection,
                                              size_t reflectionSize,
                                              const char *entry,
                                              size_t entrySize) {
  if (!context || context->backend != VERNON_RUNTIME_CPU || !entryPoint ||
      !reflection || !entry)
    return nullptr;
  auto kernel = std::make_unique<VernonLoadedKernel>();
  kernel->context = context;
  kernel->cpuEntry = entryPoint;
  if (!parseReflection(llvm::StringRef(reflection, reflectionSize),
                       llvm::StringRef(entry, entrySize), kernel->reflection,
                       context->error))
    return nullptr;
  ++context->liveKernels;
  return kernel.release();
}

VernonLoadedKernel *
vernonRuntimeLoadArtifact(VernonRuntimeContext *context, const void *artifact,
                          size_t artifactSize, const char *reflection,
                          size_t reflectionSize, const char *entry,
                          size_t entrySize) {
  if (!context || !artifact || !reflection || !entry)
    return nullptr;
  if (context->backend == VERNON_RUNTIME_CPU) {
    auto kernel = std::make_unique<VernonLoadedKernel>();
    kernel->context = context;
    if (!parseReflection(llvm::StringRef(reflection, reflectionSize),
                         llvm::StringRef(entry, entrySize), kernel->reflection,
                         context->error))
      return nullptr;
    auto targetMachine = llvm::orc::JITTargetMachineBuilder::detectHost();
    if (!targetMachine) {
      fail(context, llvm::toString(targetMachine.takeError()),
           VERNON_STATUS_INTERNAL_ERROR);
      return nullptr;
    }
    auto dataLayout = targetMachine->getDefaultDataLayoutForTarget();
    if (!dataLayout) {
      fail(context, llvm::toString(dataLayout.takeError()),
           VERNON_STATUS_INTERNAL_ERROR);
      return nullptr;
    }
    auto jit = llvm::orc::LLJITBuilder()
                   .setJITTargetMachineBuilder(std::move(*targetMachine))
                   .setDataLayout(*dataLayout)
                   .create();
    if (!jit) {
      fail(context, llvm::toString(jit.takeError()),
           VERNON_STATUS_INTERNAL_ERROR);
      return nullptr;
    }
    auto llvmContext = std::make_unique<llvm::LLVMContext>();
    llvm::SMDiagnostic diagnostic;
    std::unique_ptr<llvm::Module> module = llvm::parseAssemblyString(
        llvm::StringRef(static_cast<const char *>(artifact), artifactSize),
        diagnostic, *llvmContext);
    if (!module) {
      std::string message;
      llvm::raw_string_ostream stream(message);
      diagnostic.print("VernonDSLRuntime", stream);
      fail(context, stream.str(), VERNON_STATUS_PARSE_ERROR);
      return nullptr;
    }
    module->setDataLayout(*dataLayout);
    module->setTargetTriple((*jit)->getTargetTriple());
    if (llvm::Error error = (*jit)->addIRModule(llvm::orc::ThreadSafeModule(
            std::move(module), std::move(llvmContext)))) {
      fail(context, llvm::toString(std::move(error)),
           VERNON_STATUS_INTERNAL_ERROR);
      return nullptr;
    }
    std::string symbolName = "__vernon_cpu_" + std::string(entry, entrySize);
    auto symbol = (*jit)->lookup(symbolName);
    if (!symbol) {
      fail(context, llvm::toString(symbol.takeError()),
           VERNON_STATUS_INTERNAL_ERROR);
      return nullptr;
    }
    kernel->cpuEntry = symbol->toPtr<VernonCpuEntryPoint>();
    kernel->cpuJit = std::move(*jit);
    ++context->liveKernels;
    return kernel.release();
  }
  if (context->backend != VERNON_RUNTIME_CUDA)
    return nullptr;
#if defined(VERNON_HAS_CUDA_RUNTIME)
  auto kernel = std::make_unique<VernonLoadedKernel>();
  kernel->context = context;
  if (!parseReflection(llvm::StringRef(reflection, reflectionSize),
                       llvm::StringRef(entry, entrySize), kernel->reflection,
                       context->error))
    return nullptr;
  std::string image(static_cast<const char *>(artifact), artifactSize);
  image.push_back('\0');
  if (cudaFail(context,
               cudaDriver().moduleLoadData(&kernel->cudaModule, image.data(), 0,
                                           nullptr, nullptr),
               "cuModuleLoadDataEx") != VERNON_STATUS_OK)
    return nullptr;
  std::string symbol(entry, entrySize);
  if (cudaFail(context,
               cudaDriver().moduleGetFunction(
                   &kernel->cudaFunction, kernel->cudaModule, symbol.c_str()),
               "cuModuleGetFunction") != VERNON_STATUS_OK) {
    cudaDriver().moduleUnload(kernel->cudaModule);
    return nullptr;
  }
  ++context->liveKernels;
  return kernel.release();
#else
  (void)artifactSize;
  (void)reflectionSize;
  (void)entrySize;
  return nullptr;
#endif
}

VernonLoadedKernel *
vernonRuntimeLoadComputeBundle(VernonRuntimeContext *context,
                               const char *directory) {
  if (!context || !directory)
    return nullptr;
  std::filesystem::path root(directory);
  std::string manifestText = readFile(root / "compute.json");
  llvm::Expected<llvm::json::Value> parsed = llvm::json::parse(manifestText);
  if (!parsed || !parsed->getAsObject()) {
    fail(context, "cannot parse compute.json");
    return nullptr;
  }
  llvm::json::Object &manifest = *parsed->getAsObject();
  if (manifest.getInteger("schema_version").value_or(0) != 1 ||
      manifest.getInteger("gpu_launch_abi_version").value_or(0) != 1) {
    fail(context, "unsupported compute bundle schema or launch ABI");
    return nullptr;
  }
  llvm::StringRef expectedTarget =
      context->backend == VERNON_RUNTIME_CPU ? "cpu" : "cuda";
  if (manifest.getString("target").value_or("") != expectedTarget) {
    fail(context, "compute bundle target does not match runtime backend");
    return nullptr;
  }
  std::string entry = manifest.getString("entry").value_or("").str();
  std::string artifactName = manifest.getString("artifact").value_or("").str();
  llvm::json::Value *reflectionValue = manifest.get("reflection");
  if (entry.empty() || artifactName.empty() || !reflectionValue) {
    fail(context, "compute bundle is missing required fields");
    return nullptr;
  }
  std::string reflection;
  llvm::raw_string_ostream reflectionStream(reflection);
  reflectionStream << *reflectionValue;
  reflectionStream.flush();
  std::string artifact = readFile(root / artifactName);
  if (artifact.empty()) {
    fail(context, "compute bundle artifact is empty or absent");
    return nullptr;
  }
  if (manifest.getInteger("artifact_size").value_or(-1) !=
      static_cast<int64_t>(artifact.size())) {
    fail(context, "compute bundle artifact size mismatch");
    return nullptr;
  }
  std::string expectedHash =
      manifest.getString("artifact_sha256").value_or("").str();
  llvm::ArrayRef<uint8_t> bytes(
      reinterpret_cast<const uint8_t *>(artifact.data()), artifact.size());
  std::string actualHash = llvm::toHex(llvm::SHA256::hash(bytes), true);
  if (expectedHash.empty() ||
      !llvm::StringRef(expectedHash).equals_insensitive(actualHash)) {
    fail(context, "compute bundle artifact SHA-256 mismatch");
    return nullptr;
  }
  return vernonRuntimeLoadArtifact(context, artifact.data(), artifact.size(),
                                   reflection.data(), reflection.size(),
                                   entry.data(), entry.size());
}

VernonStatus vernonRuntimeKernelUnload(VernonLoadedKernel *kernel) {
  if (!kernel)
    return VERNON_STATUS_OK;
#if defined(VERNON_HAS_CUDA_RUNTIME)
  if (kernel->context->backend == VERNON_RUNTIME_CUDA && kernel->cudaModule)
    if (cudaFail(kernel->context, cudaDriver().moduleUnload(kernel->cudaModule),
                 "cuModuleUnload") != VERNON_STATUS_OK)
      return VERNON_STATUS_INTERNAL_ERROR;
#endif
  --kernel->context->liveKernels;
  delete kernel;
  return VERNON_STATUS_OK;
}

VernonStatus vernonRuntimeLaunch(VernonLoadedKernel *kernel,
                                 VernonLaunchSize globalSize,
                                 const VernonLaunchArgument *arguments,
                                 size_t argumentCount) {
  if (!kernel || !globalSize.x || !globalSize.y || !globalSize.z)
    return fail(kernel ? kernel->context : nullptr,
                "grid dimensions must be positive");
  const size_t expected = static_cast<size_t>(std::count_if(
      kernel->reflection.arguments.begin(), kernel->reflection.arguments.end(),
      [](const ReflectedArgument &argument) {
        return argument.kind != "builtin";
      }));
  if (argumentCount != expected || (expected && !arguments))
    return fail(kernel->context,
                "launch argument count does not match reflection");
  size_t validatedIndex = 0;
  for (const ReflectedArgument &reflected : kernel->reflection.arguments) {
    if (reflected.kind == "builtin")
      continue;
    const VernonLaunchArgument &argument = arguments[validatedIndex++];
    if (reflected.kind == "tensor") {
      if (argument.kind != VERNON_LAUNCH_TENSOR || !argument.buffer ||
          argument.buffer->context != kernel->context ||
          (reflected.tensorBytes &&
           argument.buffer->size < reflected.tensorBytes) ||
          argument.buffer->alignment < reflected.alignment)
        return fail(kernel->context,
                    "Tensor launch argument does not match reflection");
    } else if (argument.kind != VERNON_LAUNCH_SCALAR || !argument.scalar_data ||
               (reflected.cpuSize &&
                argument.scalar_size != reflected.cpuSize)) {
      return fail(kernel->context,
                  "scalar launch argument does not match reflection");
    }
  }

  if (kernel->context->backend == VERNON_RUNTIME_CPU) {
    std::vector<unsigned char> packed(kernel->reflection.cpuArgumentsSize);
    size_t supplied = 0;
    for (const ReflectedArgument &reflected : kernel->reflection.arguments) {
      if (reflected.kind == "builtin")
        continue;
      const VernonLaunchArgument &argument = arguments[supplied++];
      if (reflected.kind == "tensor") {
        if (argument.kind != VERNON_LAUNCH_TENSOR || !argument.buffer ||
            argument.buffer->context != kernel->context ||
            reflected.cpuSize != sizeof(uintptr_t) ||
            (reflected.tensorBytes &&
             argument.buffer->size < reflected.tensorBytes) ||
            argument.buffer->alignment < reflected.alignment)
          return fail(kernel->context, "invalid Tensor launch argument");
        uintptr_t pointer =
            reinterpret_cast<uintptr_t>(argument.buffer->host.data());
        std::memcpy(packed.data() + reflected.cpuOffset, &pointer,
                    sizeof(pointer));
      } else {
        if (argument.kind != VERNON_LAUNCH_SCALAR ||
            argument.scalar_size != reflected.cpuSize || !argument.scalar_data)
          return fail(kernel->context, "invalid scalar launch argument");
        std::memcpy(packed.data() + reflected.cpuOffset, argument.scalar_data,
                    argument.scalar_size);
      }
    }
    for (uint32_t z = 0; z < globalSize.z; ++z)
      for (uint32_t y = 0; y < globalSize.y; ++y)
        for (uint32_t x = 0; x < globalSize.x; ++x) {
          uint32_t id[3]{x, y, z};
          for (const ReflectedArgument &reflected :
               kernel->reflection.arguments)
            if (reflected.kind == "builtin" &&
                reflected.builtin == "global_invocation_id")
              std::memcpy(packed.data() + reflected.cpuOffset, id,
                          std::min(reflected.cpuSize, sizeof(id)));
          VernonCpuInvocation invocation{packed.data(), packed.size(), nullptr,
                                         0, nullptr};
          VernonStatus status = kernel->cpuEntry(&invocation);
          if (status != VERNON_STATUS_OK)
            return fail(kernel->context, "CPU kernel invocation failed",
                        status);
        }
    return VERNON_STATUS_OK;
  }

#if defined(VERNON_HAS_CUDA_RUNTIME)
  struct CudaMemRefDescriptor {
    CudaDevicePointer allocated;
    CudaDevicePointer aligned;
    uint64_t offset;
    uint64_t size;
    uint64_t stride;
  };
  std::vector<CudaMemRefDescriptor> descriptors;
  std::vector<void *> parameters;
  descriptors.reserve(argumentCount);
  parameters.reserve(argumentCount * 5);
  size_t suppliedIndex = 0;
  for (const ReflectedArgument &reflected : kernel->reflection.arguments) {
    if (reflected.kind == "builtin")
      continue;
    const VernonLaunchArgument &argument = arguments[suppliedIndex++];
    if (argument.kind == VERNON_LAUNCH_TENSOR) {
      if (!argument.buffer || argument.buffer->context != kernel->context)
        return fail(kernel->context, "invalid CUDA Tensor argument");
      descriptors.push_back(
          {argument.buffer->device, argument.buffer->device, 0,
           reflected.tensorElements
               ? reflected.tensorElements
               : argument.buffer->size /
                     std::max(reflected.tensorElementSize, size_t{1}),
           1});
      CudaMemRefDescriptor &descriptor = descriptors.back();
      parameters.push_back(&descriptor.allocated);
      parameters.push_back(&descriptor.aligned);
      parameters.push_back(&descriptor.offset);
      parameters.push_back(&descriptor.size);
      parameters.push_back(&descriptor.stride);
    } else {
      if (!argument.scalar_data || !argument.scalar_size)
        return fail(kernel->context, "invalid CUDA scalar argument");
      parameters.push_back(const_cast<void *>(argument.scalar_data));
    }
  }
  uint32_t *workgroup = kernel->reflection.workgroup;
  return cudaFail(kernel->context,
                  cudaDriver().launchKernel(
                      kernel->cudaFunction,
                      (globalSize.x + workgroup[0] - 1) / workgroup[0],
                      (globalSize.y + workgroup[1] - 1) / workgroup[1],
                      (globalSize.z + workgroup[2] - 1) / workgroup[2],
                      workgroup[0], workgroup[1], workgroup[2], 0, nullptr,
                      parameters.data(), nullptr),
                  "cuLaunchKernel");
#else
  return fail(kernel->context, "CUDA runtime support is not built",
              VERNON_STATUS_UNSUPPORTED_TARGET);
#endif
}

VernonStatus vernonRuntimeSynchronize(VernonRuntimeContext *context) {
  if (!context)
    return VERNON_STATUS_INVALID_ARGUMENT;
#if defined(VERNON_HAS_CUDA_RUNTIME)
  if (context->backend == VERNON_RUNTIME_CUDA)
    return cudaFail(context, cudaDriver().contextSynchronize(),
                    "cuCtxSynchronize");
#endif
  return VERNON_STATUS_OK;
}

} // extern "C"

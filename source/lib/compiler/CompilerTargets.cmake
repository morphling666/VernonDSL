add_subdirectory("${VERNON_SOURCE_DIR}/include/mlir/Dialect/Vernon"
                 "${CMAKE_CURRENT_BINARY_DIR}/include/mlir/Dialect/Vernon")
add_subdirectory("${VERNON_SOURCE_DIR}/lib/Dialect/Vernon" "${CMAKE_CURRENT_BINARY_DIR}/lib/Dialect/Vernon")

add_library(
    VernonDSLCompiler SHARED
    compiler_artifacts.cpp
    compiler_cpu.cpp
    compiler_cuda.cpp
    compiler_dispatch.cpp
    compiler_dxc.cpp
    compiler_frontend.cpp
    compiler_internal.cpp
    compiler_reflection.cpp
    compiler_spirv.cpp
    VernonCompiler.cpp
    VernonCpuAbiWrapper.cpp)
llvm_map_components_to_libnames(
    VERNON_LLVM_JIT_LIBS
    AllTargetsAsmParsers
    AllTargetsCodeGens
    AllTargetsDescs
    AllTargetsInfos
    AsmParser
    Core
    OrcJIT
    Target
    native)
target_compile_definitions(VernonDSLCompiler PRIVATE VERNON_DSL_COMPILER_BUILD)
if(WIN32)
    option(VERNON_FETCH_DXC "Download the pinned DXC redistributable when no executable is specified" ON)
    set(VERNON_DXC_EXECUTABLE
        ""
        CACHE FILEPATH "Optional DXC executable override")
    if(NOT VERNON_DXC_EXECUTABLE AND VERNON_FETCH_DXC)
        include(FetchContent)
        FetchContent_Declare(
            vernon_dxc
            URL "https://api.nuget.org/v3-flatcontainer/microsoft.direct3d.dxc/1.9.2602.24/microsoft.direct3d.dxc.1.9.2602.24.nupkg"
            URL_HASH
                "SHA512=354182aa58f528d5138ff2bfc97fd48cd1cfe8d0fcff146830d35d7630e73e4bb1db3aa39b0e807dfe905bf13d4b96229f9385ac14d562831dbfdd0d28eafb05"
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
        FetchContent_MakeAvailable(vernon_dxc)
        if(CMAKE_SIZEOF_VOID_P EQUAL 4)
            set(_vernon_dxc_arch x86)
        elseif(CMAKE_GENERATOR_PLATFORM MATCHES "^[Aa][Rr][Mm]64$" OR CMAKE_SYSTEM_PROCESSOR MATCHES
                                                                      "^(ARM64|arm64|aarch64)$")
            set(_vernon_dxc_arch arm64)
        else()
            set(_vernon_dxc_arch x64)
        endif()
        set(VERNON_DXC_EXECUTABLE
            "${vernon_dxc_SOURCE_DIR}/build/native/bin/${_vernon_dxc_arch}/dxc.exe"
            CACHE FILEPATH "DXC executable used for DirectX artifact compilation" FORCE)
    elseif(NOT VERNON_DXC_EXECUTABLE)
        find_program(VERNON_DXC_EXECUTABLE NAMES dxc.exe)
    endif()
    if(VERNON_DXC_EXECUTABLE AND EXISTS "${VERNON_DXC_EXECUTABLE}")
        file(TO_CMAKE_PATH "${VERNON_DXC_EXECUTABLE}" _vernon_dxc_path)
        target_compile_definitions(VernonDSLCompiler PRIVATE VERNON_DXC_EXECUTABLE="${_vernon_dxc_path}")
        get_filename_component(_vernon_dxc_directory "${VERNON_DXC_EXECUTABLE}" DIRECTORY)
        message(STATUS "Using DXC: ${VERNON_DXC_EXECUTABLE}")
    else()
        message(WARNING "DXC was not found; the DirectX target will report unavailable")
    endif()
endif()
if(MSVC)
    target_compile_options(VernonDSLCompiler PRIVATE /EHsc)
endif()
target_sources(VernonDSLCompiler PRIVATE compiler_spirv_cross.cpp)
if(NOT MSVC)
    target_compile_options(VernonDSLCompiler PRIVATE -fexceptions)
endif()
target_include_directories(VernonDSLCompiler PUBLIC $<BUILD_INTERFACE:${VERNON_SOURCE_DIR}/include>
                                                    $<INSTALL_INTERFACE:include>)
target_link_libraries(
    VernonDSLCompiler
    PRIVATE MLIRArithToLLVM
            MLIRBufferizationTransforms
            MLIRControlFlowToLLVM
            MLIRConvertToLLVMPass
            MLIRVernonDialect
            MLIRVernonTransforms
            MLIRFuncDialect
            MLIRFuncToLLVM
            MLIRGPUToSPIRV
            MLIRGPUPipelines
            MLIRIR
            MLIRIndexToLLVM
            MLIRLinalgTransforms
            MLIRMathToLLVM
            MLIRMathToSPIRV
            MLIRMemRefToLLVM
            MLIRParser
            MLIRPass
            MLIRRegisterAllDialects
            MLIRRegisterAllExtensions
            MLIRSCFToControlFlow
            MLIRSPIRVDialect
            MLIRSPIRVSerialization
            MLIRSPIRVTransforms
            MLIRSupport
            MLIRToLLVMIRTranslationRegistration
            MLIRUBToLLVM
            MLIRVectorToLLVM
            ${VERNON_LLVM_JIT_LIBS})
if(WIN32)
    target_link_libraries(VernonDSLCompiler PRIVATE lldCommon lldCOFF)
elseif(APPLE)
    target_link_libraries(VernonDSLCompiler PRIVATE lldCommon lldMachO)
else()
    target_link_libraries(VernonDSLCompiler PRIVATE lldCommon lldELF)
endif()
target_link_libraries(VernonDSLCompiler PRIVATE spirv-cross-glsl spirv-cross-hlsl spirv-cross-msl)

if(SKBUILD)
    set(_vernon_compiler_runtime_destination vernon_dsl)
    set(_vernon_compiler_library_destination vernon_dsl)
else()
    set(_vernon_compiler_runtime_destination bin)
    set(_vernon_compiler_library_destination lib)
endif()
if(VERNON_DXC_EXECUTABLE)
    foreach(_vernon_dxc_file IN ITEMS dxc.exe dxcompiler.dll dxil.dll)
        if(EXISTS "${_vernon_dxc_directory}/${_vernon_dxc_file}")
            add_custom_command(
                TARGET VernonDSLCompiler
                POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_vernon_dxc_directory}/${_vernon_dxc_file}"
                        "$<TARGET_FILE_DIR:VernonDSLCompiler>/${_vernon_dxc_file}"
                VERBATIM)
            install(
                FILES "${_vernon_dxc_directory}/${_vernon_dxc_file}"
                DESTINATION ${_vernon_compiler_runtime_destination}
                COMPONENT VernonWheel)
        endif()
    endforeach()
endif()
install(
    TARGETS VernonDSLCompiler
    RUNTIME DESTINATION ${_vernon_compiler_runtime_destination} COMPONENT VernonWheel
    LIBRARY DESTINATION ${_vernon_compiler_library_destination} COMPONENT VernonWheel
    ARCHIVE DESTINATION lib COMPONENT VernonDevelopment)
install(FILES "${VERNON_SOURCE_DIR}/include/VernonCommon.h" "${VERNON_SOURCE_DIR}/include/VernonCompiler.h"
        DESTINATION include)
install(FILES "${VERNON_SOURCE_DIR}/include/vernon-c/Common.h" "${VERNON_SOURCE_DIR}/include/vernon-c/Compiler.h"
        DESTINATION include/vernon-c)

add_library(vernon-compile-packaging STATIC "${VERNON_SOURCE_DIR}/tools/vernon_compile_packaging.cpp")
target_include_directories(vernon-compile-packaging PUBLIC "${VERNON_SOURCE_DIR}/tools")
target_link_libraries(
    vernon-compile-packaging
    PUBLIC VernonDSLCompiler
    PRIVATE LLVMSupport LLVMTargetParser)
add_executable(vernon-compile "${VERNON_SOURCE_DIR}/tools/vernon_compile.cpp")
target_link_libraries(vernon-compile PRIVATE vernon-compile-packaging)
if(MSVC)
    target_compile_options(vernon-compile-packaging PRIVATE /EHsc)
    target_compile_options(vernon-compile PRIVATE /EHsc)
endif()
if(SKBUILD)
    install(TARGETS vernon-compile RUNTIME DESTINATION vernon_dsl COMPONENT VernonWheel)
else()
    install(TARGETS vernon-compile RUNTIME DESTINATION bin)
endif()

add_subdirectory("${VERNON_SOURCE_DIR}/tools/vernon_opt" "${CMAKE_CURRENT_BINARY_DIR}/tools/vernon_opt")

set(VERNON_LIT_CONFIG_INPUT "${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py.in")
set(VERNON_LIT_CONFIG "${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py")

find_package(LLVM REQUIRED CONFIG)

if(NOT Python3_EXECUTABLE)
    set(Python3_EXECUTABLE "${Python_EXECUTABLE}")
endif()
if(NOT LLVM_EXTERNAL_LIT)
    set(LLVM_EXTERNAL_LIT "${CMAKE_CURRENT_SOURCE_DIR}/run_lit.py")
endif()

if(NOT TARGET FileCheck)
    add_executable(FileCheck EXCLUDE_FROM_ALL
                   "${VERNON_REPOSITORY_ROOT}/llvm-project/llvm/utils/FileCheck/FileCheck.cpp")
    target_include_directories(FileCheck PRIVATE ${LLVM_INCLUDE_DIRS})
    target_link_directories(FileCheck PRIVATE ${LLVM_LIBRARY_DIRS})
    target_link_libraries(FileCheck PRIVATE VernonCompilerBuildHeaders LLVMFileCheck LLVMSupport)
endif()
if(NOT TARGET not)
    add_executable(not EXCLUDE_FROM_ALL "${VERNON_REPOSITORY_ROOT}/llvm-project/llvm/utils/not/not.cpp")
    target_include_directories(not PRIVATE ${LLVM_INCLUDE_DIRS})
    target_link_directories(not PRIVATE ${LLVM_LIBRARY_DIRS})
    target_link_libraries(not PRIVATE VernonCompilerBuildHeaders LLVMSupport)
endif()

configure_file(lit.site.cfg.py.in "${VERNON_LIT_CONFIG_INPUT}" @ONLY)
file(
    GENERATE
    OUTPUT "${VERNON_LIT_CONFIG}"
    INPUT "${VERNON_LIT_CONFIG_INPUT}")

add_lit_testsuite(
    check-vernon-mlir
    "Running Vernon MLIR tests"
    "${CMAKE_CURRENT_BINARY_DIR}"
    DEPENDS
    FileCheck
    not
    vernon-opt)
set_target_properties(check-vernon-mlir PROPERTIES FOLDER "Tests")

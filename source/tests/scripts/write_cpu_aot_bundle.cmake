include("${CMAKE_CURRENT_LIST_DIR}/../../../cmake/VernonVersions.cmake")

if(NOT DEFINED ARTIFACT
   OR NOT DEFINED OUTPUT
   OR NOT DEFINED OPERATING_SYSTEM
   OR NOT DEFINED ARCHITECTURE)
    message(FATAL_ERROR "CPU AOT bundle generator arguments are incomplete")
endif()

file(MAKE_DIRECTORY "${OUTPUT}")
get_filename_component(ARTIFACT_NAME "${ARTIFACT}" NAME)
file(
    COPY_FILE
    "${ARTIFACT}"
    "${OUTPUT}/${ARTIFACT_NAME}"
    ONLY_IF_DIFFERENT)
file(SIZE "${ARTIFACT}" ARTIFACT_SIZE)
file(SHA256 "${ARTIFACT}" ARTIFACT_SHA256)

if(OPERATING_SYSTEM STREQUAL "windows")
    set(TARGET_TRIPLE "${ARCHITECTURE}-pc-windows-msvc")
    set(OBJECT_FORMAT "coff")
elseif(OPERATING_SYSTEM STREQUAL "macos" OR OPERATING_SYSTEM STREQUAL "darwin")
    set(TARGET_TRIPLE "${ARCHITECTURE}-apple-darwin")
    set(OBJECT_FORMAT "macho")
else()
    set(TARGET_TRIPLE "${ARCHITECTURE}-unknown-linux-gnu")
    set(OBJECT_FORMAT "elf")
endif()

set(PIPELINE_CANONICAL
    "{\"id\":\"cpu/fill\",\"pipeline_version\":${VERNON_PIPELINE_VERSION},\
\"runtime_requirements\":{\"backend\":\"cpu\",\"features\":[\"compute\",\"tensor_views\"],\
\"object_format\":\"${OBJECT_FORMAT}\",\"target_triple\":\"${TARGET_TRIPLE}\"},\
\"stage_artifacts\":{\"fill\":{\"artifact\":{\"format\":\"native_library\",\"path\":\"${ARTIFACT_NAME}\",\
\"sha256\":\"${ARTIFACT_SHA256}\",\"size\":${ARTIFACT_SIZE},\"storage\":\"external\"},\
\"entry\":\"fill\",\
\"reflection\":{\"compiler_contract_version\":${VERNON_COMPILER_CONTRACT_VERSION},\
\"entries\":[{\"arguments\":[\
{\"dtype\":\"f32\",\"element_layout\":{\"alignment\":4,\"byte_size\":4,\
\"layout_hash\":\"cb580e347f23fbe3afbd1c5f72b4d2339b09e33d876f79e9d290445edb43c03b\",\
\"leaves\":[{\"byte_offset\":0,\"dtype\":\"f32\",\"path\":[],\"scalar_count\":1}],\
\"logical_type\":\"f32\"},\"kind\":\"tensor\",\"physical_layouts\":{\"host_value\":{\
\"alignment\":8,\"frame_offset\":0,\"kind\":\"resource_binding\",\"profile\":\"host_value\",\
\"resource_kind\":\"tensor_view_descriptor\",\"size\":8}}},\
{\"builtin\":\"global_invocation_id\",\"kind\":\"builtin\",\
\"physical_layouts\":{\"host_value\":{\"frame_offset\":8,\"kind\":\"cpu_call\",\"profile\":\"host_value\",\
\"root\":{\"alignment\":4,\"byte_strides\":[4],\"children\":[{\"alignment\":4,\"kind\":\"scalar\",\
\"offset\":0,\"representation\":\"i32\",\"size\":4}],\"kind\":\"array\",\"offset\":0,\"shape\":[3],\
\"size\":12}}}}],\
\"dispatch_contract\":{\"requires_unit_workgroup\":false,\"unit_grid_axes\":[]},\
\"name\":\"fill\",\"physical_layouts\":{\"host_value\":{\"packed_arguments_size\":20,\
\"profile\":\"host_value\"}},\"workgroup_size\":[2,2,1]}],\
\"pipeline_version\":${VERNON_PIPELINE_VERSION}},\"stage\":\"compute\",\"symbol\":\"vernon_test_fill\"}},\
\"target\":{\"kind\":\"cpu\",\"options\":{\"triple\":\"${TARGET_TRIPLE}\"}},\"type\":\"pipeline\",\
\"variants\":[{\"key\":[],\"outputs\":[{\"access\":\"write\",\"dtype\":\"f32\",\
\"kind\":\"tensor\",\"location\":0,\"name\":\"result\",\"shape\":[16]}],\
\"parameters\":[{\"access\":\"write\",\"address_space\":\"device\",\
\"element_layout\":{\"alignment\":4,\"byte_size\":4,\
\"layout_hash\":\"cb580e347f23fbe3afbd1c5f72b4d2339b09e33d876f79e9d290445edb43c03b\",\
\"leaves\":[{\"byte_offset\":0,\"dtype\":\"f32\",\"path\":[],\"scalar_count\":1}],\
\"logical_type\":\"f32\"},\"kind\":\"tensor\",\"name\":\"output\",\"shape\":[16],\"slot\":0,\
\"type\":\"!vernon.tensor_view<f32, [16], \\\"write\\\", \\\"device\\\">\",\
\"uses\":[{\"dtype\":\"f32\",\"index\":0,\"interface\":\"storage\",\
\"shape\":[16],\"stage\":\"compute\"}]}],\
\"program\":{\"compute\":\"fill\"}}]}")
string(SHA256 PIPELINE_HASH "${PIPELINE_CANONICAL}")
string(
    SUBSTRING "${PIPELINE_CANONICAL}"
              1
              -1
              PIPELINE_BODY)
file(WRITE "${OUTPUT}/cpu_fill.pipeline.json" "{\"content_hash\":\"${PIPELINE_HASH}\",${PIPELINE_BODY}\n")

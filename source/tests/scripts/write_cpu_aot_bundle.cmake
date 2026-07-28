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
elseif(OPERATING_SYSTEM STREQUAL "darwin")
    set(TARGET_TRIPLE "${ARCHITECTURE}-apple-darwin")
    set(OBJECT_FORMAT "macho")
else()
    set(TARGET_TRIPLE "${ARCHITECTURE}-unknown-linux-gnu")
    set(OBJECT_FORMAT "elf")
endif()

file(
    WRITE "${OUTPUT}/compute.json"
    "{
  \"schema_version\": 3,
  \"cpu_invocation_abi_version\": 1,
  \"target\": \"cpu\",
  \"operating_system\": \"${OPERATING_SYSTEM}\",
  \"architecture\": \"${ARCHITECTURE}\",
  \"entry\": \"fill\",
  \"symbol\": \"vernon_test_fill\",
  \"artifact\": \"${ARTIFACT_NAME}\",
  \"artifact_format\": \"native_library\",
  \"artifact_size\": ${ARTIFACT_SIZE},
  \"artifact_sha256\": \"${ARTIFACT_SHA256}\",
  \"reflection\": {
    \"gpu_launch_abi_version\": 1,
    \"entries\": [{
      \"name\": \"fill\",
      \"physical_layouts\": {
        \"host_value\": {\"profile\":\"host_value\",\"packed_arguments_size\":20}
      },
      \"workgroup_size\": [2, 2, 1],
      \"arguments\": [
        {
          \"kind\": \"tensor\",
          \"dtype\": \"f32\",
          \"element_layout\": {\"logical_type\":\"f32\",\"byte_size\":4,\"alignment\":4,
            \"layout_hash\":\"cb580e347f23fbe3afbd1c5f72b4d2339b09e33d876f79e9d290445edb43c03b\",
            \"leaves\":[{\"path\":[],\"dtype\":\"f32\",\"byte_offset\":0,\"scalar_count\":1}]},
          \"physical_layouts\":{\"host_value\":{\"profile\":\"host_value\",\"size\":8,\"alignment\":8,
            \"byte_strides\":[],\"offset\":0}}
        },
        {
          \"kind\": \"builtin\",
          \"builtin\": \"global_invocation_id\",
          \"physical_layouts\":{\"host_value\":{\"profile\":\"host_value\",\"size\":12,\"alignment\":4,
            \"byte_strides\":[],\"offset\":8}}
        }
      ]
    }]
  }
}
")

set(PIPELINE_CANONICAL
    "{\"features\":[],\"id\":\"cpu/fill\",\"invocation_abi_version\":6,\
\"runtime_requirements\":{\"backend\":\"cpu\",\"features\":[\"compute\",\"tensor_views\"],\
\"invocation_abi_version\":1,\"object_format\":\"${OBJECT_FORMAT}\",\"target_triple\":\"${TARGET_TRIPLE}\"},\
\"schema_version\":4,\
\"stage_artifacts\":{\"fill\":{\"architecture\":\"${ARCHITECTURE}\",\
\"artifact\":{\"format\":\"native_library\",\"path\":\"${ARTIFACT_NAME}\",\
\"sha256\":\"${ARTIFACT_SHA256}\",\"size\":${ARTIFACT_SIZE},\"storage\":\"external\"},\
\"cpu_invocation_abi_version\":1,\"entry\":\"fill\",\"format\":\"native_library\",\
\"id\":\"fill\",\"operating_system\":\"${OPERATING_SYSTEM}\",\
\"reflection\":{\"entries\":[{\"arguments\":[\
{\"dtype\":\"f32\",\"element_layout\":{\"alignment\":4,\"byte_size\":4,\
\"layout_hash\":\"cb580e347f23fbe3afbd1c5f72b4d2339b09e33d876f79e9d290445edb43c03b\",\
\"leaves\":[{\"byte_offset\":0,\"dtype\":\"f32\",\"path\":[],\"scalar_count\":1}],\
\"logical_type\":\"f32\"},\"kind\":\"tensor\",\"physical_layouts\":{\"host_value\":{\
\"alignment\":8,\"byte_strides\":[],\"offset\":0,\"profile\":\"host_value\",\"size\":8}}},\
{\"builtin\":\"global_invocation_id\",\"kind\":\"builtin\",\
\"physical_layouts\":{\"host_value\":{\"alignment\":4,\"byte_strides\":[],\"offset\":8,\
\"profile\":\"host_value\",\"size\":12}}}],\
\"name\":\"fill\",\"physical_layouts\":{\"host_value\":{\"packed_arguments_size\":20,\
\"profile\":\"host_value\"}},\"workgroup_size\":[2,2,1]}],\
\"gpu_launch_abi_version\":1},\"stage\":\"compute\",\"symbol\":\"vernon_test_fill\",\
\"target\":\"cpu\"}},\"target\":\"cpu\",\"type\":\"pipeline\",\
\"variants\":[{\"key\":[],\"outputs\":[{\"access\":\"write\",\"dtype\":\"f32\",\
\"kind\":\"tensor\",\"location\":0,\"name\":\"result\",\"shape\":[12]}],\
\"parameters\":[{\"access\":\"write\",\"element_layout\":{\"alignment\":4,\"byte_size\":4,\
\"layout_hash\":\"cb580e347f23fbe3afbd1c5f72b4d2339b09e33d876f79e9d290445edb43c03b\",\
\"leaves\":[{\"byte_offset\":0,\"dtype\":\"f32\",\"path\":[],\"scalar_count\":1}],\
\"logical_type\":\"f32\"},\"kind\":\"tensor\",\
\"name\":\"output\",\"shape\":[12],\"slot\":0,\"uses\":[{\"access\":\"write\",\
\"dtype\":\"f32\",\"entry\":\"fill\",\"index\":0,\"interface\":\"storage\",\
\"kind\":\"tensor\",\"shape\":[12],\"stage\":\"compute\"}]}],\
\"program\":{\"compute\":\"fill\"}}]}")
string(SHA256 PIPELINE_HASH "${PIPELINE_CANONICAL}")
string(
    SUBSTRING "${PIPELINE_CANONICAL}"
              1
              -1
              PIPELINE_BODY)
file(WRITE "${OUTPUT}/cpu_fill.pipeline.json" "{\"content_hash\":\"${PIPELINE_HASH}\",${PIPELINE_BODY}\n")

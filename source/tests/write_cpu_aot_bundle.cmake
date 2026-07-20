if(NOT DEFINED ARTIFACT OR NOT DEFINED OUTPUT OR NOT DEFINED OPERATING_SYSTEM
   OR NOT DEFINED ARCHITECTURE)
  message(FATAL_ERROR "CPU AOT bundle generator arguments are incomplete")
endif()

file(MAKE_DIRECTORY "${OUTPUT}")
get_filename_component(ARTIFACT_NAME "${ARTIFACT}" NAME)
file(COPY_FILE "${ARTIFACT}" "${OUTPUT}/${ARTIFACT_NAME}" ONLY_IF_DIFFERENT)
file(SIZE "${ARTIFACT}" ARTIFACT_SIZE)
file(SHA256 "${ARTIFACT}" ARTIFACT_SHA256)

file(WRITE "${OUTPUT}/compute.json"
"{
  \"schema_version\": 2,
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
      \"cpu_arguments_size\": 20,
      \"workgroup_size\": [2, 2, 1],
      \"arguments\": [
        {
          \"kind\": \"tensor\",
          \"dtype\": \"f32\",
          \"alignment\": 4,
          \"cpu_offset\": 0,
          \"cpu_size\": 8
        },
        {
          \"kind\": \"builtin\",
          \"builtin\": \"global_invocation_id\",
          \"cpu_offset\": 8,
          \"cpu_size\": 12
        }
      ]
    }]
  }
}
")

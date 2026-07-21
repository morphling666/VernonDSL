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

file(WRITE "${OUTPUT}/pipeline.bundle"
"{
  \"pipeline_bundle_schema_version\": 1,
  \"invocation_abi_version\": 1,
  \"type\": \"vernon_pipeline_bundle\",
  \"id\": \"cpu/fill\",
  \"target\": \"cpu\",
  \"features\": [],
  \"variants\": [{
    \"key\": [],
    \"parameters\": [{
      \"slot\": 0,
      \"name\": \"output\",
      \"kind\": \"tensor\",
      \"dtype\": \"f32\",
      \"shape\": [12],
      \"access\": \"write\",
      \"uses\": [{
        \"stage\": \"compute\",
        \"entry\": \"fill\",
        \"index\": 0,
        \"kind\": \"tensor\",
        \"dtype\": \"f32\",
        \"shape\": [12],
        \"interface\": \"storage\",
        \"access\": \"write\"
      }]
    }],
    \"outputs\": [{
      \"name\": \"result\",
      \"kind\": \"tensor\",
      \"dtype\": \"f32\",
      \"shape\": [12],
      \"access\": \"write\",
      \"location\": 0
    }],
    \"steps\": [{\"kind\": \"dispatch\", \"stage\": \"fill\"}]
  }],
  \"stage_artifacts\": {
    \"fill\": {
      \"id\": \"fill\",
      \"entry\": \"fill\",
      \"stage\": \"compute\",
      \"target\": \"cpu\",
      \"format\": \"native_library\",
      \"artifact\": {
        \"path\": \"${ARTIFACT_NAME}\",
        \"size\": ${ARTIFACT_SIZE},
        \"sha256\": \"${ARTIFACT_SHA256}\"
      },
      \"symbol\": \"vernon_test_fill\",
      \"operating_system\": \"${OPERATING_SYSTEM}\",
      \"architecture\": \"${ARCHITECTURE}\",
      \"cpu_invocation_abi_version\": 1,
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
  }
}
")

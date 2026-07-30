include_guard(GLOBAL)

include("${CMAKE_CURRENT_LIST_DIR}/VernonTargetHelpers.cmake")

function(vernon_configure_dxc target runtime_destination)
    if(NOT WIN32)
        return()
    endif()

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

    if(NOT VERNON_DXC_EXECUTABLE OR NOT EXISTS "${VERNON_DXC_EXECUTABLE}")
        message(WARNING "DXC was not found; the DirectX target will report unavailable")
        return()
    endif()

    file(TO_CMAKE_PATH "${VERNON_DXC_EXECUTABLE}" _vernon_dxc_path)
    target_compile_definitions(${target} PRIVATE VERNON_DXC_EXECUTABLE="${_vernon_dxc_path}")
    get_filename_component(_vernon_dxc_directory "${VERNON_DXC_EXECUTABLE}" DIRECTORY)

    set(_vernon_dxc_runtime_files)
    foreach(_vernon_dxc_file IN ITEMS dxc.exe dxcompiler.dll dxil.dll)
        set(_vernon_dxc_runtime_file "${_vernon_dxc_directory}/${_vernon_dxc_file}")
        if(EXISTS "${_vernon_dxc_runtime_file}")
            list(APPEND _vernon_dxc_runtime_files "${_vernon_dxc_runtime_file}")
            add_custom_command(
                TARGET ${target}
                POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_vernon_dxc_runtime_file}"
                        "$<TARGET_FILE_DIR:${target}>/${_vernon_dxc_file}"
                VERBATIM)
            install(
                FILES "${_vernon_dxc_runtime_file}"
                DESTINATION "${runtime_destination}"
                COMPONENT VernonWheel)
        endif()
    endforeach()
    vernon_register_runtime_files(${target} ${_vernon_dxc_runtime_files})
    message(STATUS "Using DXC: ${VERNON_DXC_EXECUTABLE}")
endfunction()

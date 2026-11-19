cmake_minimum_required(VERSION 3.20)
project(HauntedManor LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# MSYS2 UCRT64 is the reference toolchain; this is only a fallback hint when packages are not found otherwise.
if(WIN32 AND EXISTS "C:/msys64/ucrt64")
    list(APPEND CMAKE_PREFIX_PATH "C:/msys64/ucrt64")
endif()

set(OpenGL_GL_PREFERENCE GLVND)
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(GLEW REQUIRED)
find_package(assimp REQUIRED)
find_package(glm CONFIG QUIET)   # header-only; plain include path works too

file(GLOB SRC CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/*.cpp)
add_executable(HauntedManor ${SRC})

target_include_directories(HauntedManor PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_definitions(HauntedManor PRIVATE PROJECT_ROOT="${CMAKE_SOURCE_DIR}" GLM_ENABLE_EXPERIMENTAL)
target_link_libraries(HauntedManor PRIVATE glfw GLEW::GLEW assimp::assimp OpenGL::GL)
if(TARGET glm::glm)
    target_link_libraries(HauntedManor PRIVATE glm::glm)
endif()
if(WIN32)
    target_link_libraries(HauntedManor PRIVATE winmm)   # Audio uses mciSendString / PlaySound
endif()
if(MINGW)
    target_link_options(HauntedManor PRIVATE -static-libgcc -static-libstdc++)
endif()

# Put every DLL the exe needs next to it so it runs by double-click.
if(MINGW)
    # Imported targets only know their own DLL, not what that DLL pulls in (libstdc++, zlib, minizip...),
    # so walk the real dependency tree with objdump.
    set(_deps_script ${CMAKE_BINARY_DIR}/copy_runtime_deps.cmake)
    file(WRITE ${_deps_script} [=[
if(POLICY CMP0207)
    cmake_policy(SET CMP0207 NEW)   # normalise paths to forward slashes before the regexes run
endif()
file(GET_RUNTIME_DEPENDENCIES
    EXECUTABLES "${EXE}"
    DIRECTORIES ${DIRS}
    RESOLVED_DEPENDENCIES_VAR deps
    UNRESOLVED_DEPENDENCIES_VAR missing
    PRE_EXCLUDE_REGEXES "^api-ms-" "^ext-ms-"
    POST_EXCLUDE_REGEXES "[Ww][Ii][Nn][Dd][Oo][Ww][Ss][/\\][Ss][Yy][Ss][Tt][Ee][Mm]32")
foreach(dll IN LISTS deps)
    file(COPY "${dll}" DESTINATION "${DEST}")
endforeach()
if(missing)
    message(WARNING "Runtime DLLs not found (copy them next to the exe): ${missing}")
endif()
]=])
    get_filename_component(_toolchain_bin "${CMAKE_CXX_COMPILER}" DIRECTORY)
    find_program(HM_OBJDUMP NAMES objdump HINTS "${_toolchain_bin}")
    set(_dll_dirs "${_toolchain_bin};$<TARGET_FILE_DIR:glfw>;$<TARGET_FILE_DIR:GLEW::GLEW>;$<TARGET_FILE_DIR:assimp::assimp>;$ENV{SystemRoot}/System32")
    add_custom_command(TARGET HauntedManor POST_BUILD
        COMMAND ${CMAKE_COMMAND}
            -DCMAKE_GET_RUNTIME_DEPENDENCIES_PLATFORM=windows+pe
            -DCMAKE_GET_RUNTIME_DEPENDENCIES_TOOL=objdump
            "-DCMAKE_GET_RUNTIME_DEPENDENCIES_COMMAND=${HM_OBJDUMP}"
            "-DEXE=$<TARGET_FILE:HauntedManor>"
            "-DDEST=$<TARGET_FILE_DIR:HauntedManor>"
            "-DDIRS=${_dll_dirs}"
            -P ${_deps_script}
        COMMENT "Copying runtime DLLs next to HauntedManor.exe"
        VERBATIM)
elseif(WIN32)
    add_custom_command(TARGET HauntedManor POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_RUNTIME_DLLS:HauntedManor> $<TARGET_FILE_DIR:HauntedManor>
        COMMAND_EXPAND_LISTS)
endif()

cmake_minimum_required(VERSION 3.16)
project(muzero_go CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(MZ_USE_BLAS "Use BLAS for gemm (Accelerate on macOS, OpenBLAS elsewhere)" OFF)

find_package(Threads REQUIRED)

# ---- Library ---------------------------------------------------------------
# Every src/*.cpp belongs to the library, so a task that adds a component adds
# no CMake lines. CONFIGURE_DEPENDS makes `cmake --build` re-glob when files
# appear. The BLAS backend is the one exception: it only compiles against a
# BLAS header, so it is dropped unless MZ_USE_BLAS is on.
file(GLOB MZGO_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
if(NOT MZ_USE_BLAS)
  list(REMOVE_ITEM MZGO_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/backend_blas.cpp)
endif()

add_library(mzgo STATIC ${MZGO_SOURCES})
target_include_directories(mzgo PUBLIC include)
target_compile_options(mzgo PRIVATE -Wall -Wextra)
target_link_libraries(mzgo PUBLIC Threads::Threads)

if(MZ_USE_BLAS)
  if(APPLE)
    # Accelerate ships with macOS. ACCELERATE_NEW_LAPACK selects the current
    # cblas headers instead of the deprecated ones.
    target_link_libraries(mzgo PUBLIC "-framework Accelerate")
    target_compile_definitions(mzgo PUBLIC MZ_USE_BLAS ACCELERATE_NEW_LAPACK)
  else()
    find_package(BLAS REQUIRED)
    find_path(CBLAS_INCLUDE_DIR cblas.h PATH_SUFFIXES openblas)
    if(NOT CBLAS_INCLUDE_DIR)
      message(FATAL_ERROR "MZ_USE_BLAS=ON but cblas.h was not found (install OpenBLAS headers)")
    endif()
    target_include_directories(mzgo PRIVATE ${CBLAS_INCLUDE_DIR})
    target_link_libraries(mzgo PUBLIC ${BLAS_LIBRARIES})
    target_compile_definitions(mzgo PUBLIC MZ_USE_BLAS)
  endif()
endif()

# ---- Tests -----------------------------------------------------------------
# One executable per tests/test_<component>.cpp, named after the file.
# -UNDEBUG: the default build type is Release, which defines NDEBUG and would
# compile every assert() in a test to nothing. Stripping it here keeps the
# tests meaningful without slowing the library down.
# MZGO_SOURCE_DIR lets tests find repository files.
enable_testing()

file(GLOB MZGO_TESTS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_*.cpp)
foreach(test_src ${MZGO_TESTS})
  get_filename_component(test_name ${test_src} NAME_WE)
  add_executable(${test_name} ${test_src})
  target_link_libraries(${test_name} PRIVATE mzgo)
  target_compile_options(${test_name} PRIVATE -UNDEBUG)
  target_compile_definitions(${test_name} PRIVATE MZGO_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
  add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

# Shell tests: tests/*.sh, each run as `bash <file> <build dir> <source dir>`
# and named after the file.
file(GLOB MZGO_SHELL_TESTS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tests/*.sh)
foreach(test_sh ${MZGO_SHELL_TESTS})
  get_filename_component(test_name ${test_sh} NAME_WE)
  add_test(NAME ${test_name} COMMAND bash ${test_sh} ${CMAKE_BINARY_DIR} ${CMAKE_SOURCE_DIR})
endforeach()

# ---- Apps ------------------------------------------------------------------
# One executable per apps/*.cpp, named mzgo_<file>. apps/ is on the include
# path for headers that live next to the apps.
file(GLOB MZGO_APPS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/apps/*.cpp)
foreach(app_src ${MZGO_APPS})
  get_filename_component(app_name ${app_src} NAME_WE)
  add_executable(mzgo_${app_name} ${app_src})
  target_link_libraries(mzgo_${app_name} PRIVATE mzgo)
  target_include_directories(mzgo_${app_name} PRIVATE ${CMAKE_SOURCE_DIR}/apps)
endforeach()

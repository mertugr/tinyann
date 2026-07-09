cmake_minimum_required(VERSION 3.14)
project(tinyann VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(TINYANN_BUILD_CLI "Build tinyann CLI binary" ON)
option(TINYANN_BUILD_TESTS "Build tinyann unit tests" ON)

# Header-only style library target (interface) for consumers; CLI/tests compile against it.
add_library(tinyann INTERFACE)
add_library(tinyann::tinyann ALIAS tinyann)
target_include_directories(tinyann INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Optional AVX2 on x86/x64 (SSE2 is assumed on x86-64; NEON on AArch64 is baseline).
option(TINYANN_ENABLE_AVX2 "Compile with AVX2+FMA distance kernels on x86" ON)
include(CheckCXXCompilerFlag)
if(TINYANN_ENABLE_AVX2)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i.86|x86")
    if(MSVC)
      target_compile_options(tinyann INTERFACE /arch:AVX2)
    else()
      check_cxx_compiler_flag("-mavx2" TINYANN_HAS_MAVX2)
      check_cxx_compiler_flag("-mfma" TINYANN_HAS_MFMA)
      if(TINYANN_HAS_MAVX2)
        target_compile_options(tinyann INTERFACE -mavx2)
      endif()
      if(TINYANN_HAS_MFMA)
        target_compile_options(tinyann INTERFACE -mfma)
      endif()
    endif()
  endif()
endif()

if(TINYANN_BUILD_CLI)
  add_executable(tinyann_cli src/cli.cpp)
  target_link_libraries(tinyann_cli PRIVATE tinyann)
  set_target_properties(tinyann_cli PROPERTIES OUTPUT_NAME tinyann)
  install(TARGETS tinyann_cli RUNTIME DESTINATION bin)
endif()

if(TINYANN_BUILD_TESTS)
  enable_testing()
  add_executable(tinyann_tests tests/test_tinyann.cpp)
  target_link_libraries(tinyann_tests PRIVATE tinyann)
  add_test(NAME tinyann_unit_tests COMMAND tinyann_tests)
endif()

install(DIRECTORY include/tinyann DESTINATION include)

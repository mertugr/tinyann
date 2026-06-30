cmake_minimum_required(VERSION 3.14)
project(tinyann VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Header-only style library target (interface) for consumers; CLI/tests compile against it.
add_library(tinyann INTERFACE)
add_library(tinyann::tinyann ALIAS tinyann)
target_include_directories(tinyann INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# CLI executable
add_executable(tinyann_cli src/cli.cpp)
target_link_libraries(tinyann_cli PRIVATE tinyann)
set_target_properties(tinyann_cli PROPERTIES OUTPUT_NAME tinyann)

# Unit tests
enable_testing()
add_executable(tinyann_tests tests/test_tinyann.cpp)
target_link_libraries(tinyann_tests PRIVATE tinyann)
add_test(NAME tinyann_unit_tests COMMAND tinyann_tests)

install(TARGETS tinyann_cli RUNTIME DESTINATION bin)
install(DIRECTORY include/tinyann DESTINATION include)

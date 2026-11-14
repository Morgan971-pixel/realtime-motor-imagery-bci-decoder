cmake_minimum_required(VERSION 3.25)  # CMakePresets.json version 6
project(mi_core_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(MI_CORE_TESTS "Build the GoogleTest suite" ON)
option(MI_CORE_PYTHON "Build the pybind11 module mi_core_cpp" OFF)
option(MI_CORE_TOOLS "Build the mi_replay command" ON)
option(MI_CORE_BENCH "Build the Google Benchmark micro-benchmarks" OFF)
option(MI_CORE_FUZZ "Build the fuzz targets in fuzz/" OFF)
option(MI_CORE_FUZZ_STANDALONE "Link the fuzz targets with a plain main that runs the files given, for compilers without libFuzzer" OFF)

include(FetchContent)
FetchContent_Declare(nlohmann_json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
  URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d)
FetchContent_MakeAvailable(nlohmann_json)

# One file per contract step; every source in src/ belongs to the library.
file(GLOB MI_CORE_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
add_library(mi_core STATIC ${MI_CORE_SOURCES})
target_include_directories(mi_core PUBLIC include)
target_link_libraries(mi_core PUBLIC nlohmann_json::nlohmann_json)
# -ffp-contract=off: no fused multiply-add, so arithmetic follows the contract's order exactly.
# PUBLIC, so inline header code compiled into tests and bindings obeys it too.
target_compile_options(mi_core PUBLIC -ffp-contract=off PRIVATE -Wall -Wextra -Wpedantic -Werror)
set_target_properties(mi_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(MI_CORE_TOOLS)
  add_executable(mi_replay tools/mi_replay.cpp)
  target_link_libraries(mi_replay PRIVATE mi_core)
  target_compile_options(mi_replay PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

if(MI_CORE_TESTS)
  FetchContent_Declare(googletest
    URL https://github.com/google/googletest/releases/download/v1.15.2/googletest-1.15.2.tar.gz
    URL_HASH SHA256=7b42b4d6ed48810c5362c265a17faebe90dc2373c885e5216439d37927f02926)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  enable_testing()
  file(GLOB MI_CORE_TEST_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_*.cpp)
  add_executable(mi_core_tests ${MI_CORE_TEST_SOURCES})
  target_link_libraries(mi_core_tests PRIVATE mi_core GTest::gtest_main)
  target_compile_definitions(mi_core_tests PRIVATE MI_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../golden")
  target_compile_options(mi_core_tests PRIVATE -Wall -Wextra -Werror)
  include(GoogleTest)
  gtest_discover_tests(mi_core_tests DISCOVERY_MODE PRE_TEST)  # building never runs the test binary
endif()

if(MI_CORE_BENCH)
  FetchContent_Declare(benchmark
    URL https://github.com/google/benchmark/archive/refs/tags/v1.9.1.tar.gz
    URL_HASH SHA256=32131c08ee31eeff2c8968d7e874f3cb648034377dfc32a4c377fa8796d84981)
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(benchmark)
  add_executable(mi_core_bench bench/bench_decoder.cpp)
  target_link_libraries(mi_core_bench PRIVATE mi_core benchmark::benchmark)
  target_compile_definitions(mi_core_bench PRIVATE MI_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../golden")
  target_compile_options(mi_core_bench PRIVATE -Wall -Wextra -Werror)
endif()

if(MI_CORE_FUZZ)
  # Linux clang (preset fuzz): libFuzzer supplies main and the library is built with
  # -fsanitize=fuzzer-no-link,address,undefined. Elsewhere (preset fuzz-seeds) a plain main
  # replays files, so the targets still build and run under UBSan on Apple clang.
  foreach(target model csv)
    add_executable(fuzz_${target} fuzz/fuzz_${target}.cpp)
    target_link_libraries(fuzz_${target} PRIVATE mi_core)
    target_compile_options(fuzz_${target} PRIVATE -Wall -Wextra -Wpedantic -Werror)
    if(MI_CORE_FUZZ_STANDALONE)
      target_sources(fuzz_${target} PRIVATE fuzz/standalone_main.cpp)
    else()
      target_link_options(fuzz_${target} PRIVATE -fsanitize=fuzzer)
    endif()
  endforeach()
endif()

if(MI_CORE_PYTHON)
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(mi_core_cpp python/bindings.cpp)
  target_link_libraries(mi_core_cpp PRIVATE mi_core)
  install(TARGETS mi_core_cpp LIBRARY DESTINATION .)
endif()

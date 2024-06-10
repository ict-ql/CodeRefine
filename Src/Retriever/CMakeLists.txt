cmake_minimum_required(VERSION 3.4)
project(ActionModelTool)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

string(REPLACE "-DNDEBUG" "" CMAKE_CXX_FLAGS_RELEASE
               "${CMAKE_CXX_FLAGS_RELEASE}")

include(sanitizers.cmake)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(LLVM_BUILD_TOOLS ON)

set(KEY_WORD_FILE_PATH ${CMAKE_SOURCE_DIR}/config/transform_key_word)
add_definitions(-DKEY_WORD_FILE_PATH="${KEY_WORD_FILE_PATH}")

add_definitions(${LLVM_DEFINITIONS})
include_directories(${LLVM_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include "Retriever/lib/json" "IR2Vec/src/include")
link_directories(${LLVM_LIBRARY_DIRS})
add_definitions(-fno-rtti -fPIC)
add_definitions(-std=c++14)

add_subdirectory(lib)


include(fmt.cmake)

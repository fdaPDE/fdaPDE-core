cmake_minimum_required(VERSION 3.14)

project(MatrixGenerator CXX)

set(CMAKE_CXX_STANDARD 23)

add_executable(MatrixGenerator src/matrix_generator.cpp)

include(${CMAKE_SOURCE_DIR}/../user_config.cmake)

target_include_directories(MatrixGenerator PRIVATE ${IEIGENDIR})



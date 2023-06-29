cmake_minimum_required(VERSION 3.23)
project(
        libAL
        VERSION 0.1
        LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)

include_directories(include)
include_directories(include/libal)

add_library(${PROJECT_NAME} SHARED
        include/libal/queryAL.hpp
        include/libal/uncertainty.hpp)

target_link_libraries(${PROJECT_NAME} PUBLIC
        -fopenmp
        -larmadillo)

target_include_directories(${PROJECT_NAME} PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include/${PROJECT_NAME}-${PROJECT_VERSION}>)

set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)

add_subdirectory(test)

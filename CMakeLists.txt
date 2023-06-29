cmake_minimum_required(VERSION 3.23)
project(libAL)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)
include_directories(include)
include_directories(include/libal)

add_executable(libAL
        include/libal/queryAL.hpp)

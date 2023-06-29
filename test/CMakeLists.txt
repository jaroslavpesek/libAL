include(FetchContent)
message(STATUS "Fetching googletest library...")
FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG 58d77fa8070e8cec2dc1ed015d66b454c8d78850 # release-1.12.1
)
FetchContent_MakeAvailable(googletest)
enable_testing()

add_executable(libALTest uncert.test.cpp)

target_compile_definitions(
        libALTest
        PUBLIC
        TEST_RESOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)
target_link_libraries(
        libALTest
        PUBLIC GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(libALTest)

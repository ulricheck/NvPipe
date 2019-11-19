
set(TEST_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/test_main.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/test_decode.cpp
        )

add_executable(nvenc_test ${TEST_SOURCES})
set_target_properties(nvenc_test PROPERTIES CXX_STANDARD 17)

target_link_libraries(nvenc_test
        NvPipe
        CONAN_PKG::gtest
        CONAN_PKG::spdlog
        ${NVPIPE_LIBRARIES}
        )

target_include_directories(nvenc_test
        PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_BINARY_DIR}/src
        ${PROJECT_SOURCE_DIR}/src
        )

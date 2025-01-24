cmake_minimum_required(VERSION 3.10)
project(lightglue VERSION 1.0.0)

find_package(OpenCV CONFIG REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Library
add_library(lightglue
    cpp/lightglue.cpp
)

target_include_directories(lightglue
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(lightglue
    PUBLIC
        ${OpenCV_LIBS}
        pybind11::embed
)

# Example
add_executable(example
    examples/main.cpp
)

target_link_libraries(example
    PRIVATE
        lightglue
        ${OpenCV_LIBS}
)
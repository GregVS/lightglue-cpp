cmake_minimum_required(VERSION 3.10)
project(lightglue VERSION 1.0.0)

find_package(OpenCV CONFIG REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

set(Python3_FIND_VIRTUALENV FIRST) 
find_package(Python3 REQUIRED)
execute_process(
    COMMAND ${Python3_EXECUTABLE} -m pip install -r ${CMAKE_SOURCE_DIR}/python/requirements.txt
    RESULT_VARIABLE result
)
if(result)
    message(FATAL_ERROR "Failed to install Python dependencies")
endif()

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
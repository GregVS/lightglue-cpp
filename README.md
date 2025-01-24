# LightGlue C++

C++ binding for LightGlue. It is based on the [LightGlue Python package](https://github.com/cvg/LightGlue).

Note that this runs Python code. The library currently does not have a native C++ implementation. It's useful if you want to quickly integrate LightGlue in a C++ project.

You'll want to create a Python virtual environment and activate it before building.

## Usage
* Clone this repository in your project directory `git submodule add https://github.com/GregVS/lightglue-cpp.git lightglue`
* In your CMakeLists.txt add `add_subdirectory(lightglue)` and `target_link_libraries(your_target lightglue)`
* You'll need OpenCV and pybind11 installed (you can use vcpkg or install to your system).
* At runtime, you'll need to have a Python environment activated with the dependencies listed in `requirements.txt` installed.
* You will also need to add the `./python` directory to your PYTHONPATH environment variable.

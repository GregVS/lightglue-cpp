# LightGlue C++

C++ binding for LightGlue. It is based on the [LightGlue Python package](https://github.com/cvg/LightGlue).

Note that this runs Python code. The library currently does not have a native C++ implementation. It's useful if you want to quickly integrate LightGlue in a C++ project.

You'll want to create a Python virtual environment and activate it before building.

## Usage

* Add this project as a git submodule in your project and mark as a subdirectory for CMake.
* You'll need OpenCV and pybind11 installed (you can use vcpkg or install to your system).
* At runtime, you'll need to have a Python environment activated with the dependencies listed in `requirements.txt` installed.
* You will also need to add the `./python` directory to your PYTHONPATH environment variable.

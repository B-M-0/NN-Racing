cmake_minimum_required(VERSION 3.12)
project(nn_engine)

# Find Python
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Include pybind11
add_subdirectory(extern/pybind11)

# Create the module
# "nn_engine" is what you will call 'import' on in Python
pybind11_add_module(nn_engine 
    src_cpp/matrix.cpp 
    src_cpp/neural_net.cpp 
    src_cpp/bindings.cpp
)
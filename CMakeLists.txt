cmake_minimum_required(VERSION 3.12)
project(nn_engine)

# Find Python
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Include pybind11
add_subdirectory(extern/pybind11)

# Create the module
# "nn_engine" --> 'import' in Python
pybind11_add_module(nn_engine 
    src_cpp/matrix.cpp 
    src_cpp/neural_net.cpp 
    src_cpp/binding.cpp
)

# C++ Test Runner
add_executable(test_runner 
    src_cpp/test.cpp 
    src_cpp/matrix.cpp 
    src_cpp/neural_net.cpp
)
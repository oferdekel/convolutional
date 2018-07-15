cmake_minimum_required(VERSION 3.5)

project(convolutional)
set(target_name convolutional)

set(include
    include/BlasHelpers.h
    include/Tensor.h
)

set(src
    src/BlasHelpers.cpp
    src/Main.cpp
    src/Matrix.cpp
)

source_group("src" FILES ${src})
source_group("include" FILES ${include})

# Turn on ability to create folders to organize projects
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Set C++ version
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Apply -fPIC where applicable to the platform
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# create executable in build\bin
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(${target_name} ${src} ${include})
target_include_directories(${target_name} PRIVATE include)


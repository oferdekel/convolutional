cmake_minimum_required(VERSION 3.5)

project(convolutional)
set(target_name convolutional)

set(include
    include/BlasHelpers.h
    include/ConvolutionProperties.h
    include/CSVParser.h
    include/ForLoopConvolution.h
    include/PartiallyUnrolledInputExplicitOutPaddingConvolution.h
    include/PartiallyUnrolledInputExplicitPaddingConvolution.h
    include/PartiallyUnrolledInputImplicitInPaddingConvolution.h
    include/Tensor.h
    include/TestHelpers.h
    include/UnrolledInputChlMajInputConvolution.h
    include/UnrolledInputConvolution.h
    include/UnrolledInputExplicitOutPaddingConvolution.h
    include/UnrolledInputExplicitPaddingConvolution.h
    include/UnrolledInputImplicitInPaddingConvolution.h
    include/UnrolledOutputConvolution.h
)

set(src
    src/BlasHelpers.cpp
    src/Main.cpp
)

source_group("src" FILES ${src})
source_group("include" FILES ${include})

# Turn on ability to create folders to organize projects
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Setup Blas
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake")
include(OpenBLASSetup)
find_package(BLAS)
if(BLAS_FOUND)
    message(STATUS "Blas libraries: ${BLAS_LIBRARIES}")
    message(STATUS "Blas linker flags: ${BLAS_LINKER_FLAGS}")
    message(STATUS "Blas vendor: ${BLA_VENDOR}")
endif()

# Set C++ version
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compilaiton flags
if(MSVC)
  add_compile_options(/O2)
  add_compile_options(/Ob2)
  add_compile_options(/Ot)
else()
  add_compile_options(-O3)
endif()

# create executable in build\bin
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# message
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

add_executable(${target_name} ${src} ${include})
target_include_directories(${target_name} PRIVATE include)


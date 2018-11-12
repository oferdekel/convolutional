cmake_minimum_required(VERSION 3.5)

project(convolutional)
set(target_name convolutional)

# files
set(include
    include/BlasHelpers.h
    include/ConvProperties.h
    include/CSVParser.h
    include/ForLoopConv.h
    include/PartiallyUnrolledInputExplicitOutPaddingConv.h
    include/PartiallyUnrolledInputExplicitPaddingConv.h
    include/PartiallyUnrolledInputImplicitInPaddingConv.h
    include/Tensor.h
    include/TestHelpers.h
    include/UnrolledInputConv_cI.h
    include/UnrolledInputConv_rI.h
    include/UnrolledInputExplicitOutPaddingConv.h
    include/UnrolledInputExplicitPaddingConv.h
    include/UnrolledInputImplicitInPaddingConv.h
    include/UnrolledOutputConv.h
)

set(src
    src/BlasHelpers.cpp
    src/Main.cpp
)

source_group("src" FILES ${src})
source_group("include" FILES ${include})
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Set C++ version
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# create executable in build\bin
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin)
add_executable(${target_name} ${src} ${include})
target_include_directories(${target_name} PRIVATE include)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}")
include(BlasConfig)
if(USE_BLAS)
    if(BLAS_COPY_DLLS)
        foreach(DLL ${BLAS_COPY_DLLS})
            add_custom_command(TARGET ${target_name} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different ${DLL} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE}
            )
        endforeach()
    endif()

    add_definitions(-DUSE_BLAS)
    add_definitions(-DBLAS_HEADER_FILE="${BLAS_HEADER_FILE}")
    add_definitions(-DBLAS_VERSION="${BLAS_VERSION}")
    target_include_directories(${target_name} SYSTEM PUBLIC ${BLAS_INCLUDE_DIRS})
    target_link_libraries(${target_name} ${BLAS_LIBRARIES})
endif()


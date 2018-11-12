# Convolutional
Reference implementations of 2-dimensional tensor convolution

## Requirements
* CMake 3.5 or newer
* On Windows: Visual C++ 2017
* On Linux: gcc 5.4 or newer

## Setting paths to BLAS
This code does not automatically detect the path to your computer's BLAS implementation. This allows you to explicitly choose which BLAS implementation to use. Before building, edit the file `BlasConfig.cmake` and set the paths manually. For example, for Intel MKL BLAS on Windows 10, the BLAS configuration looks like this:

```
if(${BLAS_VERSION} MATCHES "winmkl") 
    set(INTEL_ROOT "/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows")
    set(BLAS_COPY_DLLS 
        ${INTEL_ROOT}/redist/intel64/mkl/mkl_rt.dll
        ${INTEL_ROOT}/redist/intel64/mkl/mkl_intel_thread.dll
        ${INTEL_ROOT}/redist/intel64/mkl/mkl_core.dll
        ${INTEL_ROOT}/redist/intel64/mkl/mkl_avx2.dll
        ${INTEL_ROOT}/redist/intel64/compiler/libiomp5md.dll
    )
    set(BLAS_HEADER_FILE "mkl_cblas.h")
    set(BLAS_INCLUDE_DIRS ${INTEL_ROOT}/mkl/include)
    set(BLAS_LIBRARIES ${INTEL_ROOT}/mkl/lib/intel64/mkl_rt.lib)
    set(USE_BLAS TRUE)
endif()
```

The relevant CMake variables that need to be set are:
* `BLAS_COPY_DLLS` - a list of DLLs that are copied into the executable directory. 
* `BLAS_HEADER_FILE` - the name of the BLAS header file to include in the code.
* `BLAS_LIBRARIES` - a list of libraries to provide to the linker.
* `USE_BLAS` - must be set to `true`, otherwise BLAS is not used and the results are very slow and meaningless.

## Build and execute on Windows

After cloning the repository, `cd` into the main repository directory, create a new directory named `build` and `cd` into that directory. Next, type the command
```
cmake -DBLAS_VERSION=winmkl -G "Visual Studio 15 2017 Win64" ..
```
where the `BLAS_VERSION` parameter (set to `winmkl` in the example above) matchs the configuration that you defined in `BlasConfig.cmake` (see instructions above). Finally, to build the executable, type `cmake --build . --config Release`. The new executable will appear as `\build\bin\convolutional.exe`. These instructions are summarized in `build.cmd`.

To run the test, `cd` back to the main project directory and type `build\bin\convolutional.exe benchmarks.csv`. Edit `benchmarks.csv` to control the filter and output shapes used in the test. 

## Build and execute on Linux
After cloning the repository, `cd` into the main repository directory, create a new directory named `build` and `cd` into that directory. Next, type the command
```
cmake -DBLAS_VERSION=ubuntuopenblas ..
```
where the `BLAS_VERSION` parameter (set to `ubuntuopenblas` in the example above) matchs the configuration that you defined in `BlasConfig.cmake` (see instructions above). Finally, to build the executable, type `cmake --build . --config Release`. The new executable will appear as `build/convolutional`. These instructions are summarized in `build.sh`.

To run the test, `cd` back to the main project directory and type `build/convolutional.exe benchmarks.csv`. Edit `benchmarks.csv` to control the filter and output shapes used in the test. 


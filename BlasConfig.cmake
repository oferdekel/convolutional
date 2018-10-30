
if(${BLAS_VERSION} MATCHES "winmkl") 
    set(BLAS_HEADER_FILE "mkl_cblas.h")
    set(BLAS_INCLUDE_DIRS "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows/mkl/include")
    set(BLAS_LIBRARIES "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows/mkl/lib/intel64_win/mkl_rt.lib")
    set(USE_BLAS TRUE)
endif()

if(${BLAS_VERSION} MATCHES "winopenblas") 
    set(BLAS_COPY_DLLS 
        "C:/xianyi-OpenBLAS-fd8d186/build/lib/openblas.dll"
        /ProgramData/Miniconda3/Library/bin/flang.dll
        /ProgramData/Miniconda3/Library/bin/libomp.dll
    )
    set(BLAS_HEADER_FILE "cblas.h")
    set(BLAS_INCLUDE_DIRS 
        /xianyi-OpenBLAS-fd8d186
        /xianyi-OpenBLAS-fd8d186/build)
    set(BLAS_LIBRARIES "C:/xianyi-OpenBLAS-fd8d186/build/lib/Release/openblas.lib")
    add_definitions(-DC_MSVC)
    set(USE_BLAS TRUE)
endif()

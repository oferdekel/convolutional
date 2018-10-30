if(${BLAS_VERSION} MATCHES "winmkl") 
    set(BLAS_COPY_DLLS 
        "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows/redist/intel64/mkl/mkl_rt.dll"
        "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows/redist/intel64/mkl/mkl_intel_thread.dll"
        "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows/redist/intel64/mkl/mkl_core.dll"
        "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows/redist/intel64/mkl/mkl_avx2.dll"
        "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows/redist/intel64/compiler/libiomp5md.dll"
    )
    set(BLAS_HEADER_FILE "mkl_cblas.h")
    set(BLAS_INCLUDE_DIRS "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows/mkl/include")
    set(BLAS_LIBRARIES "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows/mkl/lib/intel64/mkl_rt.lib")
    set(USE_BLAS TRUE)
endif()

if(${BLAS_VERSION} MATCHES "winopenblas") 
    set(BLAS_COPY_DLLS "C:/OpenBLAS/bin/libopenblas.dll")
    set(BLAS_HEADER_FILE "cblas.h")
    set(BLAS_INCLUDE_DIRS "C:/OpenBLAS/include")
    set(BLAS_LIBRARIES "C:/OpenBLAS/lib/libopenblas.dll.a")
    add_definitions(-DC_MSVC)
    set(USE_BLAS TRUE)
endif()

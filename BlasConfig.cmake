
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

if(${BLAS_VERSION} MATCHES "winopenblas") 
    set(OPENBLAS_ROOT /xianyi-OpenBLAS-fd8d186)
    set(BLAS_COPY_DLLS 
        ${OPENBLAS_ROOT}/build/lib/openblas.dll
        /ProgramData/Miniconda3/Library/bin/flang.dll
        /ProgramData/Miniconda3/Library/bin/libomp.dll
    )
    set(BLAS_HEADER_FILE "cblas.h")
    set(BLAS_INCLUDE_DIRS 
        ${OPENBLAS_ROOT}
        ${OPENBLAS_ROOT}/build)
    set(BLAS_LIBRARIES ${OPENBLAS_ROOT}/build/lib/Release/openblas.lib)
    add_definitions(-DC_MSVC)
    set(USE_BLAS TRUE)
endif()

if(${BLAS_VERSION} MATCHES "winopenblas219") 
    set(OPENBLAS_ROOT /OpenBLASLibs.0.2.19.3/build/native/x64/haswell)   
    set(BLAS_COPY_DLLS 
        ${OPENBLAS_ROOT}/bin/libopenblas.dll
        ${OPENBLAS_ROOT}/bin/libgfortran-3.dll
        ${OPENBLAS_ROOT}/bin/libquadmath-0.dll
        ${OPENBLAS_ROOT}/bin/libgcc_s_seh-1.dll
    )
    set(BLAS_HEADER_FILE "cblas.h")
    set(BLAS_INCLUDE_DIRS ${OPENBLAS_ROOT}/include)
    set(BLAS_LIBRARIES ${OPENBLAS_ROOT}/lib/libopenblas.dll.a)
    add_definitions(-DC_MSVC)
    set(USE_BLAS TRUE)
endif()

if(${BLAS_VERSION} MATCHES "ubuntuopenblas") 
    set(BLAS_HEADER_FILE "cblas.h")
    set(BLAS_INCLUDE_DIRS /usr/include/openblas)
    set(BLAS_LIBRARIES /usr/lib/libopenblas.so)
    set(USE_BLAS TRUE)
endif()

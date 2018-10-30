mkdir build_gcc
cd build_gcc
cmake -DBLAS_VERSION=ubuntuopenblas ..
cmake --build .
cd ..

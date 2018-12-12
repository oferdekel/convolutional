mkdir build_gcc
cd build_gcc
cmake -DCMAKE_BUILD_TYPE=Release -DBLAS_VERSION=ubuntuopenblas ..
cmake --build .
cd ..

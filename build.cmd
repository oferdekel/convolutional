if not exist build_vs mkdir build_vs
cd build_vs
cmake -DBLAS_VERSION=winopenblas -G "Visual Studio 15 2017 Win64" ..
cmake --build . --config Release 
cd ..

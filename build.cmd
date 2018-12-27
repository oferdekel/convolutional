if not exist build_vs mkdir build_vs
cd build_vs
cmake -DBLAS_VERSION=winmkl -G "Visual Studio 15 2017 Win64" ..
cmake --build . --config Release
cmake --build . --config Debug
cd ..

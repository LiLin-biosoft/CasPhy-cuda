mkdir build
cd build

cmake ..
cmake -DCUDA_ARCH=75 ..
make
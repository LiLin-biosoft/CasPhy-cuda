# CUDA code for CasPhy
Li Lin<br>

## Requirements:
* CMake v3.13+
* C++ 11 complier
* CUDA Toolkit

## installation:
```
git clone https://github.com/LiLin-biosoft/CasPhy-cuda.git
cd CasPhy-cuda
mkdir build
cd build
cmake ..
make
```
or specify the compute capacity for your NVIDIA GPU:<br>
```
git clone https://github.com/LiLin-biosoft/CasPhy-cuda.git
cd CasPhy-cuda
mkdir build
cd build
cmake -DCUDA_ARCH=89 .. ## set compute capacity to 89 for RTX 40 series
make
```
## usage:
```
cd ..
./build/fasta2bin example/test.fasta test.bin test_names.txt test_positions.txt
```
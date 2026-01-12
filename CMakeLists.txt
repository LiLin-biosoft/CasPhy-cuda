cmake_minimum_required(VERSION 3.13)
project(CasPhy-cuda LANGUAGES CXX CUDA)

set(CUDA_ARCH "native" CACHE STRING 
    "CUDA architecture: 'native' for auto-detect, or specific arch (e.g., 52, 60, 70, 75, 80, 86)")

set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})

if(CUDA_ARCH STREQUAL "native")
    message(STATUS "CUDA: Auto-detecting architecture for current GPU")
else()
    message(STATUS "CUDA: Targeting architecture(s): ${CUDA_ARCH}")
endif()

add_executable(fasta2bin fasta2bin.cpp)
add_executable(joinEdgeBitReverse joinEdgeBitReverse.cu)
add_executable(joinEdgeIndel joinEdgeIndel.cu)
add_executable(nodeKeepBit nodeKeepBit.cu)
add_executable(nodeKeepIndel nodeKeepIndel.cu)
add_executable(radiusNeighbor radiusNeighbor.cu)
add_executable(radiusNeighborDNA radiusNeighborDNA.cu)
add_executable(tipMerge tipMerge.cu)
add_executable(vootRoot2 vootRoot2.cu)


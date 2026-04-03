cmake_minimum_required(VERSION 3.18)
project(gpu_topo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(CUDAToolkit REQUIRED)

if(NOT TARGET CUDA::nvml)
    message(FATAL_ERROR
        "CUDAToolkit found but CUDA::nvml target is missing.\n"
        "Ensure the CUDA Toolkit provides nvidia-ml (libnvidia-ml).")
endif()

add_executable(gpu_topo local_nvml.cpp)

target_link_libraries(gpu_topo PRIVATE
    MPI::MPI_CXX
    CUDA::cudart
    CUDA::nvml
)

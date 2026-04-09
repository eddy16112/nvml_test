cmake_minimum_required(VERSION 3.18)
project(gpu_topo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(HWLOC REQUIRED IMPORTED_TARGET hwloc)

if(NOT TARGET CUDA::nvml)
    message(FATAL_ERROR
        "CUDAToolkit found but CUDA::nvml target is missing.\n"
        "Ensure the CUDA Toolkit provides nvidia-ml (libnvidia-ml).")
endif()

if(NOT TARGET CUDA::cuda_driver)
    message(FATAL_ERROR
        "CUDAToolkit found but CUDA::cuda_driver target is missing.\n"
        "cuDeviceGet / cuInit require libcuda (driver API).")
endif()

add_executable(gpu_topo main.cpp machine_manager.cpp processor.cpp pal.cpp topology_node.cpp)

target_link_libraries(gpu_topo PRIVATE
    MPI::MPI_CXX
    CUDA::cuda_driver
    CUDA::nvml
    PkgConfig::HWLOC
)

# Builds the two GPU programs and installs them into the infernal_gpu package (pip install .).
# CUDA architectures: the local GPU by default; override with e.g.
#   CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=80;86;89" pip install .
cmake_minimum_required(VERSION 3.24)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()
project(infernal_gpu LANGUAGES CXX CUDA)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)
find_package(Threads REQUIRED)

set(ENGINE_SOURCES src/cyk_kernel.cu src/inside_kernel.cu src/gpu_transfer.cu src/gpu_batch_scan.cu src/scan.cu)
set(ENGINE_DEFS IL_PARALLEL_SCAN IL_ZERO_EMISSION_SCAN BIF_RING_INDEX ROOT_STATE_FUSED INSIDE_MIN_BLOCKS_256=4)

function(engine_program name)
  add_executable(${name} ${ENGINE_SOURCES})
  target_include_directories(${name} PRIVATE src)
  target_compile_definitions(${name} PRIVATE ${ENGINE_DEFS} ${ARGN})
  target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O2 -Xcompiler=-O2>)
  set_target_properties(${name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  target_link_libraries(${name} PRIVATE Threads::Threads)
  install(TARGETS ${name} RUNTIME DESTINATION infernal_gpu/bin)
endfunction()

engine_program(infernal-gpu-scan)                          # search
engine_program(infernal-gpu-calibrate-scan CALIB_RECORDS)  # calibration (record emission)

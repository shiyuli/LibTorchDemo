# CMakeList.txt : CMake project for LibTorchDemo, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.0 FATAL_ERROR)

set(LIBTORCH_DIR "D:/LIB/libtorch")
find_package(Torch REQUIRED PATHS "${LIBTORCH_DIR}/share/cmake/Torch")

# Add source to this project's executable.
add_executable (LibTorchDemo "LibTorchDemo.cpp" "LibTorchDemo.h" "ModelManager.cpp" "ModelManager.h")

target_link_libraries(LibTorchDemo "${TORCH_LIBRARIES}")

set_property(TARGET LibTorchDemo PROPERTY CXX_STANDARD 11)

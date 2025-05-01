cmake_minimum_required(VERSION 4.0)
set(CMAKE_CXX_STANDARD 11)


# cuda setup
find_package(CUDA REQUIRED)
if(NOT DEFINED CMAKE_CUDA_STANDARD)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)    
endif()
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}) 
link_directories(${CMAKE_CUDA_INCLUDE_DIRS})

project(cudaAtScaleFinalAssignment LANGUAGES CUDA CXX)

# define source code
set(SRCS
    main.cpp
)
set(INCLUDES
    CLI11.hpp
)

# create cmake target
add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME}
    ${CUDA_LIBRARIES}
    nppc
    nppi
)

# Optional: CUDA-Code mit nvcc kompilieren
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
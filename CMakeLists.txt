cmake_minimum_required(VERSION 3.20.2)

project(spd_K)

include_directories(${LIBRARY_PATH})

find_package(MPI)
if(MPI_CXX_FOUND)
  message(STATUS "MPI support found")
  message(STATUS "MPI compile flags: " ${MPI_CXX_COMPILE_FLAGS})
  message(STATUS "MPI include path: " ${MPI_CXX_INCLUDE_PATH})
  message(STATUS "MPI LINK flags path: " ${MPI_CXX_LINK_FLAGS})
  message(STATUS "MPI libraries: " ${MPI_CXX_LIBRARIES})
  include_directories(${MPI_CXX_INCLUDE_PATH})
endif()

find_package(HDF5 COMPONENTS CXX)
if(HDF5_FOUND)
  message(STATUS "HDF5 support found")
  message(STATUS "HDF5 include path: " ${HDF5_INCLUDE_DIR})
  message(STATUS "HDF5 libraries: " ${HDF5_LIBRARIES})
  include_directories(${HDF5_INCLUDE_DIR})
endif()

include(FetchContent)
FetchContent_Declare(
  Kokkos
  GIT_REPOSITORY https://github.com/kokkos/kokkos.git
  GIT_TAG master
)
FetchContent_MakeAvailable(Kokkos)

add_executable(${PROJECT_NAME} 
    src/global.cpp
    src/main.cpp
    src/output.cpp
    src/polynomials.cpp
    src/initial_conditions.cpp
    src/transforms.cpp
    src/boundary.cpp
    src/hydro.cpp
    src/induction.cpp
)


target_link_libraries(${PROJECT_NAME} ${MPI_CXX_LIBRARIES} ${HDF5_LIBRARIES} Kokkos::kokkos)
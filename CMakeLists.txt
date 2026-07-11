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
  GIT_TAG 5.1.1
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(Kokkos)

#Extra compile definitions for validation builds, e.g.
#  cmake -DSPD_DEFINES="REF_TRANSFORMS" (see src/define.hpp)
set(SPD_DEFINES "" CACHE STRING "Semicolon-separated compile definitions for spd_K targets")

add_executable(${PROJECT_NAME}
    src/global.cpp
    src/main.cpp
    src/parameter_input.cpp
    src/output.cpp
    src/polynomials.cpp
    src/initial_conditions.cpp
    src/transforms.cpp
    src/boundary.cpp
    src/hydro.cpp
    src/induction.cpp
    src/trouble_detection.cpp
    src/muscl.cpp
)
if(SPD_DEFINES)
  target_compile_definitions(${PROJECT_NAME} PRIVATE ${SPD_DEFINES})
endif()


target_link_libraries(${PROJECT_NAME} ${MPI_CXX_LIBRARIES} Kokkos::kokkos)
if(HDF5_FOUND)
  target_link_libraries(${PROJECT_NAME} ${HDF5_LIBRARIES})
endif()

add_executable(spd_K_test
    src/global.cpp
    src/parameter_input.cpp
    src/test_transforms.cpp
    src/output.cpp
    src/polynomials.cpp
    src/initial_conditions.cpp
    src/transforms.cpp
    src/boundary.cpp
    src/hydro.cpp
    src/induction.cpp
    src/trouble_detection.cpp
    src/muscl.cpp
)
if(SPD_DEFINES)
  target_compile_definitions(spd_K_test PRIVATE ${SPD_DEFINES})
endif()
target_link_libraries(spd_K_test ${MPI_CXX_LIBRARIES} Kokkos::kokkos)
if(HDF5_FOUND)
  target_link_libraries(spd_K_test ${HDF5_LIBRARIES})
endif()

enable_testing()
add_test(NAME transforms_unit COMMAND spd_K_test)
find_package(Python3 COMPONENTS Interpreter)
if(Python3_FOUND)
  add_test(NAME regression
           COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/run_tests.py
                   --build-dir ${CMAKE_BINARY_DIR} --skip-unit)
  set_tests_properties(regression PROPERTIES TIMEOUT 3600)
endif()
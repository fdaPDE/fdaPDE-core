cmake_minimum_required(VERSION 3.14)

project(MUMPS_TESTING CXX Fortran)

set(CMAKE_CXX_STANDARD 23)

set(CMAKE_CXX_FLAGS "-O -fopenmp -Wno-deprecated-declarations")
set(CMAKE_Fortran_FLAGS "-O -fopenmp")

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Fetch the user defined directories
include(${CMAKE_SOURCE_DIR}/user_config.cmake)

# Define library directories
set(IEIGENDIR "${EIGENDIR}")
set(LPORDDIR "${MUMPSDIR}/PORD/lib")
set(IDMUMPSDIR "${MUMPSDIR}/include")
set(LDMUMPSDIR "${MUMPSDIR}/lib")

# Define library variables
set(LIBS_DMUMPS "dmumps" "mumps_common")
set(LIBS_SCOTCH "ptesmumps" "ptscotch" "ptscotcherr")
set(LIBS_METIS "parmetis" "metis")
set(LIBS_PORD "pord")
set(LIBS_LAPACK "lapack")
set(LIBS_SCALAPACK "scalapack-openmpi")
set(LIBS_BLAS "blas")
set(LIBS_OTHERS "pthread" "stdc++")

add_executable(gen_data_rand src/gen_data_rand.cpp)
set_target_properties(gen_data_rand PROPERTIES LINKER_LANGUAGE Fortran)

target_include_directories(gen_data_rand PRIVATE
    ${IEIGENDIR}
    ${IDMUMPSDIR}
    ${MPI_INCLUDE_PATH}
)

target_link_directories(gen_data_rand PRIVATE
    ${LSCOTCHDIR}
    ${LPORDDIR}
    ${LMETISDIR}
    ${LDMUMPSDIR}
)

target_link_libraries(gen_data_rand PRIVATE
    ${LIBS_DMUMPS}
    ${LIBS_SCOTCH}
    ${LIBS_METIS}
    ${LIBS_PORD}
    ${LIBS_LAPACK}
    ${LIBS_SCALAPACK}
    ${LIBS_BLAS}
    ${LIBS_OTHERS}
    MPI::MPI_Fortran
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    gtest_main
)

add_executable(gen_data_mesh src/gen_data_mesh.cpp)
set_target_properties(gen_data_mesh PROPERTIES LINKER_LANGUAGE Fortran)

target_include_directories(gen_data_mesh PRIVATE
    ${IEIGENDIR}
    ${IDMUMPSDIR}
    ${MPI_INCLUDE_PATH}
)

target_link_directories(gen_data_mesh PRIVATE
    ${LSCOTCHDIR}
    ${LPORDDIR}
    ${LMETISDIR}
    ${LDMUMPSDIR}
)

target_link_libraries(gen_data_mesh PRIVATE
    ${LIBS_DMUMPS}
    ${LIBS_SCOTCH}
    ${LIBS_METIS}
    ${LIBS_PORD}
    ${LIBS_LAPACK}
    ${LIBS_SCALAPACK}
    ${LIBS_BLAS}
    ${LIBS_OTHERS}
    MPI::MPI_Fortran
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    gtest_main
)

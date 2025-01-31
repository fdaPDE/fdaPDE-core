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

# enable testing functionality using gtest
include(FetchContent)
FetchContent_Declare(
  googletest
  # version 1.14.0
  URL https://github.com/google/googletest/archive/f8d7d77c06936315286eb55f8de22cd23c188571.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(mumps_fdapde_test main.cpp)
set_target_properties(mumps_fdapde_test PROPERTIES LINKER_LANGUAGE Fortran)

target_include_directories(mumps_fdapde_test PRIVATE
    ${IEIGENDIR}
    ${IDMUMPSDIR}
    ${MPI_INCLUDE_PATH}
)

target_link_directories(mumps_fdapde_test PRIVATE
    ${LSCOTCHDIR}
    ${LPORDDIR}
    ${LMETISDIR}
    ${LDMUMPSDIR}
)

target_link_libraries(mumps_fdapde_test PRIVATE
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

add_test(NAME mumps_fdapde_test
	 COMMAND $<TARGET_FILE:mumps_fdapde_test>
  )

include(GoogleTest)
gtest_discover_tests(mumps_fdapde_test)

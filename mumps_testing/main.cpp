#include <gtest/gtest.h>

#include "src/MumpsLU_test.cpp"
#include "src/MumpsLDLT_test.cpp"
#include "src/MumpsBLR_test.cpp"
#include "src/MumpsRR_test.cpp"
#include "src/MumpsSchur_test.cpp"
// #include "src/Mumps_set_vs_vec.cpp"

int main (int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

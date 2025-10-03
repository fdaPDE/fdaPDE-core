#ifndef RAND_INDICES_H
#define RAND_INDICES_H

#include <random>
#include <set>
#include <vector>

constexpr bool Seeded = true;
constexpr int Seed = 42;

inline void randomInvIndices(std::set<std::pair<int, int>>& set, int rows) {
    std::mt19937 gen;
    if (Seeded) {
        gen.seed(Seed);
    } else {
        std::random_device rd;
        gen.seed(rd());
    }
    std::uniform_int_distribution<> dis(0, rows - 1);

    // Determine the random number of elements to add to the set (between 1 and rows/10)
    std::uniform_int_distribution<> sizeDis(1, rows / 10);
    int numElements = sizeDis(gen);

    // Generate random indices
    std::set<std::pair<int, int>> uniqueNumbers;
    while (uniqueNumbers.size() < numElements) { uniqueNumbers.insert({dis(gen), dis(gen)}); };

    set = uniqueNumbers;
}

inline void randomInvIndices(std::vector<std::pair<int, int>>& vec, int rows){
    std::set<std::pair<int, int>> set;
    randomInvIndices(set, rows);
    vec.assign(set.begin(), set.end());

    // scrambe the vector
    std::mt19937 gen;
    if (Seeded) {
        gen.seed(Seed);
    } else {
        std::random_device rd;
        gen.seed(rd());
    }
    std::shuffle(vec.begin(), vec.end(), gen);
}

#endif   // RAND_INDICES_H

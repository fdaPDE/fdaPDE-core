#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>

int main(){
    // load fdaPDE solution
    std::vector<double> fdaPDE_solution;
    std::ifstream file("../test/build/solution_nonlinear_P1.txt");
    double number;
    while (file >> number) {
        fdaPDE_solution.push_back(number);
    }
    file.close();

    // load freefem solution
    std::vector<double> freefem_solution;
    freefem_solution.reserve(fdaPDE_solution.size());
    file.open("freefem_P1.txt");
    while (file >> number) {
        freefem_solution.push_back(number);
    }
    file.close();

    // check if solutions have the same length
    if (freefem_solution.size() != fdaPDE_solution.size()){
        std::cout << "The two solutions have different lengths: " << freefem_solution.size() << "!=" << fdaPDE_solution.size() << std::endl;
        return 0;
    }

    // check L2 norm of the two solutions
    std::vector<double> difference;
    difference.reserve(fdaPDE_solution.size());
    for (std::size_t i = 0; i < freefem_solution.size(); i++){
        difference.push_back(freefem_solution[i] - fdaPDE_solution[i]);
    }

    // compute the norm of the difference
    double norm = 0;
    for (std::size_t i = 0; i < difference.size(); i++){
        norm += difference[i] * difference[i];
    }
    norm = std::sqrt(norm);
    std::cout << "The norm of the difference is " << norm << std::endl;

    return 0;
}
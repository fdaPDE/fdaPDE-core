#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>


int main(){
    // load fdaPDE solution
    int n = 10;
    std::vector<double> x0 = {0.01, 0.01};

    for (int i=0; i<n; ++i){
        std::vector<double> fdaPDE_solution;
        std::vector<double> x(2);
        x[0] = x0[0] + 0.1*i;
        x[1] = x0[1] + 0.1*1 + 0.1*i; 
        // std::ifstream file("../test/build/BroydenArmijo_(" + std::to_string(x[0]) + "," + std::to_string(x[1]) + ")");
        std::ifstream file("../test/build/Broyden_(" + std::to_string(x[0]) + "," + std::to_string(x[1]) + ")");
        double number;
        while (file >> number) {
            fdaPDE_solution.push_back(number);
        }
        file.close();

        // load freefem solution
        std::vector<double> matlab_solution;
        matlab_solution.reserve(fdaPDE_solution.size());
        // file.open("BroydenArmijo_matlab_(" + std::to_string(x[0]) + "," + std::to_string(x[1]) + ").txt");
        file.open("Broyden_matlab_(" + std::to_string(x[0]) + "," + std::to_string(x[1]) + ").txt");
        while (file >> number) {
            matlab_solution.push_back(number);
        }
        file.close();

        // check if solutions have the same length
        if (matlab_solution.size() != fdaPDE_solution.size()){
            std::cout << "The two solutions have different lengths: " << matlab_solution.size() << "!=" << fdaPDE_solution.size() << std::endl;
        }

        // check L2 norm of the two solutions
        std::vector<double> difference;
        int min_lenght = std::min(fdaPDE_solution.size(), matlab_solution.size());
        difference.reserve(min_lenght);
        for (int i = 0; i < min_lenght; i++){
            difference.push_back(matlab_solution[i] - fdaPDE_solution[i]);
        }

        // compute the norm of the difference
        double norm = 0;
        for (std::size_t i = 0; i < difference.size(); i++){
            norm += difference[i] * difference[i];
        }
        norm = std::sqrt(norm);
        std::cout << "The norm of the difference is " << std::setprecision(15) << norm << std::endl;
    }

    return 0;
}
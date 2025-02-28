#include <fstream>
#include <nlohmann/json.hpp> 
using json = nlohmann::json;
#include <geometry.h>
using namespace fdapde;
#include "domains.h"

int main() {

    //-----------------------TEST OF COMPUTATIONAL EFFICIENCY OF CONFLICT GRAPH ALGORITHM------------------------------

    std::vector<Eigen::Matrix<double, Eigen::Dynamic, 2>> domain={rectangle};

    std::ofstream outfile("Meshes/Delaunay/timing_conflict.csv");
    outfile << "NumPoints,TimeElapsed(ms)\n";

    auto start = high_resolution_clock::now(); 
    Delaunay<2, 2> del_10000(domain, 10000);
    auto end = high_resolution_clock::now();  

    auto duration = duration_cast<milliseconds>(end - start).count();
    std::cout << "elapsed time for 10000 internal points: " << duration << " ms" << std::endl;
    outfile << 10000 << "," << duration << "\n"; 

    start = high_resolution_clock::now(); 
    Delaunay<2, 2> del_100000(domain, 100000);  
    end = high_resolution_clock::now(); 

    duration = duration_cast<milliseconds>(end - start).count();
    std::cout << "elapsed time for 100000 internal points: " << duration << " ms" << std::endl;
    outfile << 100000 << "," << duration << "\n"; 

    start = high_resolution_clock::now(); 
    Delaunay<2, 2> del_1000000(domain, 1000000);  
    end = high_resolution_clock::now(); 

    duration = duration_cast<milliseconds>(end - start).count();
    std::cout << "elapsed time for 1000000 internal points: " << duration << " ms" << std::endl;
    outfile << 1000000 << "," << duration << "\n"; 
    
    outfile.close();

    return 0;
}
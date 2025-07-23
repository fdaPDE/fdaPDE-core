#include <fstream>
#include <nlohmann/json.hpp> 
using json = nlohmann::json;
#include <geometry.h>
using namespace fdapde;
#include "domains.h"


int main() {

    //-----------------------CALL TO BUILD THE MESH AND PRINT ITS QUALITY MEASURES-------------------------------------

    std::vector<Eigen::Matrix<double, Eigen::Dynamic, 2>> domain={skyline, building1, building2, building3};  //insert the domain you want to triangulate, with its eventual subregions
    
    Delaunay<2, 2> del(domain,20,1,0);
    del.dcel().export_to_json("Meshes/Delaunay/delaunay_output.json");
    del.print_statistics();
    del.flip();

    return 0;
}
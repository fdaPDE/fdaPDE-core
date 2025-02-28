#ifndef GEOMETRIES_H
#define GEOMETRIES_H

#include <Eigen/Dense>
#include <vector>

inline Eigen::Matrix<double, 4, 2> rectangle =
    (Eigen::Matrix<double, 4, 2>() <<
        0.0, 0.0,
        4000.0, 0.0,
        4000.0, 2000.0,
        0.0, 2000.0
    ).finished();


inline Eigen::Matrix<double, 10, 2> star =
    (Eigen::Matrix<double, 10, 2>() <<
        0.0, 100.0,
        -22.45, 30.90,
        -95.11, 30.90,
        -36.33, -11.80,
        -58.78, -80.90,
        0.0, -38.20,
        58.78, -80.90,
        36.33, -11.80,
        95.11, 30.90,
        22.45, 30.90).finished();

inline Eigen::Matrix<double, 25, 2> skyline =
    (Eigen::Matrix<double, 25, 2>() <<
        0.0, 0.0,
        0.0, 15.0,
        5.0, 15.0,
        5.0, 10.0,
        10.0, 10.0,
        10.0, 13.0,
        15.0, 13.0,
        15.0, 18.0,
        25.0, 18.0,
        25.0, 20.0,
        26.5, 22.0,
        27.0, 25.0,
        27.45, 30.0,
        28.0, 25.0,
        28.5, 22.0,
        30.0, 20.0,
        30.0, 17.0,
        35.0, 17.0,
        35.0, 12.0,
        40.0, 12.0,
        40.0, 0.0,
        30.0, 0.0,
        25.0, 0.0,
        20.0, 0.0,
        10.0, 0.0
    ).finished();

inline Eigen::Matrix<double, 10, 2> building1 =
    (Eigen::Matrix<double, 10, 2>() <<
        0.0, 0.0,
        0.0, 15.0,
        5.0, 15.0,
        5.0, 10.0,
        10.0, 10.0,
        10.0, 13.0,
        15.0, 13.0,
        15.0, 18.0,
        25.0, 18.0,
        25.0, 0.0).finished();

inline Eigen::Matrix<double, 11, 2> building2 =
    (Eigen::Matrix<double, 11, 2>() <<
        25.0, 0.0,
        25.0, 18.0,
        25.0, 20.0,
        26.5, 22.0,
        27.0, 25.0,
        27.45, 30.0,
        28.0, 25.0,
        28.5, 22.0,
        30.0, 20.0,
        30.0, 17.0,
        30.0, 0.0).finished();
        //35.0, 17.0,
        //35.0, 12.0,
        //40.0, 12.0,
        //40.0, 0.0

inline Eigen::Matrix<double, 6, 2> building3 =
    (Eigen::Matrix<double, 6, 2>() <<
        30.0, 0.0,
        30.0, 17.0,   
        35.0, 17.0,
        35.0, 12.0,
        40.0, 12.0,
        40.0, 0.0).finished();


inline Eigen::Matrix<double, 4, 2> hole_building1 =
    (Eigen::Matrix<double, 4, 2>() <<
        20.0, 10.0,
        22.0, 10.0,
        22.0, 12.0,
        20.0, 12.0).finished();

inline Eigen::Matrix<double, 4, 2> hole_building2 =
    (Eigen::Matrix<double, 4, 2>() <<
        20.0, 5.0,
        22.0, 5.0,
        22.0, 7.0,
        20.0, 7.0).finished();

inline std::vector<std::vector<Eigen::Matrix<double, Eigen::Dynamic, 2>>> holes_skyline = {
    {hole_building1, hole_building2},
    {hole_building1, hole_building2},
    {},
    {}
};


inline Eigen::Matrix<double, 8, 2> letter_A =
    (Eigen::Matrix<double, 8, 2>() <<
        0.0, 0.0,     
        7.5, 0.0,
        10.0, 10.0,  
        20.0, 10.0,
        22.5, 0.0,
        30.0, 0.0,    
        22.5, 30.0,   
        7.5, 30.0   
    ).finished();

inline Eigen::Matrix<double, 4, 2> hole_A =
    (Eigen::Matrix<double, 4, 2>() <<
        11.5, 16.0,
        18.5, 16.0,
        17, 24.0,
        13, 24.0
    ).finished();

inline std::vector<std::vector<Eigen::Matrix<double, Eigen::Dynamic, 2>>> holes_A = {
    {hole_A}
};



#endif

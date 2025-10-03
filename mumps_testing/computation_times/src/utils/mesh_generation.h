#ifndef __MESH_GENERATION_H__
#define __MESH_GENERATION_H__

#include <Eigen/Dense>

using namespace Eigen;

struct Mesh {
    MatrixXd nodes;
    MatrixXi elements;
    MatrixXi boundary;
};

Mesh meshInterval(double a, double b, int n_nodes) {
    MatrixXd nodes(n_nodes, 1);
    MatrixXi elements(n_nodes - 1, 2);
    MatrixXi boundary(n_nodes, 1);
    double h = (b - a) / (n_nodes - 1);
    for (int i = 0; i < n_nodes; ++i) {
        double x = a + i * h;
        nodes(i, 0) = x;

        if (i < n_nodes - 1) {
            elements(i, 0) = i;
            elements(i, 1) = i + 1;
        }

        boundary(i, 0) = (i == 0 || i == n_nodes - 1) ? 1 : 0;
    }

    return Mesh {nodes, elements, boundary};
}

Mesh meshUnitInterval(int n_nodes) { return meshInterval(0.0, 1.0, n_nodes); }

Mesh meshRectangle(double a_x, double b_x, double a_y, double b_y, int n_nodes_x, int n_nodes_y) {
    MatrixXd nodes(n_nodes_x * n_nodes_y, 2);
    MatrixXi elements((n_nodes_x - 1) * (n_nodes_y - 1) * 2, 3);
    MatrixXi boundary(n_nodes_x * n_nodes_y, 1);
    double h_x = (b_x - a_x) / (n_nodes_x - 1);
    double h_y = (b_y - a_y) / (n_nodes_y - 1);
    int elementIndex = 0;
    for (int j = 0; j < n_nodes_y; ++j) {
        for (int i = 0; i < n_nodes_x; ++i) {
            int nodeIndex = j * n_nodes_x + i;
            double x = a_x + i * h_x;
            double y = a_y + j * h_y;
            nodes(nodeIndex, 0) = x;
            nodes(nodeIndex, 1) = y;

            if (i < n_nodes_x - 1 && j < n_nodes_y - 1) {
                int n1 = j * n_nodes_x + i;
                int n2 = j * n_nodes_x + i + 1;
                int n3 = (j + 1) * n_nodes_x + i;
                int n4 = (j + 1) * n_nodes_x + i + 1;

                elements(elementIndex, 0) = n1;
                elements(elementIndex, 1) = n2;
                elements(elementIndex, 2) = n3;
                ++elementIndex;

                elements(elementIndex, 0) = n2;
                elements(elementIndex, 1) = n4;
                elements(elementIndex, 2) = n3;
                ++elementIndex;
            }

            boundary(nodeIndex, 0) = (i == 0 || i == n_nodes_x - 1 || j == 0 || j == n_nodes_y - 1) ? 1 : 0;
        }
    }

    return Mesh {nodes, elements, boundary};
}

Mesh meshSquare(double a, double b, int n_nodes) { return meshRectangle(a, b, a, b, n_nodes, n_nodes); }

Mesh meshUnitSquare(int n_nodes) { return meshSquare(0.0, 1.0, n_nodes); }

Mesh meshParallelepiped(
  double a_x, double b_x, double a_y, double b_y, double a_z, double b_z, int n_nodes_x, int n_nodes_y, int n_nodes_z) {
    MatrixXd nodes(n_nodes_x * n_nodes_y * n_nodes_z, 3);
    MatrixXi elements((n_nodes_x - 1) * (n_nodes_y - 1) * (n_nodes_z - 1) * 6, 4);
    MatrixXi boundary(n_nodes_x * n_nodes_y * n_nodes_z, 1);
    double h_x = (b_x - a_x) / (n_nodes_x - 1);
    double h_y = (b_y - a_y) / (n_nodes_y - 1);
    double h_z = (b_z - a_z) / (n_nodes_z - 1);
    int elementIndex = 0;
    for (int k = 0; k < n_nodes_z; ++k) {
        for (int j = 0; j < n_nodes_y; ++j) {
            for (int i = 0; i < n_nodes_x; ++i) {
                int nodeIndex = k * n_nodes_x * n_nodes_y + j * n_nodes_x + i;
                double x = a_x + i * h_x;
                double y = a_y + j * h_y;
                double z = a_z + k * h_z;
                nodes(nodeIndex, 0) = x;
                nodes(nodeIndex, 1) = y;
                nodes(nodeIndex, 2) = z;

                if (i < n_nodes_x - 1 && j < n_nodes_y - 1 && k < n_nodes_z - 1) {
                    int n1 = k * n_nodes_x * n_nodes_y + j * n_nodes_x + i;
                    int n2 = k * n_nodes_x * n_nodes_y + j * n_nodes_x + i + 1;
                    int n3 = k * n_nodes_x * n_nodes_y + (j + 1) * n_nodes_x + i;
                    int n4 = k * n_nodes_x * n_nodes_y + (j + 1) * n_nodes_x + i + 1;
                    int n5 = (k + 1) * n_nodes_x * n_nodes_y + j * n_nodes_x + i;
                    int n6 = (k + 1) * n_nodes_x * n_nodes_y + j * n_nodes_x + i + 1;
                    int n7 = (k + 1) * n_nodes_x * n_nodes_y + (j + 1) * n_nodes_x + i;
                    int n8 = (k + 1) * n_nodes_x * n_nodes_y + (j + 1) * n_nodes_x + i + 1;

                    elements(elementIndex, 0) = n7;
                    elements(elementIndex, 1) = n3;
                    elements(elementIndex, 2) = n2;
                    elements(elementIndex, 3) = n1;
                    ++elementIndex;

                    elements(elementIndex, 0) = n7;
                    elements(elementIndex, 1) = n5;
                    elements(elementIndex, 2) = n2;
                    elements(elementIndex, 3) = n1;
                    ++elementIndex;

                    elements(elementIndex, 0) = n7;
                    elements(elementIndex, 1) = n6;
                    elements(elementIndex, 2) = n8;
                    elements(elementIndex, 3) = n2;
                    ++elementIndex;

                    elements(elementIndex, 0) = n7;
                    elements(elementIndex, 1) = n4;
                    elements(elementIndex, 2) = n8;
                    elements(elementIndex, 3) = n2;
                    ++elementIndex;

                    elements(elementIndex, 0) = n7;
                    elements(elementIndex, 1) = n4;
                    elements(elementIndex, 2) = n3;
                    elements(elementIndex, 3) = n2;
                    ++elementIndex;

                    elements(elementIndex, 0) = n7;
                    elements(elementIndex, 1) = n6;
                    elements(elementIndex, 2) = n5;
                    elements(elementIndex, 3) = n2;
                    ++elementIndex;
                }

                boundary(nodeIndex, 0) =
                  (i == 0 || i == n_nodes_x - 1 || j == 0 || j == n_nodes_y - 1 || k == 0 || k == n_nodes_z - 1) ? 1 :
                                                                                                                   0;
            }
        }
    }

    return Mesh {nodes, elements, boundary};
}

Mesh meshCube(double a, double b, int n_nodes) {
    return meshParallelepiped(a, b, a, b, a, b, n_nodes, n_nodes, n_nodes);
}

Mesh meshUnitCube(int n_nodes) { return meshCube(0.0, 1.0, n_nodes); }

#endif   // __MESH_GENERATION_H__
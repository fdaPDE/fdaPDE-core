#include <fdaPDE/drivers.h>
#include <mpi.h>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <numbers>
#include <random>

using namespace fdapde;
using namespace std::chrono;

using std::cos;
using std::exp;
using std::sin;
using std::numbers::pi;

void printSparsityPattern(const Eigen::SparseMatrix<double>& A, std::string filename);
void saveCSV(std::string input_filename, std::string output_filename);

int main() {
    /**
     * SET THE OUTPUT DIRECTORY
     */
    std::string output_directory = "../data";

    /**
     * SET THE TEMPORARY FILENAME
     *
     * Make sure that the name in not already used as it will be overwritten.
     */
    std::string temp_filename = "temp.csv";

    /**
     * SET THE OUTPUT FILENAMES
     */
    std::string output_filename = "srpde_mumps.csv";

    /**
     * SET THE PROBLEM SIZE
     */
    std::vector<int> N = {4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
    std::vector<int> N_obs = {4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};

    /**
     * SET THE NUMBER OF ITERATIONS
     */
    int n_iter = 10;

    // MPI initialization + rank (for output)
    MPI_Init(NULL, NULL);
    int rank;
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (rank == 0) std::cout << "MPI initialized with " << size << " processes." << std::endl;

    // create output directory
    if (rank == 0) {
        if (!std::filesystem::exists(output_directory)) { std::filesystem::create_directory(output_directory); }
    }

    // file manager for output
    std::ofstream file;
    std::string temp_path = output_directory + "/" + temp_filename;
    std::string output_path = output_directory + "/" + output_filename;

    if (rank == 0) {
        file.open(temp_path);
        if (!file.is_open()) { throw std::runtime_error("Unable to open file for writing."); }
        file << "N,N_obs,time\n";
    }

    for (auto n_nodes : N) {
        for (auto n_obs_per_side : N_obs) {
            for (int i = 0; i < n_iter; i++) {
                // ------------------------------------ geometry
                // int n_nodes = 21;
                Triangulation<2, 2> unit_square = Triangulation<2, 2>::UnitSquare(n_nodes, cache_cells);

                // ------------------------------------ data generation
                // int n_obs_per_side = 40;
                int n_obs = n_obs_per_side * n_obs_per_side;
                Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> coords(n_obs, 2);
                for (int i = 0; i < n_obs_per_side; ++i) {
                    for (int j = 0; j < n_obs_per_side; ++j) {
                        coords(i * n_obs_per_side + j, 0) = (1.0 / n_obs_per_side) * j;
                        coords(i * n_obs_per_side + j, 1) = (1.0 / n_obs_per_side) * i;
                    }
                }
                // evaluate spatial field at locations

                double std_dev = 0.1;
                unsigned int seed = 42;   // Fixed seed for reproducibility

                std::mt19937 gen(seed);   // Mersenne Twister PRNG with fixed seed
                std::normal_distribution<double> dist(0.0, std_dev);

                std::vector<double> noise;
                noise.resize(n_obs);
                for (int i = 0; i < n_obs; ++i) { noise[i] = dist(gen); }

                std::vector<double> y_vec;
                y_vec.resize(n_obs);
                // define your spatial field and evaluate in y_vec...
                // std::fill(y_vec.begin(), y_vec.end(), 1.0);   // here I just set ones to check it runs
                for (int i = 0; i < n_obs; i++) {
                    y_vec[i] = sin(
                                 2 * pi *
                                 ((0.5 * sin(0.5 * pi * coords(i, 2)) * exp(-1) + 1) * coords(i, 1) * cos(1) +
                                  coords(i, 2) * sin(1))) *
                                 cos(
                                   2 * pi *
                                   ((0.5 * sin(0.5 * pi * coords(i, 2)) * exp(-1) + 1) * coords(i, 1) * sin(1) -
                                    (0.5 * sin(5 * pi * coords(i, 1)) * exp(-1) + 1) * coords(i, 2))) +
                               noise[i];
                }

                // (this is still experimental.......)
                GeoFrame data(unit_square);
                data.add_scalar_layer<POINT>("layer1");
                auto geo_layer = geo_cast<POINT>(data["layer1"]);
                geo_layer->push_back(coords);
                geo_layer->load_vec({"y"}, y_vec);

                std::cout << *geo_layer << std::endl;   // display loaded data

                // ------------------------------------ physics
                FeSpace Vh(unit_square, P1<1>);
                TrialFunction f(Vh);
                TestFunction v(Vh);
                auto a = integral(unit_square)(
                  dot(grad(f), grad(v)));   // laplacian weak form (play with the PDE if you want...)
                ScalarField<2, decltype([](const Eigen::Matrix<double, 2, 1>&) { return 0; })> u;
                auto F = integral(unit_square)(u * v);   // homogeneous forcing

                // ------------------------------------ statistical model (still in development.....)
                // TIME START
                auto t1 = high_resolution_clock::now();

                internals::fe_elliptic_driver_impl model("y ~ f", data, a, F);
                model(1e-4);   // fit model with \lambda = 1e-4

                // TIME END
                auto t2 = high_resolution_clock::now();

                duration<double> duration = t2 - t1;
                MPI_Allreduce(MPI_IN_PLACE, &duration, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
                if (rank == 0) {
                    std::cout << n_nodes << "," << n_obs << "," << duration.count() << "\n";
                    file << n_nodes << "," << n_obs << "," << duration.count() << "\n";
                }
            }
        }
    }

    // print estimated spatial field
    // std::cout << model.f() << std::endl;
    if (rank == 0) {
        file.close();
        saveCSV(temp_path, output_path);
    }

    // delete temporary file
    if (rank == 0) {
        if (std::filesystem::exists(temp_path)) std::filesystem::remove(temp_path);
    }

    MPI_Finalize();

    return 0;
}

void printSparsityPattern(const Eigen::SparseMatrix<double>& A, std::string filename) {
    std::ofstream file(filename);
    file << "i,j\n";
    if (!file.is_open()) { throw std::runtime_error("Unable to open file for writing."); }
    for (int i = 0; i < A.rows(); ++i) {
        for (int j = 0; j < A.cols(); ++j) {
            if (A.coeff(i, j) != 0) { file << i << "," << j << "\n"; }
        }
    }
    file.close();
}

void saveCSV(std::string input_filename, std::string output_filename) {
    std::ifstream input_file(input_filename);
    std::ofstream output_file(output_filename);
    if (!input_file.is_open()) { throw std::runtime_error("Unable to open file for reading."); }
    if (!output_file.is_open()) { throw std::runtime_error("Unable to open file for writing."); }
    output_file << input_file.rdbuf();
    output_file.close();
}

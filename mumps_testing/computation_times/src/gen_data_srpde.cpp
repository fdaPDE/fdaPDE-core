#include <fdaPDE/drivers.h>

using namespace fdapde;

int main() {

    // ------------------------------------ geometry
    int n_nodes = 21;
    Triangulation<2, 2> unit_square = Triangulation<2, 2>::UnitSquare(n_nodes, cache_cells);

    // ------------------------------------ data generation
    int n_obs_per_side = 40;
    int n_obs = n_obs_per_side * n_obs_per_side;
    Eigen::Matrix<double, Dynamic, Dynamic> coords(n_obs, 2);
    for (int i = 0; i < n_obs_per_side; ++i) {
        for (int j = 0; j < n_obs_per_side; ++j) {
            coords(i * n_obs_per_side + j, 0) = (1.0 / n_obs_per_side) * j;
            coords(i * n_obs_per_side + j, 1) = (1.0 / n_obs_per_side) * i;
        }
    }
    // evaluate spatial field at locations
    std::vector<double> y_vec;
    y_vec.resize(n_obs);
    // define your spatial field and evaluate in y_vec...
    std::fill(y_vec.begin(), y_vec.end(), 1.0);   // here I just set ones to check it runs
    
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
    TestFunction  v(Vh);
    auto a = integral(unit_square)(dot(grad(f), grad(v)));   // laplacian weak form (play with the PDE if you want...)
    ScalarField<2, decltype([](const Eigen::Matrix<double, 2, 1>&) { return 0; })> u;
    auto F = integral(unit_square)(u * v);   // homogeneous forcing

    // ------------------------------------ statistical model (still in development.....)
    internals::fe_elliptic_driver_impl model("y ~ f", data, a, F);
    model(1e-4);   // fit model with \lambda = 1e-4

    // print estimated spatial field
    // std::cout << model.f() << std::endl;
    return 0;
}
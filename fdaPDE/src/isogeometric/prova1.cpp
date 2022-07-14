#include <array>
#include <vector>

template<M>
class Spline {
    public:

    Spline(const std::array<std::vector<double>, M>& knots,
            const std::array<int,M>& degree,
            const std::array<int,M>& index
            ) : knots_(knots), degree_(degree), index_(index) { } ;

    double operator() (const std::array<double,M>& x) {
        double value = 1;
        for(int dim = 0; dim < M; dim++){
            value *= evaluate1D(knots_[dim], degree_[dim], index[dim], x[dim]);
        }
        return value;
    }

    protected:
    
    double evaluate1D(const std::vector<double>& knots, int degree, int index, double x) {
        
        if(degree == 0){
            if(knots[index] <= x && x < knots[index + 1])
                return 1;
            else    
                return 0;
        } else{
            double denom1 = knots[index + degree] - knots[index];
            double denom2 = knots[index + degree + 1] - knots[index + 1];

            double term1 = 0;
            double term2 = 0; 

            if(denom1 != 0)
                term1 = (x - knots[index]) / denom1 * evaluate1D(knots, degree - 1, index, x);
            if(denom2 != 0)
                term2 = ( knots[index + degree + 1] - x) / denom2 * evaluate1D(knots, degree - 1, index + 1 , x);


            return term1 + term2;

        }
    }




    private:
    std::array<std::vector<double>, M> knots_;
    std::array<int,M> degree_;
    std::array<int,M> index_;

}

int main() {

    std::array<std::vector<double>, 2> knots = 
        {{0,0,0,1,2,3,4,4,5,5,5},{0,0,0,1,2,3,4,4,5,5,5}};

    std::array<int,2> degree = {2,2};

    std::array<int,2> index = {0,0};

    auto f = Spline(knots, degree, index);

    std::array<double,2> P = {0. , 1.};

    std::cout<<"Evaluation of f at P: "<<f(P)<<std::endl; 

    return 0;
    
}

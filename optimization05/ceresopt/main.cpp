#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <ceres/ceres.h>
#include <vector>

using namespace std;
using namespace Eigen;

ofstream fout("./output.txt");

struct CURVE_FITTING_COST {
    CURVE_FITTING_COST(double x, double y) : _x(x), _y(y) {}

    template<typename T>
    bool operator() (const T *const abc, T *residual) const {
        residual[0] = T(_y) - ceres::exp(abc[0] * T(_x) * T(_x) + abc[1] * T(_x) + abc[2]); // y-exp(ax^2+bx+c)
        return true;
    }

    const double _x, _y;
};

int main() {

    double ar = 1.0, br = 2.0, cr = 1.0; // ground truth parameter     
    double ae = 2.0, be = -1.0, ce = 5.0; // initial parameter
    int N = 100; // number of data
    double w_sigma = 1.0; // std of noise               
    double inv_sigma = 1.0 / w_sigma; // inverse std of noise
    cv::RNG rng;

    // Making initial data
    vector<double> x_data, y_data;
    for (int i = 0; i < N; i++) {
        double x = i / 100.0;
        x_data.push_back(x);
        y_data.push_back(exp(ar * x * x + br * x + cr) + rng.gaussian(w_sigma * w_sigma));
    }

    double abc[3] = { ae, be, ce };
    ceres::Problem problem;
    for (int i = 0; i < N; i++) {
        problem.AddResidualBlock(
            // Using    Auto Diff for structure CURVE_FITTING_COST(residual type), dimension of (1: output dim,3: input dim)
            new ceres::AutoDiffCostFunction<CURVE_FITTING_COST, 1, 3>(new CURVE_FITTING_COST(x_data[i], y_data[i])), 
            nullptr, // kernel function, not using here
            abc      // estimating variable
        );
    }

    ceres::Solver::Options options; // options for solver
    options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY; // using CHOLESKY Decomposition to solve linear equation
    options.minimizer_progress_to_stdout = true;               // print progress

    ceres::Solver::Summary summary;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    ceres::Solve(options, &problem, &summary);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "solve time cost = " << time_used.count() << " seconds. " << endl;

    fout << summary.BriefReport() << endl;
    fout << "estimated a,b,c = ";
    for (auto a : abc) fout << a << " ";
    fout << "\n";
    return 0;
}
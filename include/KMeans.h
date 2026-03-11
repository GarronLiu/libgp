#ifndef KMEANS
#define KMEANS

#include <iostream>
#include <vector>
#include <random>
#include <ctime>
#include <functional>
#include <Eigen/Dense>

namespace libgp {
    class Kmeans{
    public:
        size_t n_cluster;
        int max_iter;
        double tol;
        int n_jobs;
        Eigen::MatrixXd centers;
        std::vector<int> label;
        Kmeans();
        Kmeans(int n,int mi,double to,int jobs);
        ~Kmeans();
        void fit(const Eigen::MatrixXd& input);
        Eigen::MatrixXd & center();
        std::vector<int> predict(const Eigen::MatrixXd& data);
    };
}
#endif
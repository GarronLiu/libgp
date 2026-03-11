// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "cg.h"
#include "gp.h"
#include "gp_utils.h"
#include "matplotlibcpp.h"
#include "recursive_gp.h"
#include "rprop.h"
#include <Eigen/Dense>
#include <fstream>

namespace plt = matplotlibcpp;

using namespace libgp;

int main(int argc, char **argv) {
  if (argc != 7 && argc != 8) {
    std::cout
        << " Usage: ./recursive_gp_example <path_to_train_dataset>  "
           "<path_to_test_dataset> <batch_size> <skip> <u or v or r> <jitter>"
        << std::endl;
    return -1;
  }

  /* Parameters */
  size_t input_dim = 5; // number of input dimensions

  std::string train_file = argv[1];
  train_file += "/output_" + std::string(argv[5]) + "_sparse.txt";
  std::cout << "Initializing Recursive Gaussian Process model from "
            << train_file << std::endl;
  RecursiveGaussianProcess gp(train_file.c_str());

  std::string test_file = argv[2];
  test_file += "/" + std::string(argv[5]) + "dot_test.txt";
  std::cout << "Loading test dataset from " << test_file << std::endl;
  SampleSet *test_set;
  {
    GaussianProcess test(test_file.c_str());
    test_set = new SampleSet(*test.sampleset);
  }

  int n = test_set->size();
  double tss = 0, tss_ind = 0, error, error_ind, f, f_ind, y, var, var_ind;
  double css = 0;
  double sampling_interval = 0.1;

  std::vector<double> y_orig(n);
  std::vector<double> mean_pred(n);
  std::vector<double> sample_id(n);
  std::vector<double> var_pred(n);
  std::vector<double> two_sigma_lower(n);
  std::vector<double> two_sigma_upper(n);
  std::vector<double> squared_errors(n);
  std::vector<double> points_in_update_id;
  points_in_update_id.reserve(n);
  std::vector<double> points_in_update_target;
  points_in_update_target.reserve(n);
  std::vector<double> rmse_list;
  rmse_list.reserve(100);

  gp.setInducingTargetZeros();

  int batch_size = std::stoi(argv[3]);
  int skip = std::stoi(argv[4]);
  double jitter = std::stod(argv[6]);
  // total squared error

  std::map<std::string, std::string> keywords = {{"figure.dpi", "600"}};
  plt::rcparams(keywords);
  keywords = {{"font.family", "Times New Roman"}};
  plt::rcparams(keywords);
  keywords = {{"font.size", "7"}};
  plt::rcparams(keywords);
  float cm_to_inch = 1.0 / 2.54;
  plt::figure_size(17.6 * cm_to_inch * 600, 4.5 * cm_to_inch * 600);

  for (int batch_count = 0; batch_count < 100; batch_count++) {
    // prepare batch training data
    Eigen::MatrixXd batch_inputs(input_dim, batch_size);
    Eigen::VectorXd batch_targets(batch_size);
    for (int i = 0; i < batch_size; ++i) {
      int idx = batch_count * skip * batch_size + i * skip;
      if (idx > test_set->size() - 1) {
        std::cout << "Reach the end of test set!" << std::endl;
        break;
      }
      Eigen::VectorXd x_eigen = test_set->x(idx);
      batch_inputs.col(i) = x_eigen;
      batch_targets(i) = test_set->y(idx);
      points_in_update_id.push_back(idx * sampling_interval);
      points_in_update_target.push_back(test_set->y(idx));
    }

    if (batch_count > 0)
      gp.recursive_update(batch_inputs, batch_targets, jitter);

    // on test set
    for (int i = 0; i < n; ++i) {
      Eigen::VectorXd x_eigen = test_set->x(i);
      double x[] = {x_eigen[0], x_eigen[1], x_eigen[2], x_eigen[3], x_eigen[4]};
      y_orig.at(i) = test_set->y(i);
      mean_pred.at(i) = gp.f_sparse(x);
      squared_errors.at(i) =
          (mean_pred.at(i) - y_orig.at(i)) * (mean_pred.at(i) - y_orig.at(i));
      var_pred.at(i) = gp.var_sparse(x);
      two_sigma_lower.at(i) = mean_pred.at(i) - 2 * std::sqrt(var_pred.at(i));
      two_sigma_upper.at(i) = mean_pred.at(i) + 2 * std::sqrt(var_pred.at(i));
      sample_id.at(i) = i * sampling_interval;
    }

    // Calculate RMSE
    double mse = 0.0;
    for (int i = 0; i < n; ++i) {
      mse += squared_errors.at(i);
    }
    double rmse = std::sqrt(mse / n);

    if (batch_count == 0)
      std::cout << "Initial RMSE: " << rmse << std::endl;
    else
      std::cout << "After batch " << batch_count << ", RMSE: " << rmse
                << std::endl;

    rmse_list.push_back(rmse);

    if (batch_count > 16)
      break;

    if (batch_count % 4 != 0 || batch_count == 0) {
      continue;
    }
    // Create a 4x1 subplot and select the appropriate subplot based on
    // batch_count/4

    int subplot_idx = (batch_count / 4);
    plt::subplot(1, 4, subplot_idx);

    plt::title("batch counts = " + std::to_string(batch_count) +
                   "\n (RMSE: " + std::to_string(rmse) + ")",
               {{"fontsize", "8.5"}});

    keywords = {
        {"color", "#1f77b4"}, {"label", "prediction"}, {"linewidth", "0.75"}};
    plt::plot(sample_id, mean_pred, keywords);

    keywords = {{"color", "skyblue"}, {"label", "2$ \\sigma $ uncertainty"}};
    plt::fill_between(sample_id, two_sigma_lower, two_sigma_upper, keywords);

    keywords = {{"color", "gray"},
                {"label", "measurements"},
                {"linestyle", "--"},
                {"linewidth", "0.5"}};
    plt::plot(sample_id, y_orig, keywords);

    keywords = {{"color", "r"}, {"label", "batch update points"}};
    plt::scatter(points_in_update_id, points_in_update_target, 1.0, keywords);

    if (subplot_idx == 1)
      plt::legend({{"fontsize", "4.5"}});

    double y_min = *std::min_element(y_orig.begin(), y_orig.end());
    double y_max = *std::max_element(y_orig.begin(), y_orig.end());
    plt::ylim(y_min - 0.1, y_max + 0.1);

    plt::tight_layout();

    // plt::ylabel("Residual Dynamics");
    // plt::xlabel("time(s)");

    // plt::legend();

    // if(argv[5]){
    //   std::string save_file = argv[5];
    //   gp.write_sparse(save_file.c_str());
    // }
  }
  plt::save("sparse_gp_prediction.svg");
  plt::show();

  std::string rmse_output_path = argv[1];
  rmse_output_path += "/rmse_" + std::string(argv[5]) + ".txt";
  std::ofstream rmse_file(rmse_output_path);
  if (rmse_file.is_open()) {
    for (const auto &rmse_val : rmse_list) {
      rmse_file << rmse_val << std::endl;
    }
    rmse_file.close();
    std::cout << "RMSE values saved to " << rmse_output_path << std::endl;
  } else {
    std::cerr << "Failed to open file for writing RMSE: " << rmse_output_path
              << std::endl;
  }

  return EXIT_SUCCESS;
}

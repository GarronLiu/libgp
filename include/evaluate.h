#include "cg.h"

#include "gp.h"

#include "sampleset.h"

#include <iostream>

#include <Eigen/Core>

#include <chrono>

#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

using namespace libgp;

class GPEvaluation {
public:
  GPEvaluation(std::vector<std::string> legend);
  ~GPEvaluation();

  // 记录
  void record_epoch_results(std::string legend, Eigen::VectorXd &mean_pred,
                            Eigen::VectorXd &var_pred, double lml,
                            double optimization_time, bool verbose = false);

  void visualize_epoch_results();

  void visualize_uncertainty_bands();

  void visualize_summary_statistics();

  void setTrainSet(const std::shared_ptr<SampleSet> &train_set);
  void setTestSet(const std::shared_ptr<SampleSet> &test_set);

  // 绘制预测结果对比图
  void plotPredictionComparison(const Eigen::VectorXd &true_values,
                                const Eigen::VectorXd &predicted_values);

  // 记录优化时间
  void recordOptimizationTime(const std::chrono::duration<double> &duration);

  // 打印评估报告
  void printEvaluationReport();

private:
  std::vector<std::string> legend_;
  std::vector<std::vector<double>> rmse_lists_;
  std::vector<std::vector<double>> mae_lists_;
  std::vector<std::vector<double>> optimization_lists_;
  std::vector<std::vector<double>> lml_lists_;
  std::shared_ptr<SampleSet> train_set_;
  std::shared_ptr<SampleSet> test_set_;
  std::vector<Eigen::VectorXd> mean_predictions_;
  std::vector<Eigen::VectorXd> var_predictions_;

  std::vector<double> x_train_;
  std::vector<double> y_train_;
  std::vector<double> x_test_;
  std::vector<double> y_test_;
  std::string x_label;

  std::vector<std::string> linestyles_ = {"-", "--", "-.", ":"};
  // Professional, print-friendly, colorblind-safe palette (Paul Tol - Muted)
  std::vector<std::string> colors_ = {
      "#88CCEE", // light blue
      "#44AA99", // green
      "#117733", // dark green
      "#999933", // olive
      "#DDCC77", // sand
      "#CC6677", // rose
      "#882255", // wine
      "#AA4499", // purple
      "#332288", // dark blue
  };
  float cm_to_inch = 1.0 / 2.54;

  // 计算均值
  double calculateMean(const Eigen::VectorXd &values);
};
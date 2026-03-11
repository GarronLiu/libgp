#include "evaluate.h"

GPEvaluation::~GPEvaluation() {}

GPEvaluation::GPEvaluation(std::vector<std::string> legend): legend_(legend){
    rmse_lists_.resize(legend_.size());
    mae_lists_.resize(legend_.size());
    optimization_lists_.resize(legend_.size());
    lml_lists_.resize(legend_.size());
    mean_predictions_.resize(legend_.size());
    var_predictions_.resize(legend_.size());

    for(size_t i = 0; i < legend_.size(); i++){
        rmse_lists_[i].reserve(100);
        mae_lists_[i].reserve(100);
        optimization_lists_[i].reserve(100);
        lml_lists_[i].reserve(100);
    }
    std::map<std::string, std::string> keywords = { {"figure.dpi", "600"} };
    plt::rcparams(keywords);
    keywords = { {"font.family", "Times New Roman"} };
    plt::rcparams(keywords);
    keywords = { {"font.size", "5"} };
    plt::rcparams(keywords);

}

void GPEvaluation::record_epoch_results(std::string legend, Eigen::VectorXd& mean_pred, Eigen::VectorXd& var_pred, 
                                 double lml, double optimization_time, bool verbose){
    //find index
    auto it = std::find(legend_.begin(), legend_.end(), legend);
    if (it == legend_.end()) {
        std::cerr << "Legend not found!" << std::endl;
        return;
    }
    size_t index = std::distance(legend_.begin(), it);
    mean_predictions_[index] = mean_pred;
    var_predictions_[index] = var_pred;

    //calculate RMSE and MAE
    if (!test_set_) {
        std::cerr << "Test set not set!" << std::endl;
        return;
    }
    size_t N = test_set_->size();
    double rmse = 0.0;
    double mae = 0.0;
    for (size_t i = 0; i < N; ++i) {
        double y_true = test_set_->y(i);
        double y_pred = mean_pred(i);
        double error = y_true - y_pred;
        rmse += error * error;
        mae += std::abs(error);
    }
    rmse = std::sqrt(rmse / N);
    mae = mae / N;
    rmse_lists_[index].push_back(rmse);
    mae_lists_[index].push_back(mae);
    lml_lists_[index].push_back(lml);
    optimization_lists_[index].push_back(optimization_time);

    if(verbose){
        std::cout << "Evaluation for " << legend << ": RMSE = " << rmse
                  << ", MAE = " << mae
                  << ", LML = " << lml
                  << ", Optimization Time = " << optimization_time << "ms" << std::endl;
    }
    
}

void GPEvaluation::visualize_epoch_results(){
   if(!test_set_) {
        std::cerr << "Test set not set!" << std::endl;
        return;
    }
    plt::figure_size(17.6*cm_to_inch * 600, 4.5*cm_to_inch * 600);

    //绘制真值
    std::map<std::string, std::string> keywords;
    keywords = { {"color", "black"}, {"label", "GT"}, {"linestyle", "-"} };
    plt::plot(x_test_, y_test_, keywords);
    //绘制训练数据点
    keywords = { {"color", "red"}, {"marker", "o"}, {"label", "Training Data"} };
    plt::scatter(x_train_, y_train_, 1.0, keywords);

    //绘制预测均值
    for(size_t i = 0; i < legend_.size(); i++){
        std::vector<double> mean_y(mean_predictions_[i].data(), mean_predictions_[i].data() + mean_predictions_[i].size());
        keywords = { {"label", legend_[i]}, {"color", colors_[i]}, {"linestyle", linestyles_[i]}, {"linewidth", "1.0"} };
        plt::plot(x_test_, mean_y, keywords);
    }
    plt::xlabel(x_label);
    plt::ylabel("Output");
    plt::legend({{"loc", "lower right"},{"ncol", std::to_string(legend_.size() + 2)}});

    plt::show();

}

void GPEvaluation::visualize_uncertainty_bands(){
    if(!test_set_) {
        std::cerr << "Test set not set!" << std::endl;
        return;
    }
    std::map<std::string, std::string> keywords;
    plt::figure_size(8.8*cm_to_inch * 600, 9*cm_to_inch * 600);
    //绘制预测均值和不确定性带
    for(size_t i = 0; i < legend_.size(); i++){
        //绘制真值
        plt::subplot(legend_.size(), 1, i+1);
        keywords = { {"color", "black"}, {"label", "GT"}, {"linestyle", "-"} };
        plt::plot(x_test_, y_test_, keywords);
        std::vector<double> mean_y(mean_predictions_[i].data(), mean_predictions_[i].data() + mean_predictions_[i].size());
        std::vector<double> var_y(var_predictions_[i].data(), var_predictions_[i].data() + var_predictions_[i].size());
        std::vector<double> upper_y, lower_y;
        upper_y.reserve(var_y.size());
        lower_y.reserve(var_y.size());
        for(size_t j = 0; j < var_y.size(); j++){
            double stddev = std::sqrt(var_y[j]);
            upper_y.push_back(mean_y[j] + 3 * stddev);
            lower_y.push_back(mean_y[j] - 3 * stddev);
        }
        keywords = { {"label", legend_[i]}, {"color", colors_[i]}, {"linestyle", linestyles_[i]}, {"linewidth", "1.0"} };
        plt::plot(x_test_, mean_y, keywords);
        //填充不确定性带
        plt::fill_between(x_test_, lower_y, upper_y, {{"color", colors_[i]}, {"alpha", "0.2"}, {"label", "2σ Interval"} });
        plt::legend({{"loc", "lower right"},{"ncol", std::to_string(legend_.size())}});
    }
    plt::show();
}

void GPEvaluation::visualize_summary_statistics(){
    // lml
    
    // RMSE

    // MAE

    // Optimization Time
    
}

void GPEvaluation::setTrainSet(const std::shared_ptr<SampleSet>& train_set){
    train_set_ = train_set;
    x_train_.clear();
    x_train_.reserve(train_set_->size());
    y_train_ = train_set_->y();
    if(train_set_->x(0).size() != 1){
        x_label = "Sample Index";
        for(size_t i = 0; i < train_set_->size(); i++){
            x_train_.push_back(i);
        }
    }
    else{
        x_label = "Input";
        for(size_t i = 0; i < train_set_->size(); i++){
            x_train_.push_back(train_set_->x(i)(0));
        }
    }
}
void GPEvaluation::setTestSet(const std::shared_ptr<SampleSet>& test_set){
    test_set_ = test_set;
    x_test_.clear();
    x_test_.reserve(test_set_->size());
    y_test_ = test_set_->y();
    if(test_set_->x(0).size() != 1){
        x_label = "Sample Index";
        for(size_t i = 0; i < test_set_->size(); i++){
            x_test_.push_back(i);
        }
    }
    else{
        x_label = "Input";
        for(size_t i = 0; i < test_set_->size(); i++){
            x_test_.push_back(test_set_->x(i)(0));
        }
    }
}
#include "KMeans.h"

using namespace std;
using namespace Eigen;

namespace libgp
{
Kmeans::Kmeans()
{  //参数借鉴sklearn
  n_cluster = 2;
  max_iter = 300;
  tol = 0.0001;
  n_jobs = 1;
}
Kmeans::Kmeans(int n, int mi, double to, int jobs) : n_cluster(n), max_iter(mi), tol(to), n_jobs(jobs)
{
  std::srand(std::time(0));
}
Kmeans::~Kmeans()
{
}
void Kmeans::fit(const Eigen::MatrixXd& data)
{
  centers.resize(n_cluster, data.cols());
  MatrixXd temp_centers(n_cluster, data.cols());

  std::default_random_engine generator(std::time(0));
  std::uniform_int_distribution<int> distribution(0, data.rows() - 1);
  //自己定义的随机数生成，随机初始化聚类中心
  for (size_t i = 0; i < n_cluster; i++)
  {
    int index = distribution(generator);
    temp_centers.row(i) = data.row(index);
  }
  label = vector<int>(data.rows());

  int iter = 0;

  while (  //使用lambda函数，简化代码
      [=]() {
        size_t tmp_loop = centers.rows();
        for (size_t i = 0; i < tmp_loop; i++)
        {
          double diff = (centers.row(i) - temp_centers.row(i)).norm();
          if (diff > tol)
            return 1;
        }
        return 0;
      }() &&
      max_iter > iter)
  {
    centers = temp_centers;
    size_t tmp_loop2 = data.rows();
    for (size_t i = 0; i < tmp_loop2; i++)
    {
      int minclass = 0;
      double dis = 9999999999999;
      double temp_dis;
      for (size_t j = 0; j < n_cluster; j++)
      {
        temp_dis = [=]() { return (data.row(i) - centers.row(j)).norm(); }();
        // cout<<temp_dis<<endl;
        if (temp_dis < dis)
        {
          dis = temp_dis;
          minclass = j;
        }
      }
      label[i] = minclass;
    }
    temp_centers = MatrixXd::Zero(n_cluster, data.cols());
    vector<int> count = vector<int>(n_cluster);

    tmp_loop2 = data.rows();
    for (size_t i = 0; i < tmp_loop2; i++)
    {
      temp_centers.row(label[i]) = (count[label[i]] * temp_centers.row(label[i]) + data.row(i)) / (count[label[i]] + 1);
      count[label[i]] += 1;
    }
    iter += 1;
  }
  cout << "final iterator times:" << iter << endl;
}
MatrixXd& Kmeans::center()
{  //返回聚类中心
  return centers;
}
vector<int> Kmeans::predict(const Eigen::MatrixXd& data)
{
  //根据聚类中心返回聚类结果
  size_t num = data.cols();
  cout << num << endl;
  vector<int> result = vector<int>(num);
  for (size_t i = 0; i < num; i++)
  {
    int minclass = 0;
    double dis = 9999999999999;
    double temp_dis;
    for (size_t j = 0; j < n_cluster; j++)
    {
      temp_dis = [=]() { return (data.col(i) - centers.col(j)).norm(); }();
      // cout<<temp_dis<<endl;
      if (temp_dis < dis)
      {
        dis = temp_dis;
        minclass = j;
      }
    }
    result[i] = minclass;
  }
  return result;
}
}  // namespace libgp
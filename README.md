# Getting started

libgp is a C++ library for Gaussian process regression. A Gaussian process defines a distribution over functions and inference takes place directly in function space. It is fully specified by a mean function and a positive definite covariance function. This library uses two types of covariance functions, simple and composite. Composite functions can be composed of other composite functions, allowing flexible structures. 

## Building the code
Follow the standard CMake method of building:

    mkdir build; cd $_
    cmake ..
    make

### Testing the build
Once everything is built, you can check that all works fine with the following tests:

    cd tests
    ./gptest

And running an example:

    cd examples
    ./gpdense

which should return a MSE.

### Building the documentation
There are Doxygen comments in the header files. To compile (make sure you have doxygen installed):

    mkdir doc; cd $_
    doxygen ../doxygen/Doxyfile

Open doc/html/index.html with your favorite browser for the documentation.
If you want a pdf, go into latex and run:

    pdflatex refman.tex

## Implemented covariance functions

### Simple covariance functions

* Linear covariance function.
* Linear covariance function with automatic relevance detection. 
* Matern covariance function with nu=1.5 and isotropic distance measure.
* Matern covariance function with nu=2.5 and isotropic distance measure.
* Independent covariance function (white noise).
* Radial basis covariance function with compact support.
* Isotropic rational quadratic covariance function. 
* Squared exponential covariance function with automatic relevance detection.
* Squared exponential covariance function with isotropic distance measure.

### Composite covariance functions

* Sums of covariance functions.

### Mean function

* The mean function is fixed to zero.

## Training a model

{{fig1.jpg}} {{fig2.jpg}} {{fig3.jpg}} {{fig4.jpg}}

Initialize the model by specifying the input dimensionality and the covariance function.

    GaussianProcess gp(2, "CovSum ( CovSEiso, CovNoise)");

Set log-hyperparameter of the covariance function (see the Doxygen documentation, parameters should be given in order as listed).

    gp.covf().set_loghyper(params);

Add data to the training set. Input vectors x must be provided as double[] and targets y as double.

    gp.add_pattern(x, y);

Predict value or variance of an input vector x. 

    f = gp.f(x);
    v = gp.var(x);

## Read and write

Use write function to save a Gaussian process model and the complete training set to a file.

    void write(const char * filename);

A new instance of the Gaussian process can be instantiated from this file using the following constructor.

    GaussianProcess (const char * filename);

## Advanced topics

* hyper-parameter optimization
* custom covariance functions
* the libgp file format

### Hyper-parameter optimization

This library contains two methods for hyper-parameter optimization; the conjugate
gradient method, and Rprop (resilient backpropagation). We recommend using Rprop.

For an example of how to call the optimizers, see `test_optimizer.cc`

Reasons for using Rprop can be found in Blum & Riedmiller (2013),
Optimization of Gaussian Process Hyperparameters using Rprop, *European Symposium
on Artificial Neural Networks*, Computational Intelligence and Learning.


## Requirements

* [cmake](http://www.cmake.org/): cross-platform, open-source build system
* [Eigen3](http://eigen.tuxfamily.org/): template library for linear algebra
* [googletest](http://code.google.com/p/googletest) (optional)

## Release Notes

* 2012/10/11 version 0.1.4 \\
  log likelihood function and gradient computation \\
  hyper-parameter optimization using RProp \\
  online updates of the Cholesky decomposition \\

* 2011/09/28 version 0.1.3 \\
  improved organization of training data \\
  improved interfaces
  
* 2011/06/03 version 0.1.2 \\
  added Matern5 covariance function \\
  added isotropic rational quadratic covariance function \\
  added function to draw random data according to covariance function 
 
* 2011/05/27 version 0.1.1 \\
  google-tests added \\
  added Matern3 covariance function \\
  various bugfixes

* 2011/05/26 version 0.1.0
  basic functionality for standard gp regression \\
  most important covariance functions implemented \\
  capability to read and write models to disk 


## 不确定性传播 （以下内容由Gemini3 Pro提供）

在 EKF（扩展卡尔曼滤波）框架下，结合 RK4 积分和稀疏高斯过程（Sparse GP）进行预测时，协方差的传播需要考虑两部分不确定性：

状态不确定性：由上一时刻的协方差 $P_k$ 通过系统动力学雅可比矩阵传播而来。
模型不确定性：由 GP 回归产生的后验方差 $\Sigma_{GP}(x, u)$ 引入的系统噪声。
假设连续系统为： $$ \dot{x} = f_{nom}(x, u) + B_d \cdot \mu_{GP}(x, u) + w $$ 其中 $w \sim \mathcal{N}(0, B_d \Sigma_{GP}(x, u) B_d^T)$ 是由 GP 不确定性导出的过程噪声。

以下是具体的计算步骤：

1. 离散状态预测 (Mean Prediction)
这一步使用标准的 RK4 积分，将 GP 的均值 $\mu_{GP}$ 代入动力学方程。

$$ \begin{aligned} k_1 &= f(x_k, u_k) + B_d \mu_{GP}(x_k, u_k) \ k_2 &= f(x_k + 0.5 k_1 \Delta t, u_k) + B_d \mu_{GP}(x_k + 0.5 k_1 \Delta t, u_k) \ k_3 &= f(x_k + 0.5 k_2 \Delta t, u_k) + B_d \mu_{GP}(x_k + 0.5 k_2 \Delta t, u_k) \ k_4 &= f(x_k + k_3 \Delta t, u_k) + B_d \mu_{GP}(x_k + k_3 \Delta t, u_k) \ \hat{x}_{k+1} &= \hat{x}_k + \frac{\Delta t}{6}(k_1 + 2k_2 + 2k_3 + k_4) \end{aligned} $$

2. 离散协方差预测 (Covariance Prediction)
协方差预测公式为： $$ P_{k+1} = \Phi_k P_k \Phi_k^T + Q_{k} $$

我们需要计算状态转移矩阵 $\Phi_k$ 和等效离散过程噪声 $Q_k$。

A. 计算连续雅可比矩阵 $J$
首先需要计算连续动力学方程关于状态 $x$ 的偏导数（雅可比矩阵）。由于包含了 GP，需要利用链式法则求 GP 均值函数的导数（即 Kernel 的导数）。

$$ J(x) = \frac{\partial \dot{x}}{\partial x} = \frac{\partial f_{nom}}{\partial x} + B_d \frac{\partial \mu_{GP}}{\partial x} $$

注意：$\frac{\partial \mu_{GP}}{\partial x}$ 可以通过 libgp 或手动推导 RBF 核函数的导数获得。

B. 计算离散状态转移矩阵 $\Phi_k$ (RK4 线性化)
对于 RK4 积分，状态转移矩阵 $\Phi_k$ 实际上是积分过程的雅可比。最精确的方法是对 RK4 的四个步骤应用链式法则，但计算量巨大。 工程上常用的一阶近似（假设 $\Delta t$ 较小）或 矩阵指数近似：

方法一：矩阵指数（推荐） $$ \Phi_k \approx \exp(J(\hat{x}_k) \cdot \Delta t) \approx I + J(\hat{x}_k)\Delta t + \frac{1}{2}(J(\hat{x}_k)\Delta t)^2 $$

方法二：RK4 雅可比近似 如果需要更高精度，可以近似认为 $J$ 在步长内恒定： $$ \Phi_k \approx I + \Delta t J + \frac{\Delta t^2}{2} J^2 + \frac{\Delta t^3}{6} J^3 + \frac{\Delta t^4}{24} J^4 $$

C. 计算离散过程噪声 $Q_k$ (GP 不确定性积分)
GP 的后验协方差 $\Sigma_{GP}(x, u)$ 代表了模型对 $\dot{x}$ 预测的不确定性。在 EKF 中，这被视为过程噪声的功率谱密度。

$$ Q_{GP_continuous}(x) = B_d \Sigma_{GP}(x, u) B_d^T $$

离散化过程噪声 $Q_k$ 的计算公式为： $$ Q_k \approx \int_0^{\Delta t} \Phi(\tau) Q_{GP_continuous} \Phi(\tau)^T d\tau $$

工程近似计算： 通常假设在 $\Delta t$ 内 $Q_{GP}$ 和 $\Phi$ 变化不大，可以使用以下近似：

$$ Q_k \approx (B_d \Sigma_{GP}(\hat{x}_k, u_k) B_d^T) \cdot \Delta t $$

或者为了防止协方差过小导致滤波收敛慢，有时会加上额外的调节噪声 $Q_{tune}$： $$ Q_k = (B_d \Sigma_{GP}(\hat{x}k, u_k) B_d^T) \cdot \Delta t + Q{tune} $$

2. Woodbury 转换核心
我们需要计算 $K_{\alpha}^{-1} \mathbf{y}$（即 alpha 向量）和 $\log |K_{\alpha}|$（对数行列式）。

A. 逆矩阵转换 (Woodbury Identity)
Woodbury 公式指出： $$ (A + UCV)^{-1} = A^{-1} - A^{-1}U(C^{-1} + VA^{-1}U)^{-1}VA^{-1} $$

对应到我们的 $K_{\alpha} = \Lambda + H^\top I H$：

$A = \Lambda$ （$N \times N$ 对角阵，求逆只需 $O(N)$）
$U = H^\top$, $V = H$ （$N \times M$ 和 $M \times N$）
$C = I$ （$M \times M$ 单位阵）
代入公式： $$ K_{\alpha}^{-1} = \Lambda^{-1} - \Lambda^{-1} H^\top \underbrace{(I + H \Lambda^{-1} H^\top)^{-1}}_{B^{-1}} H \Lambda^{-1} $$

关键中间矩阵 $B$： 我们定义 $B = I + H \Lambda^{-1} H^\top$。 这是一个 $M \times M$ 的小矩阵。虽然 $H$ 是 $M \times N$，但计算 $H \Lambda^{-1} H^\top$ 只需要 $O(NM^2)$。对 $B$ 进行 Cholesky 分解（得到 L_B）只需要 $O(M^3)$。

计算 alpha = $K_{\alpha}^{-1} \mathbf{y}$ 的步骤：

计算 $\mathbf{y}' = \Lambda^{-1} \mathbf{y}$ （$O(N)$，即代码中的 y_scaled）
计算 $\mathbf{v} = H \mathbf{y}'$ （$O(NM)$）
计算 $\mathbf{z} = B^{-1} \mathbf{v}$ （$O(M^2)$，利用 L_B 求解）
计算结果 $\mathbf{\alpha} = \mathbf{y}' - \Lambda^{-1} H^\top \mathbf{z}$ （$O(NM)$）
全过程只有矩阵向量乘法，没有 $N \times N$ 矩阵分解。

B. 行列式转换 (Matrix Determinant Lemma)
同样利用 $K_{\alpha} = \Lambda + H^\top H$ 的结构： $$ \det(\Lambda + H^\top H) = \det(\Lambda) \cdot \det(I + H \Lambda^{-1} H^\top) $$ 即： $$ \det(K_{\alpha}) = \det(\Lambda) \cdot \det(B) $$

取对数： $$ \log |K_{\alpha}| = \underbrace{\sum \log(\Lambda_{ii})}{O(N)} + \underbrace{2 \sum \log((L_B){ii})}_{O(M)} $$

这避免了对 $N \times N$ 矩阵求行列式。

$K_{\alpha}^{-1} = \Lambda^{-1} - \Lambda^{-1} H^T (I + H \Lambda^{-1} H^T)^{-1} H \Lambda^{-1}$

在稀疏近似（如 FITC / DTC / VFE）中，我们通常使用诱导点 $u$作为中介，方差公式变为： 
$$ \mathbb{V}[f_*] \approx k_{**} - \mathbf{k}{*u} \underbrace{K{uu}^{-1} \mathbf{k}{u*}}{\text{先验部分}} + \mathbf{k}{*u} \underbrace{K{uu}^{-1} \Sigma_u K_{uu}^{-1}}{\text{后验修正}} \mathbf{k}{u*} $$ 
其中 $\Sigma_u$ 是诱导点的后验协方差矩阵。

$$ \mathbb{V}[f_*] = k_{**} - \mathbf{k}{*u} \underbrace{\left( K{uu}^{-1} - K_{uu}^{-1} \Sigma_u K_{uu}^{-1} \right)}{\text{我们定义的 } Q{pred}} \mathbf{k}_{u*} $$

3. 推导过程
将上述变量代入矩阵行列式引理：

$$ \begin{aligned} \det(K_{\alpha}) &= \det(\Lambda + H^T H) \ &= \det(\Lambda) \cdot \det(I_M + H \Lambda^{-1} H^T) \end{aligned} $$

在您的代码中，定义中间矩阵 $B$ 为： 
$$ B = I_M + H \Lambda^{-1} H^T $$

所以： 
$$ \det(K_{\alpha}) = \det(\Lambda) \cdot \det(B) $$

4. 取对数 (Log-Determinant)
为了计算对数似然（Log-Likelihood），我们需要计算 $\log(\det(\cdot))$。利用对数的性质 $\log(xy) = \log x + \log y$：

$$ \begin{aligned} \log |K_{\alpha}| &= \log \left( \det(\Lambda) \cdot \det(B) \right) \ &= \log(\det(\Lambda)) + \log(\det(B)) \ &= \log |\Lambda| + \log |B| \end{aligned} $$

利用关系 $\Sigma_u = L_{uu} B^{-1} L_{uu}^T$，我们将这一项代入 $Q_{pred}$ 的定义：

$$ \begin{aligned} Q_{pred} &= K_{uu}^{-1} - K_{uu}^{-1} (L_{uu} B^{-1} L_{uu}^T) K_{uu}^{-1} \ &= (L_{uu}^{-T} L_{uu}^{-1}) - (L_{uu}^{-T} L_{uu}^{-1}) (L_{uu} B^{-1} L_{uu}^T) (L_{uu}^{-T} L_{uu}^{-1}) \end{aligned} $$

注意矩阵乘法结合律，中间项可以消去： 
$$ L_{uu}^{-1} L_{uu} = I, \quad L_{uu}^T L_{uu}^{-T} = I $$

代入消元： 
$$ \begin{aligned} Q_{pred} &= L_{uu}^{-T} L_{uu}^{-1} - L_{uu}^{-T} (I \cdot B^{-1} \cdot I) L_{uu}^{-1} \ &= L_{uu}^{-T} L_{uu}^{-1} - L_{uu}^{-T} B^{-1} L_{uu}^{-1} \ &= L_{uu}^{-T} (I - B^{-1}) L_{uu}^{-1} \end{aligned} $$
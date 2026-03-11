// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "gp_utils.h"

namespace libgp {

double Utils::randn() {
  double u1 = 1.0 - drand48(), u2 = 1.0 - drand48();
  return sqrt(-2 * log(u1)) * cos(2 * M_PI * u2);
}

int *Utils::randperm(int n) {
  assert(n > 0);
  assert(n < RAND_MAX);
  int *array = new int[n];
  int i, j;
  double tmp;
  for (i = 0; i < n; i++)
    array[i] = i;
  for (i = n - 1; i > 0; --i) {
    j = int(drand48() * (i + 1));
    tmp = array[j];
    array[j] = array[i];
    array[i] = tmp;
  }
  return array;
}

size_t Utils::randi(size_t n) { return drand48() * n; }

double Utils::cdf_norm(double x) {
  double abs_x = fabs(x), build, norm;
  if (abs_x > 37)
    return norm = 0;
  else if (abs_x < 7.07106781186547) {
    build = 3.52624965998911E-02 * abs_x + 0.700383064443688;
    build = build * abs_x + 6.37396220353165;
    build = build * abs_x + 33.912866078383;
    build = build * abs_x + 112.079291497871;
    build = build * abs_x + 221.213596169931;
    build = build * abs_x + 220.206867912376;
    norm = exp(-abs_x * abs_x / 2) * build;
    build = 8.83883476483184E-02 * abs_x + 1.75566716318264;
    build = build * abs_x + 16.064177579207;
    build = build * abs_x + 86.7807322029461;
    build = build * abs_x + 296.564248779674;
    build = build * abs_x + 637.333633378831;
    build = build * abs_x + 793.826512519948;
    build = build * abs_x + 440.413735824752;
    norm = norm / build;
  } else {
    build = abs_x + 0.65;
    build = abs_x + 4 / build;
    build = abs_x + 3 / build;
    build = abs_x + 2 / build;
    build = abs_x + 1 / build;
    norm = exp(-abs_x * abs_x / 2) / build / 2.506628274631;
  }
  if (x > 0)
    norm = 1 - norm;
  return norm;
}

double Utils::friedman(double x[]) {
  return 10 * sin(M_PI * x[0] * x[1]) + 20 * pow(x[2] - 0.5, 2) + 10 * x[3] +
         5 * x[4];
}

double Utils::hill(double x, double y) {
  return sin(x - y) + 0.2 * pow(y, 3) + cos(y * (x - 0.5));
}

double Utils::toy_function(double x) {
  return cos(3 * x) - x * x / 9.0 + x / 6.0;
}

double Utils::irregular_wave_function(const double x) {
  // Sum of 5 sinusoidal components sampled from the JONSWAP spectrum
  // x is time (seconds)
  static bool initialized = false;
  static double omega[5];
  static double amp[5];
  static double phase[5];

  if (!initialized) {
    const int N = 5;
    const double g = 9.81;
    const double Tp = 5.0;                  // peak period (s)
    const double omega_p = 2.0 * M_PI / Tp; // peak angular frequency
    const double alpha = 0.0081;
    const double gamma = 3.3;

    // Choose 5 main frequencies around the peak
    const double factors[5] = {0.70, 0.85, 1.00, 1.15, 1.30};
    for (int i = 0; i < N; ++i) {
      omega[i] = omega_p * factors[i];
    }

    // Average spacing Δω
    double delta_omega = 0.0;
    for (int i = 1; i < N; ++i)
      delta_omega += (omega[i] - omega[i - 1]);
    delta_omega /= (N - 1);

    // Amplitudes from JONSWAP spectral density and random phases
    for (int i = 0; i < N; ++i) {
      double w = omega[i];
      double sigma = (w <= omega_p) ? 0.07 : 0.09;
      double r = exp(-0.5 * pow((w / omega_p - 1.0) / sigma, 2.0));
      double J = pow(gamma, r);
      double Si =
          alpha * g * g * pow(w, -5.0) * exp(-1.25 * pow(omega_p / w, 4.0)) * J;
      amp[i] = sqrt(2.0 * Si * delta_omega);
      phase[i] = 2.0 * M_PI * drand48();
    }

    initialized = true;
  }

  // Time-domain superposition
  double t = x;
  double eta = 0.0;
  for (int i = 0; i < 5; ++i) {
    eta += amp[i] * cos(omega[i] * t + phase[i]);
  }
  return eta;
}

double Utils::sign(double x) {
  if (x > 0)
    return 1.0;
  if (x < 0)
    return -1.0;
  return 0.0;
}
} // namespace libgp

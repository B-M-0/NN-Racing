#include "neural_net.h"
#include <random>
#include <stdexcept>

NeuralNet::NeuralNet(const std::vector<int> &topology) {
  static std::mt19937 gen;
  std::uniform_int_distribution<int> random_1_to_100(1, 100);

  // float mutation_value = gaussian_noise(gen); // Gets a tiny decimal like
  // -0.04 int random_index     = random_1_to_100(gen); // Gets a whole number
  // like 73 float probability    = chance(gen);         // Gets a flat
  // percentage like 0.82

  if (topology.size() < 2) {
    throw std::invalid_argument("Topology must have at least 2 layers");
  }
  for (size_t i = 0; i < topology.size() - 1; ++i) {
    weights.push_back(Matrix(topology[i], topology[i + 1]));
    biases.push_back(Matrix(topology[i], topology[i + 1], 0.01f));
  }
}

Matrix NeuralNet::forward(const Matrix &input, const Matrix &bias,
                          std::function<Matrix(Matrix)> activation) {
  Matrix output = input;
  for (const auto &w : weights) {
    output = activation(output * w + bias);
  }
  return output;
}

Matrix NeuralNet::mutate(Matrix &A, float mutation_rate = 0.05f,
                         float mutation_strength = 0.05f) {

  std::normal_distribution<float> gaussian_noise(0.0f, mutation_strength);
  std::uniform_real_distribution<float> chance(0.0f, 1.0f);

  for (int size_t = 0; size_t < A.size(); size_t++) {
    A(size_t) =
        A(size_t) + ((chance(gen) < mutation_rate) * gaussian_noise(gen));
  }
  return A;
}

float relu_activation(float x) { return std::max(x, 0.0f); }
float tanh_activation(float x) { return std::tanh(x); }
Matrix NeuralNet::Tanh(const Matrix &A) { return map(tanh_activation, A); }
Matrix NeuralNet::ReLu(const Matrix &A) { return map(relu_activation, A); }

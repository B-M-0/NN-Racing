#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include "matrix.h"
#include <random>
#include <vector>

class NeuralNet {
private:
  std::vector<Matrix> weights;
  std::vector<Matrix> biases;
  std::random_device rd;
  std::mt19937 gen;

public:
  NeuralNet(const std::vector<int> &topology);
  Matrix forward(const Matrix &input, const Matrix &bias,
                 std::function<Matrix(Matrix)> activation);
  Matrix mutate(Matrix &input, float mutation_rate, float mutation_strength);
  friend float relu_activation(float x);
  friend float tanh_activation(float x);
  Matrix Tanh(const Matrix &A);
  Matrix ReLu(const Matrix &A);
};

#endif // NEURAL_NET_H

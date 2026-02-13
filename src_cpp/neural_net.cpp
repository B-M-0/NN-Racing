#include "matrix.h"

class NeuralNet {
private:
  Matrix weights1, weights2;

public:
  NeuralNet(int input_size, int hidden_size, int output_size)
      : weights1(input_size, hidden_size), weights2(hidden_size, output_size) {}

  Matrix forward(const Matrix &input) {
    Matrix output = input * weights1;
    output = output * weights2;
    return output;
  }
};
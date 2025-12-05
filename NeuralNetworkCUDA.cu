#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

// Helper to check for CUDA errors
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// --- CUDA Kernels ---

// Sigmoid device function
__device__ double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Derivative of Sigmoid: y * (1 - y)
__device__ double sigmoidDerivative(double y) {
    return y * (1.0 - y);
}

// Forward Pass Kernel
// Computes outputs for a layer: Output = Sigmoid(Weights^T * Input + Bias)
// Grid: 1 block, BlockDim: n_outputs (one thread per output neuron)
__global__ void forwardLayerKernel(const double* inputs, const double* weights, const double* biases, 
                                   double* outputs, int n_inputs, int n_outputs) {
    int j = threadIdx.x;
    if (j < n_outputs) {
        double activation = biases[j];
        for (int i = 0; i < n_inputs; i++) {
            // weights is flattened: row-major [input][output]
            // index = i * n_outputs + j
            activation += inputs[i] * weights[i * n_outputs + j];
        }
        outputs[j] = sigmoid(activation);
    }
}

// Calculate Output Deltas Kernel
// delta = (target - output) * sigmoid_derivative(output)
// Grid: 1 block, BlockDim: n_outputs
__global__ void outputDeltaKernel(const double* targets, const double* outputs, 
                                  double* deltas, int n_outputs) {
    int j = threadIdx.x;
    if (j < n_outputs) {
        double error = targets[j] - outputs[j];
        deltas[j] = error * sigmoidDerivative(outputs[j]);
    }
}

// Calculate Hidden Deltas Kernel
// delta_h = (sum(delta_o * weight_ho)) * sigmoid_derivative(output_h)
// Grid: 1 block, BlockDim: n_hidden
__global__ void hiddenDeltaKernel(const double* next_deltas, const double* next_weights, 
                                  const double* curr_outputs, double* curr_deltas, 
                                  int n_curr, int n_next) {
    int i = threadIdx.x; // index of hidden neuron
    if (i < n_curr) {
        double error = 0.0;
        for (int j = 0; j < n_next; j++) {
            // next_weights is [n_curr][n_next]
            // weight connecting hidden i to output j is at [i * n_next + j]
            error += next_deltas[j] * next_weights[i * n_next + j];
        }
        curr_deltas[i] = error * sigmoidDerivative(curr_outputs[i]);
    }
}

// Update Weights and Biases Kernel
// w_ij += lr * input_i * delta_j
// b_j += lr * delta_j
// Grid: 1 block, BlockDim: n_outputs
__global__ void updateWeightsKernel(const double* inputs, const double* deltas, 
                                    double* weights, double* biases, 
                                    int n_inputs, int n_outputs, double learningRate) {
    int j = threadIdx.x; // index of output neuron (or hidden neuron for first layer)
    if (j < n_outputs) {
        // Update bias
        biases[j] += deltas[j] * learningRate;
        
        // Update weights connected to this neuron
        for (int i = 0; i < n_inputs; i++) {
            weights[i * n_outputs + j] += inputs[i] * deltas[j] * learningRate;
        }
    }
}

// --- Neural Network Class ---

class NeuralNetwork {
private:
    int inputNodes;
    int hiddenNodes;
    int outputNodes;
    double learningRate;

    // Device Pointers
    double *d_hiddenWeights; // Size: input * hidden
    double *d_outputWeights; // Size: hidden * output
    double *d_hiddenBiases;  // Size: hidden
    double *d_outputBiases;  // Size: output
    
    double *d_inputs;        // Size: input
    double *d_targets;       // Size: output
    
    double *d_hiddenOutputs; // Size: hidden
    double *d_finalOutputs;  // Size: output
    
    double *d_hiddenDeltas;  // Size: hidden
    double *d_outputDeltas;  // Size: output

public:
    NeuralNetwork(int inputs, int hidden, int outputs, double lr = 0.1) 
        : inputNodes(inputs), hiddenNodes(hidden), outputNodes(outputs), learningRate(lr) {
        
        std::srand(std::time(0));

        // Allocate Host Memory for initialization
        std::vector<double> h_hiddenWeights(inputs * hidden);
        std::vector<double> h_outputWeights(hidden * outputs);
        std::vector<double> h_hiddenBiases(hidden);
        std::vector<double> h_outputBiases(outputs);

        // Random Initialization
        for(auto& w : h_hiddenWeights) w = ((double)std::rand()/RAND_MAX)*2.0 - 1.0;
        for(auto& w : h_outputWeights) w = ((double)std::rand()/RAND_MAX)*2.0 - 1.0;
        for(auto& b : h_hiddenBiases)  b = ((double)std::rand()/RAND_MAX)*2.0 - 1.0;
        for(auto& b : h_outputBiases)  b = ((double)std::rand()/RAND_MAX)*2.0 - 1.0;

        // Allocate Device Memory
        cudaCheckError(cudaMalloc(&d_hiddenWeights, inputs * hidden * sizeof(double)));
        cudaCheckError(cudaMalloc(&d_outputWeights, hidden * outputs * sizeof(double)));
        cudaCheckError(cudaMalloc(&d_hiddenBiases, hidden * sizeof(double)));
        cudaCheckError(cudaMalloc(&d_outputBiases, outputs * sizeof(double)));
        
        cudaCheckError(cudaMalloc(&d_inputs, inputs * sizeof(double)));
        cudaCheckError(cudaMalloc(&d_targets, outputs * sizeof(double)));
        cudaCheckError(cudaMalloc(&d_hiddenOutputs, hidden * sizeof(double)));
        cudaCheckError(cudaMalloc(&d_finalOutputs, outputs * sizeof(double)));
        cudaCheckError(cudaMalloc(&d_hiddenDeltas, hidden * sizeof(double)));
        cudaCheckError(cudaMalloc(&d_outputDeltas, outputs * sizeof(double)));

        // Copy Initial Weights/Biases to Device
        cudaCheckError(cudaMemcpy(d_hiddenWeights, h_hiddenWeights.data(), inputs * hidden * sizeof(double), cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(d_outputWeights, h_outputWeights.data(), hidden * outputs * sizeof(double), cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(d_hiddenBiases, h_hiddenBiases.data(), hidden * sizeof(double), cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(d_outputBiases, h_outputBiases.data(), outputs * sizeof(double), cudaMemcpyHostToDevice));
    }

    ~NeuralNetwork() {
        cudaFree(d_hiddenWeights); cudaFree(d_outputWeights);
        cudaFree(d_hiddenBiases);  cudaFree(d_outputBiases);
        cudaFree(d_inputs);        cudaFree(d_targets);
        cudaFree(d_hiddenOutputs); cudaFree(d_finalOutputs);
        cudaFree(d_hiddenDeltas);  cudaFree(d_outputDeltas);
    }

    std::vector<double> feedForward(const std::vector<double>& inputs) {
        // Copy inputs to device
        cudaCheckError(cudaMemcpy(d_inputs, inputs.data(), inputNodes * sizeof(double), cudaMemcpyHostToDevice));

        // 1. Hidden Layer Forward
        forwardLayerKernel<<<1, hiddenNodes>>>(d_inputs, d_hiddenWeights, d_hiddenBiases, d_hiddenOutputs, inputNodes, hiddenNodes);
        cudaCheckError(cudaGetLastError());

        // 2. Output Layer Forward
        forwardLayerKernel<<<1, outputNodes>>>(d_hiddenOutputs, d_outputWeights, d_outputBiases, d_finalOutputs, hiddenNodes, outputNodes);
        cudaCheckError(cudaGetLastError());

        // Copy result back to host
        std::vector<double> result(outputNodes);
        cudaCheckError(cudaMemcpy(result.data(), d_finalOutputs, outputNodes * sizeof(double), cudaMemcpyDeviceToHost));
        
        return result;
    }

    void train(const std::vector<double>& inputs, const std::vector<double>& targets) {
        // Copy inputs and targets to device
        cudaCheckError(cudaMemcpy(d_inputs, inputs.data(), inputNodes * sizeof(double), cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(d_targets, targets.data(), outputNodes * sizeof(double), cudaMemcpyHostToDevice));

        // --- Forward Pass ---
        forwardLayerKernel<<<1, hiddenNodes>>>(d_inputs, d_hiddenWeights, d_hiddenBiases, d_hiddenOutputs, inputNodes, hiddenNodes);
        forwardLayerKernel<<<1, outputNodes>>>(d_hiddenOutputs, d_outputWeights, d_outputBiases, d_finalOutputs, hiddenNodes, outputNodes);

        // --- Backward Pass ---
        
        // 1. Calculate Output Deltas
        outputDeltaKernel<<<1, outputNodes>>>(d_targets, d_finalOutputs, d_outputDeltas, outputNodes);

        // 2. Calculate Hidden Deltas
        hiddenDeltaKernel<<<1, hiddenNodes>>>(d_outputDeltas, d_outputWeights, d_hiddenOutputs, d_hiddenDeltas, hiddenNodes, outputNodes);

        // 3. Update Output Weights and Biases
        updateWeightsKernel<<<1, outputNodes>>>(d_hiddenOutputs, d_outputDeltas, d_outputWeights, d_outputBiases, hiddenNodes, outputNodes, learningRate);

        // 4. Update Hidden Weights and Biases
        updateWeightsKernel<<<1, hiddenNodes>>>(d_inputs, d_hiddenDeltas, d_hiddenWeights, d_hiddenBiases, inputNodes, hiddenNodes, learningRate);
        
        cudaCheckError(cudaDeviceSynchronize());
    }
};

int main() {
    // 2 Inputs, 2 Hidden Neurons, 1 Output
    NeuralNetwork nn(2, 2, 1, 0.5);

    struct TrainingData {
        std::vector<double> inputs;
        std::vector<double> targets;
    };

    std::vector<TrainingData> data = {
        {{0, 0}, {0}},
        {{0, 1}, {1}},
        {{1, 0}, {1}},
        {{1, 1}, {0}}
    };

    std::cout << "Training on GPU..." << std::endl;

    // Training Loop
    for (int i = 0; i < 10000; i++) {
        int index = std::rand() % 4;
        nn.train(data[index].inputs, data[index].targets);
    }

    std::cout << "Training Complete. Testing results:" << std::endl;

    // Testing
    for (const auto& sample : data) {
        std::vector<double> output = nn.feedForward(sample.inputs);
        std::cout << "Input: " << sample.inputs[0] << ", " << sample.inputs[1] 
                  << " | Prediction: " << output[0] 
                  << " | Target: " << sample.targets[0] << std::endl;
    }

    return 0;
}
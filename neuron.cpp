#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <numeric>
using namespace std;
class Neuron{
    public:
    vector<double> weights;
    double bias;
    Neuron(int input_size){
        for (int i=0;i< input_size;i++){
            weights.push_back(0.5);
        }
        bias=0.1;
    }

    double sigmoid(double x){
        return 1.0/(1.0+exp(-x));
    }

    double feedForward(const vector<double>& inputs){
        double total =0.0;
        //Dot product
        for (size_t i=0;i<inputs.size(); i++){
            total+= inputs[i]*weights[i];
        }
        return sigmoid(total+bias);
    }
};

int main(){
    vector<double> inputs={1.0,2.0,3.0,4.0};
    Neuron n(4);
    cout<<n.feedForward(inputs)<<endl;
    return 0;
}
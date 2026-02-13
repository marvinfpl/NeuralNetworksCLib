#include <stdlib.h>
#include <math.h>
#include <layers.h>
#include <activation.h>

double forward_neuron(Neuron *neuron, double *input) {
    double sum = 0.0;
    for (int i = 0; i < sizeof(neuron->weights); i++) {
        sum += neuron->weights[i] * input[i];
    }
    return sum + neuron->bias;
}

double backward_neuron(Neuron *neuron, double *doutput) {

}

LinearLayer create_layer(int n_input, int n_output) {
    double *weights = malloc(n_input * sizeof(double));
    
}
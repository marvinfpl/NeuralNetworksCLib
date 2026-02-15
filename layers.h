#ifndef LAYERS_H
#define LAYERS_H
#include <activation.h>

typedef struct Neuron {
    double *weights;
    double bias;
    double dz;
    double *dinput;
    int n_input;
} Neuron;

typedef struct LinearLayer {
    Neuron *neurons;
    ActivationFunction activation;
    int n_input;
    int n_output;
    double learning_rate;
} LinearLayer;

double forward_neuron(Neuron *neuron, double *input);

double backward_neuron(Neuron *neuron, double doutput, double learning_rate);

LinearLayer *create_layer(int n_input, int n_output, ActivationType type, double learning_rate);

double *forward_layer(LinearLayer *layer, double *input);

double *backward_layer(LinearLayer *layer, double *doutput);

#endif
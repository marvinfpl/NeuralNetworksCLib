#ifndef LAYERS_H
#define LAYER_H
#include <activation.h>

typedef struct {
    double *weights;
    double bias;
    double *input;
    double *dinput;
    double *dweights;
    double dbias;
}Neuron;

typedef struct {
    Neuron *neurons;
    int n_input;
    int n_output;
    ActivationFunction activation;
} LinearLayer;

double forward_neuron(Neuron *neuron, double *input);
/* performs dot product given a neuron and some input data */

double backward_neuron(Neuron *neuron, double *doutput);
/* returns the gradient of the loss with respect to the given neuron */

LinearLayer create_layer(int n_input, int n_output);
/* returns a linear layer at the given dimensions */

void free_layer(LinearLayer *layer);
/* free the allocated memory of the weights of the given layer */

double *forward_layer(LinearLayer *layer, double *input);
/* performs the forward_neuron function on every neuron of the given layer */

double *backward_layer(LinearLayer *layer, double *doutput);
/* performs the backward_neuron function on every neuron of the given layer */

#endif
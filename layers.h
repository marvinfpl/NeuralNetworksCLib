#ifndef LAYERS_H
#define LAYERS_H
#include <activation.h>
#include <optimizer.h>

typedef struct Neuron {
    double *weights;
    double bias;
    double dz;
    double *dinput;
    int n_input;
} Neuron;

typedef enum LayerType {
    LINEAR,
    CONVOLUTIONAL,
    DROPOUT,
} LayerType;

typedef union LayerData {
    LinearLayer *linear;
    DropoutLayer *dropout;
    ConvolutionalLayer *convo;
} LayerData;

typedef struct Layer {
    LayerType type;
    LayerData data;
} Layer;

typedef struct LinearLayer {
    Neuron *neurons;
    ActivationFunction activation;
    int n_input;
    int n_output;
    double learning_rate;
    Optimizer optimizer;
} LinearLayer;

typedef struct DropoutLayer {
    LinearLayer *linear;
    LinearLayer *masked;
    double p;
} DropoutLayer;

void mask_layer(DropoutLayer *l);

void mask_neuron(Neuron *n, double p);

void reset_mask(DropoutLayer *l);

typedef struct ConvolutionalLayer {
    Neuron *neurons;
    ActivationFunction activation;
    int n_input;
    int n_output;
    double learning_rate;
    Optimizer optimizer;
} ConvolutionalLayer;

double forward_neuron(Neuron *neuron, double *input);

double *backward_neuron(Neuron *neuron, double doutput, ActivationFunction *activation, double learning_rate);

LinearLayer *create_layer(int n_input, int n_output, ActivationType type, double learning_rate);

double *forward_layer(LinearLayer *layer, double *input);

double *backward_layer(LinearLayer *layer, double *doutput);

#endif
#include <layers.h>
#include <activation.h>
#include <stdlib.h>

double forward_neuron(Neuron *neuron, double *input) {
    neuron->dinput = input;
    double z = neuron->bias;
    for (int i = 0; i < neuron->n_input; i++) {
        z += neuron->weights[i] * input[i];
    }
    neuron->dz = z;
    return z;
}

double *backward_neuron(Neuron *neuron, double doutput, ActivationFunction *activation, double learning_rate) {
    double dz = doutput * activation->backward(neuron->dz);
    double *dinput = calloc(neuron->n_input, sizeof(double));

    for (int i = 0; i < neuron->n_input; i++) {
        double dw = dz * neuron->dinput[i];
        dinput[i] = dz * neuron->weights[i];
        neuron->weights[i] -= learning_rate * dw; 
    }
    neuron->bias -= learning_rate * dz;
    return dinput;
}

LinearLayer *create_layer(int n_input, int n_output, ActivationType type, double learning_rate) {
    LinearLayer *layer = malloc(sizeof(LinearLayer));
    ActivationFunction *act = create_activation(type);
    Neuron *neurons = malloc(n_output * sizeof(Neuron));
    layer->activation = *act;
    layer->n_input = n_input;
    layer->n_output = n_output;
    layer->neurons = neurons;
    layer->learning_rate = learning_rate;

    for (int i = 0; i < n_output; i++) {
        Neuron *n = &neurons[i];
        n->n_input = n_input;
        n->bias = 0.0;
        n->weights = calloc(n_input, sizeof(double));
        neurons[i] = *n;
    }

    return layer;
}

double *forward_layer(LinearLayer *layer, double *input) {
    double *output = malloc(layer->n_output * sizeof(double));
    
    for (int i = 0; i < layer->n_output; i++) {
        output[i] = forward_neuron(&layer->neurons[i], input);
    }

    return output;
}

double *backward_layer(LinearLayer *layer, double *doutput) {
    double *dinput = calloc(layer->n_input, sizeof(double));

    for (int i = 0; i < layer->n_output; i++) {
        double *neuron_dinput = backward_neuron(&layer->neurons[i], doutput[i], &layer->activation, layer->learning_rate);
        for (int j = 0; j < layer->n_input; j++) {
            dinput[j] += neuron_dinput[j];
        }
        free(neuron_dinput);
    }
    return dinput;
}

void mask_layer(DropoutLayer *l) {
    for (int i = 0; i < l->linear->n_output; i++) {
        for (int j = 0; j < l->masked->neurons[i].n_input; j++) {
            if ((double)rand() / (double)RAND_MAX < l->p) {
                l->masked->neurons[i].weights[j] = 0.0;
            }
        }
    }
}

void reset_mask(DropoutLayer *l) {
    for (int i = 0; i < l->linear->n_output; i++) {
        for (int j = 0; j < l->masked->neurons[i].n_input; j++) {
            l->masked->neurons[i].weights[j] = l->linear->neurons[i].weights[j];
        }
    }
}
#include <layers.h>
#include <model.h>
#include <time.h>
#include <stdlib.h>
#include <utils.h>
#include <math.h>

Model *init_Model(LayerType *type, Optimizer *optim, ActivationFunction *activations, double learning_rate, int *n_input, int *n_output, int size) {
    Model *m = malloc(sizeof(Model));
    m->optim = optim;
    m->size = size;
    Layer **layers = malloc(sizeof(Layer*) * size);

    for (int i = 0; i < size; i++) {
        Layer *l = malloc(sizeof(Layer));
        l->type = type[i];
        switch (type[i]) {
            case DROPOUT: {
                DropoutLayer *layer = malloc(sizeof(DropoutLayer));
                layer->activation = activations[i];
                layer->learning_rate = learning_rate;
                layer->n_input = n_input[i];
                layer->n_output = n_output[i];
                layer->optimizer = *optim;
                l->data.dropout = layer;
                layers[i] = l;
                break;
            }
            case LINEAR: {
                LinearLayer *layer = malloc(sizeof(LinearLayer));
                layer->activation = activations[i];
                layer->learning_rate = learning_rate;
                layer->n_input = n_input[i];
                layer->n_output = n_output[i];
                layer->optimizer = *optim;
                l->data.linear = layer;
                layers[i] = l;
                break;
            }
            case CONVOLUTIONAL: {
                ConvolutionalLayer *layer = malloc(sizeof(ConvolutionalLayer));
                layer->activation = activations[i];
                layer->learning_rate = learning_rate;
                layer->n_input = n_input[i];
                layer->n_output = n_output[i];
                layer->optimizer = *optim;
                l->data.convo = layer;
                l->type = type[i];
                break;
            }
        }
        m->layers[i] = l;
    }
    return m;
}

void xavier(Layer *l) {
    switch (l->type) {
        case LINEAR:
            double lowerBound = -sqrt(6.0 / (l->data.linear->n_input + l->data.linear->n_output));
            double upperBound = sqrt(6.0 / (l->data.linear->n_input + l->data.linear->n_output));
            for (int i = 0; i < l->data.linear->n_output; i++) {
                for (int j = 0; j < l->data.linear->neurons[i].n_input; j++) {
                    double w_init = uniform_distribution(lowerBound, upperBound);
                    l->data.linear->neurons[i].weights[j] = w_init;
                }
                double b_init = uniform_distribution(lowerBound, upperBound);
                l->data.linear->neurons[i].bias = b_init;
            }
            break;
        case DROPOUT:
            double lowerBound = -sqrt(6.0 / (l->data.dropout->n_input + l->data.dropout->n_output));
            double upperBound = sqrt(6.0 / (l->data.dropout->n_input + l->data.dropout->n_output));
            for (int i = 0; i < l->data.dropout->n_output; i++) {
                for (int j = 0; j < l->data.dropout->neurons[i].n_input; j++) {
                    double w_init = uniform_distribution(lowerBound, upperBound);
                    l->data.dropout->neurons[i].weights[j] = w_init;
                }
                double b_init = uniform_distribution(lowerBound, upperBound);
                l->data.dropout->neurons[i].bias = b_init;
            }
            break;
        case CONVOLUTIONAL:
            double lowerBound = -sqrt(6.0 / (l->data.convo->n_input + l->data.convo->n_output));
            double upperBound = sqrt(6.0 / (l->data.convo->n_input + l->data.convo->n_output));
            /* waiting to see for when i'll have implemented CNN */
            break;
    }
}

void glorot(Layer *l) {
    switch(l->type) {
        case LINEAR:
            double sigma = sqrt(2.0 / (l->data.linear->n_input + l->data.linear->n_output));
            for (int i = 0; i < l->data.linear->n_output; i++) {
                for (int j = 0; j < l->data.linear->neurons[i].n_input; j++) {
                    double w_init = normal_distribution(0.0, sigma);
                    l->data.linear->neurons[i].weights[j] = w_init;
                }
                l->data.linear->neurons[i].bias = 0.0;
            }
            break;
        case DROPOUT:
            break;
        case CONVOLUTIONAL:
            break;
    }
}

void he(Layer *l) {
    switch(l->type) {
        case LINEAR:
            double sigma = sqrt(2.0 / l->data.linear->n_input);
            for (int i = 0; i < l->data.linear->n_output; i++) {
                for (int j = 0; j < l->data.linear->neurons[i].n_input; j++) {
                    double w_init = normal_distribution(0.0, sigma);
                    l->data.linear->neurons[i].weights[j] = w_init;
                }
                l->data.linear->neurons[i].bias = 0.0;
            }
            break;
        case DROPOUT:
            break;
        case CONVOLUTIONAL:
            break;
    }
}
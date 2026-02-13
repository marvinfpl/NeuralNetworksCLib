#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <math.h>

double relu_act(double input);
/* performs relu function */

double sigmoid_act(double input);
/* performs sigmoid function */

double tanh_act(double input);
/* performs tanh function */

typedef enum {
    RELU,
    SIGMOID,
    TANH,
} ActivationType;

typedef struct {
    ActivationType type;
    double (*forward)(double);
} ActivationFunction;

ActivationFunction create_activation(ActivationType type); 
/* returns an instance of the activation function given the type*/

#endif
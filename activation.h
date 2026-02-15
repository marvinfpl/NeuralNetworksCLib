#ifndef ACTIVATION_C
#define ACTIVATION_C

typedef enum ActivationType {
    RELU,
    TANH,
    SIGMOID,
} ActivationType;

typedef struct ActivationFunction {
    ActivationType type;
    double (*forward)(double);
    double (*backward)(double);
} ActivationFunction;

ActivationFunction *create_activation(ActivationType type);

double forward_relu(double input);

double backward_relu(double input);

double forward_tanh(double input);

double backward_tanh(double input);

double forward_sigmoid(double input);

double backward_sigmoid(double input);

#endif
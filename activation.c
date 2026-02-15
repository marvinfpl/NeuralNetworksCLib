#include <math.h>
#include <activation.h>

ActivationFunction *create_activation(ActivationType type) {
    ActivationFunction *act = malloc(sizeof(ActivationFunction));

    switch (type) {
        case RELU:
            act->forward = forward_relu;
            act->backward = backward_relu;
            break;
        case TANH:
            act->forward = forward_tanh;
            act->backward = backward_tanh;
            break;
        case SIGMOID:
            act->forward = forward_sigmoid;
            act->backward = backward_sigmoid;
            break;
    }

    return act;
}

double forward_relu(double input) {
    if (input > 0.0) {
        return input;
    } else {
        return 0.0;
    }
}

double backward_relu(double input) {
    if (input > 0.0) {
        return 1.0;
    } else {
        return 0.0;
    }
}

double forward_tanh(double input) {
    return tanh(input);
}

double backward_tanh(double input) {
    return 1 - tanh(input) * tanh(input);
}

double forward_sigmoid(double input) {
    return 1.0 / (1.0 + exp(-input));
}

double backward_sigmoid(double input) {
    return forward_sigmoid(input) * (1.0 - forward_sigmoid(input));
}
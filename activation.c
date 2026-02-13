#include <math.h>
#include <activation.h>

double relu_act(double input) {
    if (input > 0.0) {
        return input;
    } else {
        return 0.0;
    }
}

double sigmoid_act(double input) {
    return 1.0 / (1.0 + exp(-input));
}

double tanh_act(double input) {
    return tanh(input);
}

ActivationFunction create_activation(ActivationType type) {
    ActivationFunction act;
    act.type = type;

    switch (type) {
        case RELU:
            act.forward = relu_act;
            break;
        case TANH:
            act.forward = tanh_act;
            break;
        case SIGMOID:
            act.forward = sigmoid_act;
            break;
    }
    return act;
}
#ifndef OPTIMIZER_H
#define OPTIMIZER_H
#include <layers.h>

typedef enum Optimizer {
    GD,
    SGD,
    MOMENTUM_SGD,
    ADAGRAD,
    RMSPROP,
    ADAM,
    ADAMW,
} Optimizer;

typedef struct GradientDescent {
    double learning_rate;
} GradientDescent;

typedef struct StochasticGradientDescent {
    double learning_rate;
    int batch_size;
} StochasticGradientDescent;

typedef struct MomentumSDG {
    double learning_rate;
    int batch_size;
    double momentum;
} MomentumSDG;

typedef struct Adagrad {
    double learning_rate;
    int batch_size;
    double momentum;
    double eps;
    double weights_t;
    double bias_t;
} Adagrad;

typedef struct RMSprop {
    double learning_rate;
    int batch_size;
    double momentum;
    double weights_t;
    double bias_t;
} RMSprop;

typedef struct Adam {
    double learning_rate;
    double beta_1;
    double beta_2;
    double v_t;
    double m_t;
} Adam;

typedef struct AdamW {
    double learning_rate;
    double beta_1;
    double beta_2;
    double v_t;
    double m_t;
    double weight_decay;
} AdamW;

double gradient_descent(Neuron *neuron, double learning_rate);

double stochastic_gradient_descent(Neuron *neuron, double learning_rate, int batch_size);

double momentum_sgd(Neuron *neuron, double learning_rate, int batch_size);

double adagrad(Neuron *neuron, double learning_rate, int batch_size);

double rmsprop(Neuron *neuron, double learning_rate, int batch_size);

double adam(Neuron *neuron, double learning_rate, int batch_size);

double adamw(Neuron *neuron, double learning_rate, int batch_size);

#endif 
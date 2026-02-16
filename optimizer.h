#ifndef OPTIMIZER_H
#define OPTIMIZER_H
#include <layers.h>

typedef enum OptimizerType {
    GD,
    SGD,
    MOMENTUM_SGD,
    ADAGRAD,
    RMSPROP,
    ADAM,
    ADAMW,
} OptimizerType;

typedef struct Optimizer {
    OptimizerType Type;
    void (*forward)(OptimizerType, Neuron*, double*, double);
} Optimizer;

typedef struct GradientDescent {
    double learning_rate;
} GradientDescent;

GradientDescent init_GradientDescent();

void forward_GradientDescent(GradientDescent *gd, Neuron *neuron, double *dw, double dz);

typedef struct StochasticGradientDescent {
    double learning_rate;
    int batch_size;
} StochasticGradientDescent;

StochasticGradientDescent init_StochasticGradientDescent();

void forward_SGD(StochasticGradientDescent *sgd, Neuron *neuron, double *dw, double dz);

typedef struct MomentumSDG {
    double learning_rate;
    int batch_size;
    double *momentum_w;
    double Beta_w;
    double momentum_b;
    double Beta_b;
} MomentumSDG;

MomentumSDG init_MomentumSGD();

void forward_MomentumSGD(MomentumSDG *m, Neuron *neuron, double *dw, double dz);

typedef struct Adagrad {
    double learning_rate;
    int batch_size;
    double eps;
    double *sqr_grad_w;
    double sqr_grad_b;
} Adagrad;

Adagrad init_Adagrad();

void forward_Adagrad(Adagrad *a, Neuron *neuron, double *dw, double dz);

typedef struct RMSprop {
    double learning_rate;
    int batch_size;
    double eps;
    double *momentum_w;
    double beta_w;
    double momentum_b;
    double beta_b;
} RMSprop;

RMSprop init_RMSprop();

void forward_RMSprop(RMSprop *r, Neuron *neuron ,double *dw, double dz);

typedef struct Adam {
    double learning_rate;
    double eps;
    double beta1;
    double *momentum_w;
    double momentum_b;
    double beta2;
    double *momentum_sqr_w;
    double momentum_sqr_b;
    int step;
} Adam;

Adam init_Adam();

void forward_Adam(Adam *a, Neuron *neuron, double *dw, double dz);

typedef struct AdamW {
    double learning_rate;
    double beta_1;
    double beta_2;
    double v_t;
    double m_t;
    double weight_decay;
} AdamW;

#endif 
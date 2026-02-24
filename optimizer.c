#include <optimizer.h>
#include <layers.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

Optimizer *init_optimizer(OptimizerType type, double learning_rate) {
    Optimizer *opt = malloc(sizeof(Optimizer));
    opt->forward = forward_optimizer;
    opt->type = type;

    switch (type) {
        case GD:
            opt->data.gd = init_GradientDescent(learning_rate);
            break;
        case MOMENTUM_SGD:
            opt->data.m = init_MomentumSGD(learning_rate);
            break;
        case ADAGRAD:
            opt->data.ada = init_Adagrad(learning_rate);
            break;
        case RMSPROP:
            opt->data.rms = init_RMSprop(learning_rate);
            break;
        case ADAM:
            opt->data.a = init_Adam(learning_rate);
            break;
        default:
            printf("Error, no valid optimizer has been given.");
            free(opt);
            return NULL;
    }
    return opt;
}

void forward_optimizer(Optimizer *optim, Neuron *neuron, double *dw, double dz) {
    OptimizerType type = optim->type;
    switch (type) {
        case GD:
            forward_GradientDescent(optim->data.gd, neuron, dw, dz);
            break;
        case MOMENTUM_SGD:
            forward_MomentumSGD(optim->data.m, neuron, dw, dz);
            break;
        case ADAGRAD:
            forward_Adagrad(optim->data.ada, neuron, dw, dz);
            break;
        case RMSPROP:
            forward_RMSprop(optim->data.rms, neuron, dw, dz);
            break;
        case ADAM:
            forward_Adam(optim->data.a, neuron, dw, dz);
            break;
    }
}

void forward_GradientDescent(GradientDescent *gd, Neuron *neuron, double *dw, double dz) {
    for (int i = 0; i < neuron->n_input; i++) {
        neuron->weights[i] -= gd->learning_rate * dw[i];
    }
    neuron->bias -= - gd->learning_rate * dz;
}

GradientDescent *init_GradientDescent(double learning_rate) {
    GradientDescent *gd = malloc(sizeof(GradientDescent));
    gd->learning_rate = learning_rate;
    return gd;
}

void forward_MomentumSGD(MomentumSDG *m, Neuron *neuron, double *dw, double dz) {
    for (int i = 0; i < neuron->n_input; i++) {
        m->momentum_w[i] = m->momentum_w[i] * m->Beta_w + (1.0 - m->Beta_w) * dw[i];
        neuron->weights[i] -= m->learning_rate * m->momentum_w[i] * dw[i];
    }
    m->momentum_b = m->momentum_b * m->Beta_b + (1.0 - m->Beta_b) * dz;
    neuron->bias -= m->learning_rate * m->momentum_b * dz;
}

MomentumSDG *init_MomentumSGD(double learning_rate) {
    MomentumSDG *m = malloc(sizeof(MomentumSDG));
    m->learning_rate = learning_rate;
    return m;
}

void forward_Adagrad(Adagrad *a, Neuron *neuron, double *dw, double dz) {
    for (int i = 0; i < neuron->n_input; i++) {
        a->sqr_grad_w[i] += dw[i] * dw[i];
        neuron->weights[i] -= a->learning_rate * dw[i] / (sqrt(a->sqr_grad_w[i] + a->eps));
    }
    a->sqr_grad_b += dz * dz;
    neuron->bias -= a->learning_rate * dz / (sqrt(a->sqr_grad_b + a->eps));
}

Adagrad *init_Adagrad(double learning_rate) {
    Adagrad *a = malloc(sizeof(Adagrad));
    a->learning_rate = learning_rate;
    return a;
}

void forward_RMSprop(RMSprop *r, Neuron *neuron, double *dw, double dz) {
    for (int i = 0; i < neuron->n_input; i++) {
        r->momentum_w[i] = r->beta_w * r->momentum_w[i] + (1.0 - r->beta_w) * dw[i] * dw[i];
        neuron->weights[i] -= r->learning_rate * dw[i] / (sqrt(r->momentum_w[i]) + r->eps);
    }
    r->momentum_b = r->beta_b * r->momentum_b + (1.0 - r->beta_b) * dz * dz;
    neuron->bias -= r->learning_rate * dz / (sqrt(r->momentum_b) + r->eps); 
}

RMSprop *init_RMSprop(double learning_rate) {
    RMSprop *r = malloc(sizeof(RMSprop));
    r->learning_rate = learning_rate;
    return r;
}

void forward_Adam(Adam *a, Neuron *neuron, double *dw, double dz) {
    for (int i = 0; i < neuron->n_input; i++) {
        a->momentum_w[i] = a->beta1 * a->momentum_w[i] + (1.0 - a->beta1) * dw[i];
        a->momentum_sqr_w[i] = a->beta2 * a->momentum_sqr_w[i] + (1.0 - a->beta2) * dw[i] * dw[i];
        double m_corrected_w = a->momentum_w[i] / (1.0 - pow(a->beta1, (double)a->step));
        double v_corrected_w = a->momentum_sqr_w[i] / (1.0 - pow(a->beta2, (double)a->step));
        neuron->weights[i] -= a->learning_rate * m_corrected_w / (sqrt(v_corrected_w) + a->eps);
    }
    a->momentum_b = a->beta1 * a->momentum_b + (1.0 - a->beta1) * dz;
    a->momentum_sqr_b = a->beta2 * a->momentum_sqr_b + (1.0 - a->beta2) * dz * dz;
    double m_corrected_b = a->momentum_b / (1.0 - pow(a->beta1, (double)a->step));
    double v_corrected_b = a->momentum_sqr_b / (1.0 - pow(a->beta2, (double)a->step));
    neuron->bias -= a->learning_rate * m_corrected_b / (sqrt(v_corrected_b) + a->eps);
    a->step++;
}

Adam *init_Adam(double learning_rate) {
    Adam *a = malloc(sizeof(Adam));
    a->learning_rate = learning_rate;
    a->step = 0;
    return a;
}
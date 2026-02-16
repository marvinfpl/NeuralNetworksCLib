#include <optimizer.h>
#include <layers.h>
#include <math.h>

void forward_GradientDescent(GradientDescent *gd, Neuron *neuron, double *dw, double dz) {
    for (int i = 0; i < neuron->n_input; i++) {
        neuron->weights[i] -= gd->learning_rate * dw[i];
    }
    neuron->bias -= - gd->learning_rate * dz;
}

void forward_SGD(StochasticGradientDescent *sgd, Neuron *neuron, double *dw, double dz);

void forward_MomentumSGD(MomentumSDG *m, Neuron *neuron, double *dw, double dz) {
    for (int i = 0; i < neuron->n_input; i++) {
        m->momentum_w[i] = m->momentum_w[i] * m->Beta_w + (1.0 - m->Beta_w) * dw[i];
        neuron->weights[i] -= m->learning_rate * m->momentum_w[i] * dw[i];
    }
    m->momentum_b = m->momentum_b * m->Beta_b + (1.0 - m->Beta_b) * dz;
    neuron->bias -= m->learning_rate * m->momentum_b * dz;
}

void forward_Adagrad(Adagrad *a, Neuron *neuron, double *dw, double dz) {
    for (int i = 0; i < neuron->n_input; i++) {
        a->sqr_grad_w[i] += dw[i] * dw[i];
        neuron->weights[i] -= a->learning_rate * dw[i] / (sqrt(a->sqr_grad_w[i] + a->eps));
    }
    a->sqr_grad_b += dz * dz;
    neuron->bias -= a->learning_rate * dz / (sqrt(a->sqr_grad_b + a->eps));
}

void forward_RMSprop(RMSprop *r, Neuron *neuron, double *dw, double dz) {
    for (int i = 0; i < neuron->n_input; i++) {
        r->momentum_w[i] = r->beta_w * r->momentum_w[i] + (1.0 - r->beta_w) * dw[i] * dw[i];
        neuron->weights[i] -= r->learning_rate * dw[i] / (sqrt(r->momentum_w[i]) + r->eps);
    }
    r->momentum_b = r->beta_b * r->momentum_b + (1.0 - r->beta_b) * dz * dz;
    neuron->bias -= r->learning_rate * dz / (sqrt(r->momentum_b) + r->eps); 
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
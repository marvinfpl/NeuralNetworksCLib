#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <layers.h>
#include <utils.h>

double uniform_distribution(double lowerBound, double upperBound) {
    double u = (double)random() / (double)RAND_MAX;
    return lowerBound + (upperBound - lowerBound + 1) * u;
}

double normal_distribution(double mu, double sigma) {
    double u1 = (double)random() / (double)RAND_MAX;
    double u2 = (double)random() / (double)RAND_MAX;
    double z = sqrt(-2 * log(u1)) * cos(2.0 * M_PI * u2);
    return mu + z * sigma;
}

double *sample_weights_from_gaussian(Neuron *n, double mu, double sigma) {
    double *matrix = malloc(sizeof(double) * n->n_input);
    for (int i = 0; i < n->n_input; i++) {
        double w = normal_distribution(mu, sigma);
        matrix[i] = w;
    }
    return matrix;
}

qr qr_decomposition(double **A) {
    
}
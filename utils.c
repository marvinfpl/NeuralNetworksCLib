#include <time.h>
#include <stdlib.h>
#include <math.h>

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
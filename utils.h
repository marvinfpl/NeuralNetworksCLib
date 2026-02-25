double uniform_distribution(double lowerBound, double upperBound);

double normal_distribution(double mu, double sigma);

double *sample_weights_from_gaussian(Neuron *n, double mu, double sigma);

typedef struct qr {
    double **Q;
    double **R;
}qr;

qr qr_decomposition(double **A);
#include <stdarg.h>
#include <stdlib.h>
#include <layers.h>
#include <activation.h>
#include <optimizer.h>

typedef struct Model {
    Layer **layers;
    Optimizer *optim;
    int size;
} Model;

Model *init_Model(LayerType *type, Optimizer *optim, ActivationFunction *activations, double learning_rate, int *n_input, int *n_output, int size);

typedef enum Initialization {
    Xavier,
    Glorot,
    He,
    Kaiming,
    Orthogonal,
} Initialization;

void init_layers(Initialization i, Layer **layer);

void xavier(Layer *l);

void glorot(Layer *l);

void he(Layer *l);

void orthogonal(Layer *l);
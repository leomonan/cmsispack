#include "predict.h"

#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#ifdef STM32L476xx
#include "fastmath.h"
#endif


Prediction* init_predict() {
    Prediction* pred = malloc(sizeof(Prediction));
    return(pred);
}

void predict(Feature* feats, Prediction* pred) {
    QxOS_DebugPrint("This is not a real classifier!");
}

void free_predict(Prediction* pred) {
    free(pred);
}

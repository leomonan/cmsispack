#pragma GCC optimize ("O3", "fast-math")

#include "fastmath.h"

/* Customized log2f/exp2f based on the Taylor expansions around log2(0.75) and exp2(0) respectively. */

/* [-1/((-0.75)**i * i * math.log(2)) for i in range(1,7)] and math.log(0.75, 2) */
static const float fm_log2f_coeff[] = {
    -1.351001117393963f,
    1.2159010056545667f,
    -1.1399071928011564f,
    1.1399071928011564f,
    -1.2823955919013008f,
    1.9235933878519513f,
    -0.4150374992788438f
};

float fm_log2f(float X)
{
    const float *C = &fm_log2f_coeff[0];
    float Y;
    float F;
    int E;

    if(X <= 0)
        return NAN;

    F = frexpf(fabsf(X), &E);
    /* F ranges in [0.5, 1.0] so approximate around ln2(0.75) */
    F -= 0.75f;
    Y = *C++;
    Y *= F;
    Y += *C++;
    Y *= F;
    Y += *C++;
    Y *= F;
    Y += *C++;
    Y *= F;
    Y += *C++;
    Y *= F;
    Y += *C++;
    Y *= F;
    Y += *C++;
    Y += E;
    return Y;
}

/* [math.log(2)**i / math.factorial(i) for i in range(7)] */
static const float fm_exp2f_coeff[] = {
    0.00015403530393381606f,
    0.0013333558146428441f,
    0.009618129107628477f,
    0.055504108664821576f,
    0.2402265069591007f,
    0.6931471805599453f,
    1.000000000000000f
};

float fm_exp2f(float X)
{
    const float *C = &fm_exp2f_coeff[0];
    float I = floorf(X + 0.5f);
    float F = X - I;
    float Y;

    /* F ranges in [-0.5, 0.5] so approximate around exp2(0) */
    Y = *C++;
    Y *= F;
    Y += *C++;
    Y *= F;
    Y += *C++;
    Y *= F;
    Y += *C++;
    Y *= F;
    Y += *C++;
    Y *= F;
    Y += *C++;
    Y *= F;
    Y += *C++;

    return scalbnf(Y, (int)I);
}

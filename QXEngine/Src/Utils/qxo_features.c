#pragma GCC optimize ("O3", "fast-math")
#include "qxo_features.h"

#include <stdlib.h>
#include <string.h>

#ifdef QXO_FEATURES_ENABLE_PROFILING
extern uint32_t QxOS_GetCycleCount(void);
#endif

#ifdef PYTHON_FEATS
#include <stdio.h>
#endif

#include <limits.h>
#include <math.h>
#ifndef PYTHON_FEATS
/* need access to PRED_SAMPLING_FREQ and RAW_MODEL and ADAPTIVE_BINNING_LEN */
#include "predict.h"
#endif



/* Defined for raw data (i.e. non-feature) models */
#ifndef RAW_MODEL

#if defined (FFT_FEATURES) || defined (MFCC_FEATURES)

#ifndef __arm__
   /* magic define which makes CMSIS-DSP work off-device */
    #define __GNUC_PYTHON__
#endif

#include "arm_math.h"
#include "arm_const_structs.h"

#define IS_POWER_OF_TWO(x) \
  ((x) && !((x) & ((x)-1)))
_Static_assert(IS_POWER_OF_TWO(MAX_FFT_SIG_LEN), "MAX_FFT_SIG_LEN must be power of 2");

#define MAX_FFT_POW_LEN (MAX_FFT_SIG_LEN / 2)

#endif /* FFT_FEATURES */

/* This is the default value for FFT_BINS_RATIO, if it's not been provided by the user. However, the
 * current value of 0.01 will never be used, because this ratio would lead to 100 FFT bins which is
 * higher than the MAX_NUM_POW_BINS below. Leaving this at 0.01f to match the legacy default values
 * throughout the codebase and existing project_config files, which will all be overridden by the
 * MAX_NUM_POW_BINS of 64 below. */
#ifndef FFT_BINS_RATIO
#define FFT_BINS_RATIO 0.01f
#endif

#define MAX_NUM_OCTAVE_BANDS 7
#define MIN_BINS_PER_OCTAVE_BAND 4  /* minimum number of bins for each octave band */

#define MAX_NUM_POW_BINS 64
#define MAX_NUM_PEAKS 50

#define AUTOCORR_RATIO 0.02f
#define MAX_N_AUTOCORR_FEATS 16
#define MIN_N_AUTOCORR_FEATS 3

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

#ifdef FFT_FEATURES_ADAPTIVE_BINNED
#include "fft_adaptive_binning_strategy.h"
#endif

#ifndef USE_FIXEDPOINT
#ifdef MFCC_FEATURES
#include "mfcc_fb.h"
#endif

/*** Optimizations ***/

// Applying unroll-loops to the specified functions below gives us a big win in runtime,
// (up to 10%, though it will depend on what features are enabled)
// and seems to add at most 3KB to the code size.
// See EM-1126 for details.
// For now, enable it for all devices. It can be disabled for a specific
// device or use case, if reducing code size is paramount.
// This attribute only works for GCC (in clang it has no effect),
// but currently all of our embedded device code is built with GCC
#define UNROLL_LOOPS __attribute__((optimize("O3", "fast-math", "unroll-loops")))

// This will overwrite the log2f, logf, expf, and exp2f functions from the C standard library
#include "fastmath.h"

static void set_feats_rawstat_summary(Features* F_ptr, sigval_t* accel, int n);
static void set_feats_autocorr(Features* F_ptr, sigval_t* accel, int n);
static void calculate_autocorrelation(sigval_t* sig, int len_sig, sigval_t* aut, int len_aut);
static void set_feats_fft(Features* F_ptr, sigval_t* accel, int n, float samp_freq, bool is_microphone, fft_feats calc_fft_sub_feats, mfcc_feats calc_mfcc_sub_feats, int adaptive_binning_sensor_channel_key);
static void set_feats_fft_stat_simple(Features* F_ptr, Fft* fft);
static void set_feats_fft_stat_advanced(Features* F_ptr, Fft* fft);
static void set_feats_fft_binned(Features* F_ptr, Fft* fft, int len_sig);
static void set_feats_fft_octave(Features* F_ptr, Fft* fft, int len_sig);
static void set_feats_fft_thirds(Features* F_ptr, Fft* fft, int len_sig);
#ifdef MFCC_FEATURES
static void calculate_mfcc(Features* F_ptr, Fft* fft, mfcc_feats mfcc_sub_feats);
#endif
static void set_feats_peak(Features* F_ptr, sigval_t* sig, int n, float samp_freq);

#if defined(FFT_FEATURES) || defined (MFCC_FEATURES)
#ifdef PYTHON_FEATS
static Fft* init_fft(int n, float samp_freq);
static void free_fft(Fft* fft);
#endif
static void calculate_fft(Fft* fft, sigval_t* accel, int n, float weight);
static void calculate_adaptive_binned_features(Features* F_ptr, Fft* fft, int adaptive_binning_sensor_channel_key);
#endif
/* Helper */
static inline float variance(int n, float nums[n]);
static inline bool is_peak(fval_t prv, fval_t cur, fval_t nxt);
static void set_stats(Stats* stats, sigval_t* sig, int n, bool calc_higher_order);
static inline void mean_subtract(sigval_t* accel, fval_t mean, size_t n);
static float float_sum(float* nums, int lo, int hi);
static inline bool check_fft_true(fft_feats calc_fft_sub_feats);
static inline bool check_mfcc_true(mfcc_feats calc_mfcc_sub_feats);

/* * * * * * * * * * * * *
 *   C O N S T A N T S   *
 * * * * * * * * * * * * */
// FLOAT_BUF is a scratch buffer used for temporary calculations
// used in the following locations with the following memory requirements
//   calculate_axis_divergence(): max size = 6
//   set_feats_autocorr(): max size = MAX_N_AUTOCORR_FEATS
//   set_feats_fft(): max size = MAX_FFT_SIG_LEN
//   set_feats_fft_binned(): max size = MAX_NUM_POW_BINS + 1
//   set_feats_fft_stat_advanced(): max size = 3*MAX_NUM_POW_BINS + 1
//   set_feats_fft_octave(): max_size = 2*log_2(sampling frequency in Hz) + 1
//   set_feats_fft_thirds(): max_size = 3*log_2(sampling frequency in Hz) + 1
//   set_feats_peak(): max size = 2*MAX_NUM_PEAKS
// For any reasonable sampling frequency, the sizes involving log(freq) will be less than
// 3*MAX_NUM_POW_BINS, so we can neglect them when calculating maximum size necessary
// Up-to-date as of 2020-10-22

// If desired, one could add even more complex logic here to minimize FLOAT_BUF_LEN,
// in cases where certain sensors and feature groups are disabled, and we do not need
// the full length allocated below.
#if defined(FFT_FEATURES) || defined (MFCC_FEATURES)
  #define FLOAT_BUF_LEN MAX(MAX(MAX(3*MAX_NUM_POW_BINS+1, MAX_FFT_SIG_LEN+3), 2*MAX_NUM_PEAKS), MAX_N_AUTOCORR_FEATS)
#else
  #define FLOAT_BUF_LEN MAX(2*MAX_NUM_PEAKS, MAX_N_AUTOCORR_FEATS)
#endif
static sigval_t FLOAT_BUF[FLOAT_BUF_LEN];

#endif /* !USE_FIXEDPOINT */

/*** Shared between fixed and floating point ***/

/* * * * * * * * * *
 *   H E L P E R   *
 * * * * * * * * * */

// When calling this code from Python, we want to store each feature alongside its feature name,
// however, when running on-device, we do not want the overhead associated with that.
// Using a macro SET_FEAT allows us to discard the feature-name arguments at compile-time
// when appropriate.
#ifdef PYTHON_FEATS
  #define SET_FEAT(F_ptr, prefix, name, suffix, val) set_feat_w_name(F_ptr, prefix, name, suffix, val)
  static inline void set_feat_w_name(Features* F_ptr, char* prefix, char* name, char* suffix, fval_t val);
#else
  #define SET_FEAT(F_ptr, prefix, name, suffix, val) set_feat(F_ptr, val)
  static inline void set_feat(Features* F_ptr, fval_t val);
#endif

#ifdef PYTHON_FEATS
// g_feature_prefix is set by the externally-facing functions like calc_accel_features_py(),
// and allows us to avoid including prefix as an argument for every single internal function,
// which would be problematic for the use-case of on-device featurization
static char* g_feature_prefix;

/* number of digits in integer x; deliberately avoiding math functions */
static int get_num_digits(int x) {
    char buf[16];
    return snprintf(buf, sizeof(buf), "%d", x);
}

/* converts cur to string with a total of num_digits characters. if cur is not large enough, pads with zeros */
static void set_feat_num(char* str, int cur, int num_digits) {
    sprintf(str, "%0*d", num_digits, cur);
}
#endif

static inline fval_t max(fval_t x, fval_t y) {
    if (x > y) {
        return x;
    }
    return y;
}
static inline fval_t min(fval_t x, fval_t y) {
    if (x < y) {
        return x;
    }
    return y;
}

static inline int sign(fval_t x) {
    if (x > 0) {
        return 1;
    } else if (x < 0) {
        return -1;
    }
    return 0;
}

static inline bool is_peak(fval_t prv, fval_t cur, fval_t nxt) {
    return( (sign(cur - prv) == 1) && (sign(nxt - cur) == -1) );
}

/* returns index of array (with n elements) that contains the maximum value */
static inline int get_imax(fval_t* arr, int n) {
    int imax = 0;
    fval_t arr_max = arr[imax];
    for (int i = 1; i < n; i++) {
        if (arr[i] > arr_max) {
            imax = i;
            arr_max = arr[imax];
        }
    }
    return imax;
}

// The range of this function is from 0 to 31 (excluding when 0 is the input),
// so it's fine to return a signed int
static inline int log2_uint32(uint32_t val) {
    if (val == 0) return INT_MAX;
    int ret = 0;
    while (val > 1) {
        val >>= 1;
        ret++;
    }
    return ret;
}

#ifdef PYTHON_FEATS
/* returns pointer to empty Features object. should be called before get_feats_all()
 * NOTE: Not used in the embedded platform (features are statically allocated).
 */
Features* init_feats_all(int num_feats) {
    Features* F_ptr = malloc(sizeof(Features));
    F_ptr -> num_feats_total = num_feats;
    F_ptr -> num_feats_current = 0;
    F_ptr -> feats = malloc(((size_t)num_feats) * sizeof(Feature));
    return(F_ptr);
}
#else

/* Initialize any global feature data (called once at startup on the embedded platform) */
void startup_feats(void) {
}

/* Initialize a feature structure and make it ready to receive features */
void init_feats(Features* F_ptr) {
    F_ptr->num_feats_current = 0;
#ifdef QXO_FEATURES_ENABLE_PROFILING
    F_ptr->prev_time = QxOS_GetCycleCount();
#endif
}
#endif

/*
 * NOTE: Not used in the embedded platform (features are statically allocated).
 */
void free_feats_all(Features* F_ptr) {
    free(F_ptr -> feats);
    free(F_ptr);
}


/* ACCEL FEATURES */
#if defined(ACCEL_FEATS)
#ifdef PYTHON_FEATS
void calc_accel_features_py(Features* F_ptr, char* prefix, sigval_t* accel, int n, float samp_freq, int accel_type, int axis){
    g_feature_prefix = prefix;
    calc_accel_features(F_ptr, accel, n, samp_freq, accel_type, axis);
}
#endif /* PYTHON_FEATS */

void calc_accel_features(Features* F_ptr, sigval_t* accel, int n, float samp_freq, int accel_type, int axis){
    fft_feats calc_fft_sub_feats = {false, false, false, false, false, false};
    mfcc_feats calc_mfcc_sub_feats = {false, false, false};

    int adaptive_binning_sensor_channel_key = accel_type + axis;

#if defined(ACCEL_STAT_FEATURES) ||  defined(ACCEL_LOWPOWER_STAT_FEATURES) || defined(ACCEL_HIGHSENSITIVE_STAT_FEATURES)
    bool calc_stat_features = true;
#else
    bool calc_stat_features = false;
#endif

#if defined(ACCEL_AUTOCORR_FEATURES) ||  defined(ACCEL_LOWPOWER_AUTOCORR_FEATURES) || defined(ACCEL_HIGHSENSITIVE_AUTOCORR_FEATURES)
    bool calc_autocorr_features = true;
#else
    bool calc_autocorr_features = false;
#endif

#if defined(ACCEL_PEAK_FEATURES) ||  defined(ACCEL_LOWPOWER_PEAK_FEATURES) || defined(ACCEL_HIGHSENSITIVE_PEAK_FEATURES)
    bool calc_peak_features = true;
#else
    bool calc_peak_features = false;
#endif

#if defined(ACCEL_FFT_FEATURES_SIMPLE) || defined(ACCEL_LOWPOWER_FFT_FEATURES_SIMPLE) || defined(ACCEL_HIGHSENSITIVE_FFT_FEATURES_SIMPLE)
    calc_fft_sub_feats.simple = true;
#endif

#if defined(ACCEL_FFT_FEATURES_ADVANCED) || defined(ACCEL_LOWPOWER_FFT_FEATURES_ADVANCED) || defined(ACCEL_HIGHSENSITIVE_FFT_FEATURES_ADVANCED)
    calc_fft_sub_feats.advanced = true;
#endif

#if defined(ACCEL_FFT_FEATURES_LINEARLY_BINNED) || defined(ACCEL_LOWPOWER_FFT_FEATURES_LINEARLY_BINNED) || defined(ACCEL_HIGHSENSITIVE_FFT_FEATURES_LINEARLY_BINNED)
    calc_fft_sub_feats.binned = true;
#endif

#if defined(ACCEL_FFT_FEATURES_OCTAVE_BINNED) || defined(ACCEL_LOWPOWER_FFT_FEATURES_OCTAVE_BINNED) || defined(ACCEL_HIGHSENSITIVE_FFT_FEATURES_OCTAVE_BINNED)
    calc_fft_sub_feats.octave = true;
#endif

#if defined(ACCEL_FFT_FEATURES_THIRDS_BINNED) || defined(ACCEL_LOWPOWER_FFT_FEATURES_THIRDS_BINNED) || defined(ACCEL_HIGHSENSITIVE_FFT_FEATURES_THIRDS_BINNED)
    calc_fft_sub_feats.thirds = true;
#endif
#if defined(ACCEL_FFT_FEATURES_ADAPTIVE_BINNED) || defined(ACCEL_LOWPOWER_FFT_FEATURES_ADAPTIVE_BINNED) || defined(ACCEL_HIGHSENSITIVE_FFT_FEATURES_ADAPTIVE_BINNED)
    calc_fft_sub_feats.adaptive = true;
#endif

    featurize(F_ptr, accel, n, samp_freq, false, false, calc_stat_features, calc_autocorr_features, calc_fft_sub_feats, calc_peak_features, calc_mfcc_sub_feats, adaptive_binning_sensor_channel_key);
}
#endif /* ACCEL_FEATS */


/* GYRO FEATURES */
#if defined(GYRO_FEATS)

#ifdef PYTHON_FEATS
void calc_gyro_features_py(Features* F_ptr, char* prefix, sigval_t* gyro, int n, float samp_freq, int axis){
    g_feature_prefix = prefix;
    calc_gyro_features(F_ptr, gyro, n, samp_freq, axis);
}
#endif /* PYTHON_FEATS */

void calc_gyro_features(Features* F_ptr, sigval_t* gyro, int n, float samp_freq, int axis){
    fft_feats calc_fft_sub_feats = {false, false, false, false, false, false};
    mfcc_feats calc_mfcc_sub_feats = {false, false, false};
    /* Gyro keys start after 4 as 1, 2, 3, 4 are for aceel x, y, z, combined respectively.*/
    int adaptive_binning_sensor_channel_key = 5 + axis;
#ifdef GYRO_STAT_FEATURES
    bool calc_stat_features = true;
#else
    bool calc_stat_features = false;
#endif

#ifdef GYRO_AUTOCORR_FEATURES
    bool calc_autocorr_features = true;
#else
    bool calc_autocorr_features = false;
#endif

#ifdef GYRO_PEAK_FEATURES
    bool calc_peak_features = true;
#else
    bool calc_peak_features = false;
#endif

#ifdef GYRO_FFT_FEATURES_SIMPLE
    calc_fft_sub_feats.simple = true;
#endif

#ifdef GYRO_FFT_FEATURES_ADVANCED
    calc_fft_sub_feats.advanced = true;
#endif

#ifdef GYRO_FFT_FEATURES_LINEARLY_BINNED
    calc_fft_sub_feats.binned = true;
#endif

#ifdef GYRO_FFT_FEATURES_OCTAVE_BINNED
    calc_fft_sub_feats.octave = true;
#endif

#ifdef GYRO_FFT_FEATURES_THIRDS_BINNED
    calc_fft_sub_feats.thirds = true;
#endif
#ifdef GYRO_FFT_FEATURES_ADAPTIVE_BINNED
    calc_fft_sub_feats.adaptive = true;
#endif
    featurize(F_ptr, gyro, n, samp_freq, false, false, calc_stat_features, calc_autocorr_features, calc_fft_sub_feats, calc_peak_features, calc_mfcc_sub_feats, adaptive_binning_sensor_channel_key);
}
#endif /* GYRO_FEATS */


/* MICROPHONE FEATURES */
#if defined(MICROPHONE_FEATS)
#ifdef PYTHON_FEATS
void calc_microphone_features_py(Features* F_ptr, char* prefix, sigval_t* microphone, int n, float samp_freq){
    g_feature_prefix = prefix;
    calc_microphone_features(F_ptr, microphone, n, samp_freq);
}
#endif

void calc_microphone_features(Features* F_ptr, sigval_t* microphone, int n, float samp_freq){
    fft_feats calc_fft_sub_feats = {false, false, false, false, false, false};
    mfcc_feats calc_mfcc_sub_feats = {false, false, false};
    int adaptive_binning_sensor_channel_key = 9;
#ifdef MICROPHONE_STAT_FEATURES
    bool calc_stat_features = true;
#else
    bool calc_stat_features = false;
#endif

#ifdef MICROPHONE_AUTOCORR_FEATURES
    bool calc_autocorr_features = true;
#else
    bool calc_autocorr_features = false;
#endif

#ifdef MICROPHONE_PEAK_FEATURES
    bool calc_peak_features = true;
#else
    bool calc_peak_features = false;
#endif

#ifdef MICROPHONE_MFCC_FEATURES_FILTER
    calc_mfcc_sub_feats.filter = true;
#endif

#ifdef MICROPHONE_MFCC_FEATURES_DELTA
    calc_mfcc_sub_feats.filter = true;
    calc_mfcc_sub_feats.delta_filter = true;
#endif

#ifdef MICROPHONE_MFCC_FEATURES_DELTADELTA
    calc_mfcc_sub_feats.filter = true;
    calc_mfcc_sub_feats.delta_filter = true;
    calc_mfcc_sub_feats.delta_delta_filter = true;
#endif

#ifdef MICROPHONE_FFT_FEATURES_SIMPLE
    calc_fft_sub_feats.simple = true;
#endif

#ifdef MICROPHONE_FFT_FEATURES_ADVANCED
    calc_fft_sub_feats.advanced = true;
#endif

#ifdef MICROPHONE_FFT_FEATURES_LINEARLY_BINNED
    calc_fft_sub_feats.binned = true;
#endif

#ifdef MICROPHONE_FFT_FEATURES_OCTAVE_BINNED
    calc_fft_sub_feats.octave = true;
#endif

#ifdef MICROPHONE_FFT_FEATURES_THIRDS_BINNED
    calc_fft_sub_feats.thirds = true;
#endif
#ifdef MICROPHONE_FFT_FEATURES_ADAPTIVE_BINNED
    calc_fft_sub_feats.adaptive = true;
#endif

    featurize(F_ptr, microphone, n, samp_freq, true, false, calc_stat_features, calc_autocorr_features, calc_fft_sub_feats, calc_peak_features, calc_mfcc_sub_feats, adaptive_binning_sensor_channel_key);
}
#endif /* MICROPHONE_FEATS */


/*
 *    input: F_ptr, pointer to Features struct. will add single feature to Features struct
 *           prefix, string to go before infix
 *           infix, string to go between prefix and suffix. full feature infix will be "{prefix}_{infix}_{suffix}"
 *           suffix, string to go after infix
 *           val, feature value to be set
 */
#ifdef PYTHON_FEATS
static inline void set_feat_w_name(Features* F_ptr, char* prefix, char* infix, char* suffix, fval_t val) {
    int i = F_ptr -> num_feats_current; /* number of features added so far */
    F_ptr -> feats[i].val = val;

#ifdef QXO_FEATURES_ENABLE_PROFILING
    uint32_t time = QxOS_GetCycleCount();
    F_ptr -> feats[i].dt = time - F_ptr->prev_time;
    F_ptr->prev_time = time;
#endif

    /* only add a infix to the feature if featurizing from python */
    if (i >= F_ptr -> num_feats_total) {
        char buf[300];
        char feat_name[200];
        sprintf(feat_name, "%s_%s_%s", prefix, infix, suffix);
        sprintf(buf, "You're making more features (%d) than you said you would (%d). You are trying to make feature %s", i+1, F_ptr -> num_feats_total, feat_name);
        perror(buf);
    }

    char buf[100];
    if (suffix[0] == '\0') {
        sprintf(buf, "%s_%s", prefix, infix);
    } else {
        sprintf(buf, "%s_%s_%s", prefix, infix, suffix);
    }
    strcpy(F_ptr -> feats[i].name, buf);

    /* keep track of how many times this function is called (how many features there are) */
    F_ptr -> num_feats_current = i+1;
}
#else
static inline void set_feat(Features* F_ptr, fval_t val) {
    int i = F_ptr -> num_feats_current; /* number of features added so far */
    F_ptr -> feats[i].val = val;

#ifdef QXO_FEATURES_ENABLE_PROFILING
    uint32_t time = QxOS_GetCycleCount();
    F_ptr -> feats[i].dt = time - F_ptr->prev_time;
    F_ptr->prev_time = time;
#endif

    /* keep track of how many times this function is called (how many features there are) */
    F_ptr -> num_feats_current = i+1;
}
#endif

static inline bool check_fft_true(fft_feats calc_fft_sub_feats) {
    return ((calc_fft_sub_feats.simple == true || calc_fft_sub_feats.advanced == true || calc_fft_sub_feats.octave == true || calc_fft_sub_feats.binned == true || calc_fft_sub_feats.thirds == true));
}

static inline bool check_mfcc_true(mfcc_feats calc_mfcc_sub_feats) {
    return ((calc_mfcc_sub_feats.filter == true || calc_mfcc_sub_feats.delta_filter == true || calc_mfcc_sub_feats.delta_delta_filter == true));
}

#ifndef USE_FIXEDPOINT
/*** Floating-point featurization pipeline ***/
/* * * * * * * * * *
 *   P U B L I C   *
 * * * * * * * * * */

/* Set features for one axis. Call three times - once per axis. */
#if defined(ACCEL_FEATS) || defined(MICROPHONE_FEATS) || defined(GYRO_FEATS) || defined(MAGNO_FEATS)
void featurize(Features* F_ptr, sigval_t* accel, int n, float samp_freq, bool is_microphone, bool is_magno,
                    bool calc_stat_features, bool calc_autocorr_features, fft_feats calc_fft_sub_feats, bool calc_peak_features, mfcc_feats calc_mfcc_sub_feats, int adaptive_binning_sensor_channel_key) {
    fval_t sig_mean = float_sum(accel, 0, n) / (float)n;
    mean_subtract(accel, sig_mean, n);
    if (calc_stat_features){
        if (!(is_magno)){
            SET_FEAT(F_ptr, g_feature_prefix, "rawstat_mean", "", sig_mean);
        }
        set_feats_rawstat_summary(F_ptr, accel, n);
    }
    if (calc_autocorr_features){
        set_feats_autocorr(F_ptr, accel, n);
    }
#if defined(FFT_FEATURES) || defined(MFCC_FEATURES)
    if (check_fft_true(calc_fft_sub_feats) || check_mfcc_true(calc_mfcc_sub_feats)){
        set_feats_fft(F_ptr, accel, n, samp_freq, is_microphone, calc_fft_sub_feats, calc_mfcc_sub_feats, adaptive_binning_sensor_channel_key);
    }
#endif /* mfcc_feature or fft_features */
    if (calc_peak_features){
        set_feats_peak(F_ptr, accel, n, samp_freq);
    }
}
#endif /* ACCEL_FEATS or GYRO_FEATS or MICROPHONE_FEATS*/

/* MAGNO FEATURES */
#if defined(MAGNO_FEATS)
#ifdef PYTHON_FEATS
void calc_magno_features_py(Features* F_ptr, char* prefix, sigval_t* magno, int n, float samp_freq){
    g_feature_prefix = prefix;
    calc_magno_features(F_ptr, magno, n, samp_freq);
}

/* Must be called after calc_magno_features or subtract mean from each axis separately and then call this function. */
void calculate_axis_divergence_py(Features* F_ptr, char* prefix, int n_axes, int n_samp, sigval_t* signal, int* magno_channels) {
    g_feature_prefix = prefix;
#ifdef MAGNO_STAT_FEATURES
    calculate_axis_divergence(F_ptr, n_axes, n_samp, signal, magno_channels);
#endif
}
#endif /* PYTHON_FEATS */

void calculate_axis_divergence(Features* F_ptr, int n_axes, int n_samp, sigval_t* signal, int* magno_channels) {
    int i, j;
    float* one_over_axis_vars = FLOAT_BUF;
    int k = 0;
    for (i = 0; i < n_axes; i++) {
        if (magno_channels[i] == 1) {
            one_over_axis_vars[k] = 0;
            for (j = 0; j < n_samp; j++) {
                one_over_axis_vars[k] += signal[i*n_samp + j] * signal[i*n_samp + j]; /* assume magno has had mean subtracted */
            }

            if (one_over_axis_vars[k] == 0)
                one_over_axis_vars[k] = 1.0;

            one_over_axis_vars[k] = (float)n_samp / one_over_axis_vars[k];
            k += 1;
        }
    }
    float div = 0, div_abs = 0;
    float* samp = FLOAT_BUF + k; /* normalized values for all axes at single sample (in time) */
    for (j = 0; j < n_samp; j++) {
        int l = 0;
        for (i = 0; i < n_axes; i++)
            if (magno_channels[i] == 1) {
                samp[l] = signal[i*n_samp + j] * one_over_axis_vars[l]; /* '*' faster than '/' */
                l += 1;
            }
        div += variance(k, samp);
        for (i = 0; i < k; i++)
            samp[i] = fabsf(samp[i]);
        div_abs += variance(k, samp);
    }
    div /= (float)n_samp;
    div_abs /= (float)n_samp;

    SET_FEAT(F_ptr, g_feature_prefix, "rawstat_axis_divergence", "", div);
    SET_FEAT(F_ptr, g_feature_prefix, "rawstat_axis_divergence", "abs", div_abs);
}


void calc_magno_features(Features* F_ptr, sigval_t* magno, int n, float samp_freq){
    fft_feats calc_fft_sub_feats = {false, false, false, false, false};
    mfcc_feats calc_mfcc_sub_feats = {false, false, false};
#ifdef MAGNO_STAT_FEATURES
    bool calc_stat_features = true;
#else
    bool calc_stat_features = false;
#endif

#ifdef MAGNO_AUTOCORR_FEATURES
    bool calc_autocorr_features = true;
#else
    bool calc_autocorr_features = false;
#endif

#ifdef MAGNO_PEAK_FEATURES
    bool calc_peak_features = true;
#else
    bool calc_peak_features = false;
#endif
    featurize(F_ptr, magno, n, samp_freq, false, true, calc_stat_features, calc_autocorr_features, calc_fft_sub_feats, calc_peak_features, calc_mfcc_sub_feats, -1);
    // Subtract mean from the signal for axis_divergence features, only if rawstat features are defined.
    if (calc_stat_features == true){
        fval_t mean = float_sum(magno, 0, n) / n;
        mean_subtract(magno, mean, n);
    }
}
#endif /* MAGNO_FEATS */


/* Set features for one low frequency signal */
#ifdef LOWFREQ_FEATS
#ifdef PYTHON_FEATS
void calculate_low_freq_features_py(Features* F_ptr, char* prefix, sigval_t* sig, int n) {
    g_feature_prefix = prefix;
    calculate_low_freq_features(F_ptr, sig, n);
}
#endif

void calculate_low_freq_features(Features* F_ptr, sigval_t* sig, int n) {
    fval_t mean = float_sum(sig, 0, n) / n;
    SET_FEAT(F_ptr, g_feature_prefix, "rawstat_mean", "", mean);
    mean_subtract(sig, mean, n);
    set_feats_rawstat_summary(F_ptr, sig, n);
}
#endif /* LOWFREQ_FEATS */


static inline float variance(int n, float nums[n]) {
    int i;
    float v;
    float avg = 0, var = 0;
    for (i = 0; i < n; i++)
        avg += nums[i];
    avg /= (float)n;
    for (i = 0; i < n; i++) {
        v = nums[i] - avg;
        var += v * v;
    }
    return var / (float)n;
}

/* * * * * * * * * * *
 *   P R I V A T E   *
 * * * * * * * * * * */
/*
 *   RAW STATISTICS
 */

static void UNROLL_LOOPS set_feats_rawstat_summary(Features* F_ptr, sigval_t* accel, int n) {
#ifdef PYTHON_FEATS
    char* infix = "rawstat";
#endif

    Stats s;
    set_stats(&s, accel, n, true);

    fval_t max_dev_from_mean = 0;
    for (int i = 0; i < n; i++) {
        max_dev_from_mean = max(max_dev_from_mean, fabsf(accel[i] - s.avg));
    }

    SET_FEAT(F_ptr, g_feature_prefix, infix, "range", s.rng);
    SET_FEAT(F_ptr, g_feature_prefix, infix, "std", s.std);
    SET_FEAT(F_ptr, g_feature_prefix, infix, "skew", s.skew);
    SET_FEAT(F_ptr, g_feature_prefix, infix, "std_by_range", s.std / s.rng);
    SET_FEAT(F_ptr, g_feature_prefix, infix, "max_dev_from_mean", max_dev_from_mean);
}

/*
 *   AUTO-CORRELATION
 */

static void set_feats_autocorr(Features* F_ptr, sigval_t* accel, int n) {
    int num_autocorr_feats = (int) (AUTOCORR_RATIO * n);
    if (num_autocorr_feats > MAX_N_AUTOCORR_FEATS){
        num_autocorr_feats = MAX_N_AUTOCORR_FEATS;
    }
    if (num_autocorr_feats < MIN_N_AUTOCORR_FEATS){
        num_autocorr_feats = MIN_N_AUTOCORR_FEATS;
    }

    sigval_t *aut = FLOAT_BUF; //length will be num_autocorr_feats
    if (aut == NULL) {
        return;
    }

    calculate_autocorrelation(accel, n, aut, num_autocorr_feats);

#ifdef PYTHON_FEATS
    int num_digits = get_num_digits(num_autocorr_feats);
    char feat_num[10];
#endif
    for (int i = 0; i < num_autocorr_feats; i++) {
#ifdef PYTHON_FEATS
        set_feat_num(feat_num, i+1, num_digits);
#endif
        SET_FEAT(F_ptr, g_feature_prefix, "autocorr", feat_num, aut[i]);
    }
}

/* takes the autocorrelation of sig and fills the first len_aut autocorrelation signals into aut */
static void UNROLL_LOOPS calculate_autocorrelation(sigval_t* sig, int len_sig, sigval_t* aut, int len_aut) {
    fval_t cur_aut, max_aut = 0;
    int i, j;
    for (j = 0; j < len_sig; j++) { /* calculate maximum autocorrelation (when signal is entirely overlapped with self) */
        max_aut += sig[j] * sig[j];
    }
    fval_t one_over_max_aut = 1.0 / max_aut; /* do not divide in loop. done for speed */
    /* Case where signal is 0 */
    if (max_aut == 0){
        one_over_max_aut = 0;
    }
    for (i = 1; i <= len_aut; i++) { /* offset of signal with self. do not re-calculate maximum autocorrelation */
        cur_aut = 0;
        for (j = 0; i+j < len_sig; j++) {
            cur_aut += sig[i+j] * sig[j];
        }
        aut[i-1] = cur_aut * one_over_max_aut;
    }
}


/*
 *   FREQUENCY DOMAIN
 */
#if defined (FFT_FEATURES) || defined (MFCC_FEATURES)
static void set_feats_fft(Features* F_ptr, sigval_t* accel, int n, float samp_freq, bool is_microphone, fft_feats calc_fft_sub_feats, mfcc_feats calc_mfcc_sub_feats, int adaptive_binning_sensor_channel_key) {
    Fft* fft;
#ifdef WEIGHTED_AVERAGE
    int fft_sig_len = MAX_FFT_SIG_LEN;
#else
    int power = log2_uint32(n);

    if (1 << power != n) {
        power += 1;
    }
    /*Pads signal to next power of 2, since ARM DSP Library supports only powers of 2 */
    int fft_sig_len = 1 << power;
#endif

#ifdef PYTHON_FEATS
    fft = init_fft(fft_sig_len, samp_freq);
#else
    int len_fft = fft_sig_len / 2;
    // Technically the real FFT has MAX_FFT_POW_LEN+1 components, but we always discard the DC bin
    static fval_t fft_pwr[MAX_FFT_POW_LEN] = { 0 };
    Fft fft_obj = {
        .pwr = fft_pwr,
        .smp_frq = samp_freq,
        .len = len_fft,
    };
    fft = &fft_obj;
#endif

    for (int i = 0; i < fft->len; i++) { // reset to 0
        fft->pwr[i] = 0.0f;
    }

#ifdef WEIGHTED_AVERAGE
    sigval_t* accel_sub = FLOAT_BUF;
    int sub_num = ceil((float)n / (float)fft_sig_len);
    int last_sub_len = fft_sig_len - (sub_num * fft_sig_len - n); // last sub length
    bool padding = false;
    float multiplier;
    float mult_constant = 2.0f/(last_sub_len-1);
    for (int i = 0; i < sub_num; i++) {
        for (int j = 0; j < fft_sig_len; j++) {
            if (i * fft_sig_len + j < n) {
                accel_sub[j] = accel[i * fft_sig_len + j];
            } else {
                if (!padding) {
                    padding = true;
                }
                accel_sub[j] = 0.0f;
            }
        }
        float weight;
        if (i == sub_num - 1) {
            weight = (float)last_sub_len / (float)n;
        } else {
            weight = (float)1 / (float)(i + 1);
        }

        if (padding) {
            for (int k = 0; k < last_sub_len; k++)  {
                multiplier = 1.0f - fabsf(k*mult_constant - 1.0f);
                accel_sub[k] = accel_sub[k]*multiplier;
            }
        }
        calculate_fft(fft, accel_sub, fft_sig_len, weight);
    }
#else
    // accel_padded will be of length fft_sig_len.
    // FLOAT_BUF is of length MAX_FFT_SIG_LEN,
    // so it will always have enough space to hold accel_padded.
    // FLOAT_BUF will be re-used by the set_feats_fft_* functions below, but by that time will be done
    // with accel_padded and don't care if it gets overwritten.
    sigval_t *accel_padded = FLOAT_BUF;
    if (fft_sig_len != n) {
        /* Applying Bartlett window before zero padding */

        float multiplier;
        float mult_constant = 2.0f/(n-1);
        for (int i = 0; i < n; i ++)  {
            multiplier = 1.0f - fabsf(i*mult_constant - 1.0f);
            accel_padded[i] = accel[i]*multiplier;
        }

        int diff = fft_sig_len - n;

        for (int x = n; x < n+diff; x++) {
            accel_padded[x] = 0.0f;
        }

    } else {
        // Even if no padding is necessary, we still need to copy accel, since our FFT function
        // overwrites its input buffer, and we need to be able to re-use accel later
        for (int x = 0; x < n; x++) {
            accel_padded[x] = accel[x];
        }
    }

    calculate_fft(fft, accel_padded, fft_sig_len, 1.0f); /* calculates FFT */
#endif

    if (calc_fft_sub_feats.simple == true) {
        set_feats_fft_stat_simple(F_ptr, fft);
    }

    if (calc_fft_sub_feats.advanced == true) {
        set_feats_fft_stat_advanced(F_ptr, fft);
    }

    if (calc_fft_sub_feats.binned == true) {
        set_feats_fft_binned(F_ptr, fft, fft_sig_len);
    }

    if (calc_fft_sub_feats.octave == true) {
        set_feats_fft_octave(F_ptr, fft, fft_sig_len);
    }

    if (calc_fft_sub_feats.thirds == true) {
        set_feats_fft_thirds(F_ptr, fft, fft_sig_len);
    }
    if (calc_fft_sub_feats.adaptive == true) {
        calculate_adaptive_binned_features(F_ptr, fft, adaptive_binning_sensor_channel_key);
    }

#ifdef MFCC_FEATURES
    if (is_microphone && check_mfcc_true(calc_mfcc_sub_feats)){
        calculate_mfcc(F_ptr, fft, calc_mfcc_sub_feats);
    }
#endif /* MFCC_FEATURES  */

#ifdef PYTHON_FEATS
    free_fft(fft);
#endif
}


/*
*    MFCC features
*/
#ifdef MFCC_FEATURES
static float mel_fb_coeff[N_MEL_FB_COEFF];
static float mfcc_coeff[N_MFCC_COEFF];
static float mfcc_delta[N_MFCC_COEFF];
static float mfcc_delta_delta[N_MFCC_COEFF];

void calc_mfcc_delta(const float *mfcc_coeff, float *delta_coeff_out) {
    /* compute value of first and last coefficient (using 0-padding) */
    delta_coeff_out[0] = mfcc_coeff[1];
    delta_coeff_out[N_MFCC_COEFF-1] = (-1.0 * mfcc_coeff[N_MFCC_COEFF-2]);

    /* compute remaining coefficients */
    size_t i;
    for (i = 1; i < N_MFCC_COEFF-1; i++) {
        delta_coeff_out[i] = (mfcc_coeff[i+1] - mfcc_coeff[i-1]);
    }
}

static void calculate_mfcc(Features* F_ptr, Fft* fft, mfcc_feats calc_mfcc_sub_feats) {
#ifdef PYTHON_FEATS
    int num_digits = get_num_digits(N_MEL_FB_COEFF);
    char bin_num[10];
#endif

    if (calc_mfcc_sub_feats.filter == true) {
        fval_t *pwr = fft->pwr;
        int n = fft -> len;
        int i, j;
        int start_index, pwr_index, length;
        int counted_len = 0;
        for (int i = 0; i < N_MEL_FB_COEFF; i++) {
            mel_fb_coeff[i] = 0.0;

            start_index = fb_nonzero_section_start_index[i];
            length = fb_nonzero_section_length[i];
            for (int j = 0; j < length; j++) {
                pwr_index = start_index - i * n + j;
                mel_fb_coeff[i] += pwr[pwr_index] * fb_nonzero_section_values[counted_len+j];
            }
            if (0.0 == mel_fb_coeff[i]) {
            /* add a small positive constant for numerical stability */
                mel_fb_coeff[i] = MEL_FB_EPSILON;
            }
            mel_fb_coeff[i] = log(mel_fb_coeff[i]);
            counted_len += length;
        }

        for (int i = 0; i < N_MEL_FB_COEFF; i++) {
            mfcc_coeff[i] = 0.0;
            for (int j = 0; j < N_MFCC_COEFF; j++) {
                mfcc_coeff[i] += (mel_fb_coeff[j] * dct[i][j]);
            }
            mfcc_coeff[i] *= 2.0;
        }
        for (int i = 0; i < N_MEL_FB_COEFF; i++) {
#ifdef PYTHON_FEATS
            set_feat_num(bin_num, i+1, num_digits);
#endif
            SET_FEAT(F_ptr, g_feature_prefix, "mfcc_filter", bin_num, mfcc_coeff[i]);
        }
    }

    if (calc_mfcc_sub_feats.delta_filter == true) {
        calc_mfcc_delta(mfcc_coeff, mfcc_delta);
            for (int i = 0; i < N_MFCC_COEFF; i++) {
#ifdef PYTHON_FEATS
                set_feat_num(bin_num, i+1, num_digits);
#endif
                SET_FEAT(F_ptr, g_feature_prefix, "mfccdelta_filter", bin_num, mfcc_delta[i]);
            }
    }

    if (calc_mfcc_sub_feats.delta_delta_filter == true) {
        calc_mfcc_delta(mfcc_delta, mfcc_delta_delta);
        for (int i = 0; i < N_MFCC_COEFF; i++) {
#ifdef PYTHON_FEATS
            set_feat_num(bin_num, i+1, num_digits);
#endif
            SET_FEAT(F_ptr, g_feature_prefix, "mfccdeltadelta_filter", bin_num, mfcc_delta_delta[i]);
        }
    }
}
#endif /* MFCC_FEATS*/


static void set_feats_fft_stat_simple(Features* F_ptr, Fft* fft) {
#ifdef PYTHON_FEATS
    char* infix = "fftpowsimple_stat";
#endif

    Stats s;
    set_stats(&s, fft -> pwr, fft -> len, false);

    /* find index (imax) of frequency with largest power */
    int imax = get_imax(fft -> pwr, fft -> len);
    fval_t scaler = fft->smp_frq / (2*fft->len);

    SET_FEAT(F_ptr, g_feature_prefix, infix, "range", s.rng);
    SET_FEAT(F_ptr, g_feature_prefix, infix, "mean", s.avg);
    SET_FEAT(F_ptr, g_feature_prefix, infix, "mean_by_range", s.avg / s.rng);
    SET_FEAT(F_ptr, g_feature_prefix, infix, "sum_sqrt", sqrtf(s.sum));
    SET_FEAT(F_ptr, g_feature_prefix, infix, "freq_of_max", (imax+1)*scaler);
}

/* calculates FFT specific stats (ex. centroid, rolloff, ...) and linearly binns FFT */
static void UNROLL_LOOPS set_feats_fft_stat_advanced(Features* F_ptr, Fft* fft) {
#ifdef PYTHON_FEATS
    char* infix = "fftpowadvanced_stat";
#endif
    int i, j;

    fval_t *pwr = fft->pwr;
    int n = fft -> len;
    fval_t scaler = fft->smp_frq / (2*fft->len);

    int num_bins = (int)min(1 / FFT_BINS_RATIO, MAX_NUM_POW_BINS);
    num_bins = (int)min(num_bins, n / 2); /* to group at least two fft cofficients */
    /* make bins for power spectrum */
    sigval_t* amp_binned = FLOAT_BUF;
    sigval_t* pwr_binned = FLOAT_BUF + num_bins; /* assumes 2*num_bins <= PRED_NUM_SAMPLES */
    float* bins = FLOAT_BUF + 2*num_bins;
    for (i = 0; i < num_bins; i++) {
        bins[i] = (i * n + num_bins/2) / num_bins;
        amp_binned[i] = 0;
        pwr_binned[i] = 0;
    }
    bins[num_bins] = n; /* last bin ends at fft->len */

    fval_t pwr_sum = 0, cen = 0, amp_sum = 0; /* power sum, power centroid, and binning */
    j = 0; /* bin index */
    for (i = 0; i < n; i++) {
        pwr_sum += pwr[i];
        cen += pwr[i] * (i+1)*scaler;

        /* bin power spectrum */
        if (i >= (int)bins[j+1]) { /* increment to next bin if necessary */
            j += 1;
        }
        fval_t amp = sqrtf(pwr[i]);
        amp_sum += amp;
        amp_binned[j] += amp;
        pwr_binned[j] += pwr[i];
    }
    cen /= pwr_sum;
    /* Case where signal is 0 */
    if (pwr_sum == 0){
       cen = 0;
    }
    fval_t pwr_avg = pwr_sum / (float)n;
    fval_t frq_avg = scaler*(1 + n) / 2.0;

    /* specral flatness from binned power spectrum */
    /* flatness is product of binned power spectrum but do sum of logs for numerical stability */
    fval_t flat = 0;
    bool all_pos = true;
    for (j = 0; j < num_bins; j++) {
        if (amp_binned[j] <= 0) {
            all_pos = false;
        }
    }
    if (all_pos) {
        for (j = 0; j < num_bins; j++) {
            flat += log2f(amp_binned[j]);
        }
        flat /= (fval_t)num_bins;
        flat = exp2f(flat) * (float)n / amp_sum;
    }

    fval_t pwr_cumsum = 0; /* cumulative power sum */
    fval_t frq_var = 0, frq_pwr_cov = 0; /* variance in frequencies, covariance between power and frequency. used for slope calc */
    int irol = -1; /* index of rolloff */
    fval_t pwr_sum_95_perc = 0.95f * pwr_sum;
    for (i = 0; i < n; i++) {
        /* rolloff. freq where cumsum exceeds 95% total sum */
        pwr_cumsum += pwr[i];
        if (pwr_cumsum > pwr_sum_95_perc && irol == -1) {
            irol = i;
        }
        /* slope */
        fval_t v = (i+1)*scaler - frq_avg;
        frq_var += v * v; /* variance in frequency */
        frq_pwr_cov += (i+1)*scaler * pwr[i];
    }
    /* In case signal is 0, irol was not updated above */
    if (irol == -1) {
        irol = 0;
    }
    fval_t roll = scaler*(irol+1);     /* rolloff */
    frq_pwr_cov -= (float)n * frq_avg * pwr_avg;
    fval_t slope = frq_pwr_cov / frq_var; /* slope of FFT. slope in simple linear regression is cov(x,y)/var(x) */
    /* NOTE: should do "frq_var /= (float)n;" and "frq_pwr_cov /= (float)n" but comp faster to not since cancels */

    SET_FEAT(F_ptr, g_feature_prefix, infix, "centroid", cen);
    SET_FEAT(F_ptr, g_feature_prefix, infix, "rolloff", roll);
    SET_FEAT(F_ptr, g_feature_prefix, infix, "flatness", flat);
    SET_FEAT(F_ptr, g_feature_prefix, infix, "slope", slope);
}

static void set_feats_fft_binned(Features* F_ptr, Fft* fft, int len_sig) {
#ifdef PYTHON_FEATS
    char bin_num[10]; /* space used for naming bins */
#endif
    fval_t *pwr = fft->pwr;
    int n = fft -> len;
    int num_bins = (int)min(1 / FFT_BINS_RATIO, MAX_NUM_POW_BINS);
    num_bins = MIN(num_bins, n / 2); /* to group at least two fft cofficients */
    int i;

    /* make linearly spaced bins for power spectrum */
    float* bins = FLOAT_BUF;
    for (i = 0; i < num_bins; i++) {
        bins[i] = i*n / num_bins;
    }
    bins[num_bins] = n;

    /* power spectrum summed into linearly spaced bins */
#ifdef PYTHON_FEATS
    int num_digits = get_num_digits(num_bins);
#endif
    for (i = 0; i < num_bins; i++) {
#ifdef PYTHON_FEATS
        set_feat_num(bin_num, i+1, num_digits);
#endif
        SET_FEAT(F_ptr, g_feature_prefix, "fftpowlinearlybinned_bin", bin_num, float_sum(pwr, (int)bins[i], (int)bins[i+1]));
    }
}


static void calculate_adaptive_binned_features(Features* F_ptr, Fft* fft, int adaptive_binning_sensor_channel_key){
#ifdef PYTHON_FEATS
    char bin_num[10];
#endif
    int *binning_strategy;
    int adaptive_bins_len;
#ifdef ACCEL_X_ADAPTIVE_BINNING_LEN
    if (adaptive_binning_sensor_channel_key == 1){
        binning_strategy = ACCEL_X_ADAPTIVE_BINNING_STRATEGY;
        adaptive_bins_len = ACCEL_X_ADAPTIVE_BINNING_LEN;
    }
#endif
#ifdef ACCEL_Y_ADAPTIVE_BINNING_LEN
    if (adaptive_binning_sensor_channel_key == 2){
        binning_strategy = ACCEL_Y_ADAPTIVE_BINNING_STRATEGY;
        adaptive_bins_len = ACCEL_Y_ADAPTIVE_BINNING_LEN;
    }
#endif
#ifdef ACCEL_Z_ADAPTIVE_BINNING_LEN
    if (adaptive_binning_sensor_channel_key == 3){
        binning_strategy = ACCEL_Z_ADAPTIVE_BINNING_STRATEGY;
        adaptive_bins_len = ACCEL_Z_ADAPTIVE_BINNING_LEN;
    }
#endif
#ifdef ACCEL_W_ADAPTIVE_BINNING_LEN
    if (adaptive_binning_sensor_channel_key == 4){
        binning_strategy = ACCEL_W_ADAPTIVE_BINNING_STRATEGY;
        adaptive_bins_len = ACCEL_W_ADAPTIVE_BINNING_LEN;
    }
#endif
#ifdef GYRO_X_ADAPTIVE_BINNING_LEN
    if (adaptive_binning_sensor_channel_key == 5){
        binning_strategy = GYRO_X_ADAPTIVE_BINNING_STRATEGY;
        adaptive_bins_len = GYRO_X_ADAPTIVE_BINNING_LEN;
    }
#endif
#ifdef GYRO_Y_ADAPTIVE_BINNING_LEN
    if (adaptive_binning_sensor_channel_key == 6){
        binning_strategy = GYRO_Y_ADAPTIVE_BINNING_STRATEGY;
        adaptive_bins_len = GYRO_Y_ADAPTIVE_BINNING_LEN;
    }
#endif
#ifdef GYRO_Z_ADAPTIVE_BINNING_LEN
    if (adaptive_binning_sensor_channel_key == 7){
        binning_strategy = GYRO_Z_ADAPTIVE_BINNING_STRATEGY;
        adaptive_bins_len = GYRO_Z_ADAPTIVE_BINNING_LEN;
    }
#endif
#ifdef GYRO_W_ADAPTIVE_BINNING_LEN
    if (adaptive_binning_sensor_channel_key == 8){
        binning_strategy = GYRO_W_ADAPTIVE_BINNING_STRATEGY;
        adaptive_bins_len = GYRO_W_ADAPTIVE_BINNING_LEN;
    }
#endif
#ifdef MICROPHONE_ADAPTIVE_BINNING_LEN
    if (adaptive_binning_sensor_channel_key == 9){
        binning_strategy = MICROPHONE_ADAPTIVE_BINNING_STRATEGY;
        adaptive_bins_len = MICROPHONE_ADAPTIVE_BINNING_LEN;
    }
#endif
#ifdef ACCEL_LOWPOWER_X_ADAPTIVE_BINNING_LEN
    if (adaptive_binning_sensor_channel_key == 10){
        binning_strategy = ACCEL_LOWPOWER_X_ADAPTIVE_BINNING_STRATEGY;
        adaptive_bins_len = ACCEL_LOWPOWER_X_ADAPTIVE_BINNING_LEN;
    }
#endif
#ifdef ACCEL_LOWPOWER_Y_ADAPTIVE_BINNING_LEN
    if (adaptive_binning_sensor_channel_key == 11){
        binning_strategy = ACCEL_LOWPOWER_Y_ADAPTIVE_BINNING_STRATEGY;
        adaptive_bins_len = ACCEL_LOWPOWER_Y_ADAPTIVE_BINNING_LEN;
    }
#endif
#ifdef ACCEL_LOWPOWER_Z_ADAPTIVE_BINNING_LEN
    if (adaptive_binning_sensor_channel_key == 12){
        binning_strategy = ACCEL_LOWPOWER_Z_ADAPTIVE_BINNING_STRATEGY;
        adaptive_bins_len = ACCEL_LOWPOWER_Z_ADAPTIVE_BINNING_LEN;
    }
#endif
#ifdef ACCEL_LOWPOWER_W_ADAPTIVE_BINNING_LEN
    if (adaptive_binning_sensor_channel_key == 13){
        binning_strategy = ACCEL_LOWPOWER_W_ADAPTIVE_BINNING_STRATEGY;
        adaptive_bins_len = ACCEL_LOWPOWER_W_ADAPTIVE_BINNING_LEN;
    }
#endif
#ifdef ACCEL_HIGHSENSITIVE_X_ADAPTIVE_BINNING_LEN
    if (adaptive_binning_sensor_channel_key == 14){
        binning_strategy = ACCEL_HIGHSENSITIVE_X_ADAPTIVE_BINNING_STRATEGY;
        adaptive_bins_len = ACCEL_HIGHSENSITIVE_X_ADAPTIVE_BINNING_LEN;
    }
#endif
#ifdef ACCEL_HIGHSENSITIVE_Y_ADAPTIVE_BINNING_LEN
    if (adaptive_binning_sensor_channel_key == 15){
        binning_strategy = ACCEL_HIGHSENSITIVE_Y_ADAPTIVE_BINNING_STRATEGY;
        adaptive_bins_len = ACCEL_HIGHSENSITIVE_Y_ADAPTIVE_BINNING_LEN;
    }
#endif
#ifdef ACCEL_HIGHSENSITIVE_Z_ADAPTIVE_BINNING_LEN
    if (adaptive_binning_sensor_channel_key == 16){
        binning_strategy = ACCEL_HIGHSENSITIVE_Z_ADAPTIVE_BINNING_STRATEGY;
        adaptive_bins_len = ACCEL_HIGHSENSITIVE_Z_ADAPTIVE_BINNING_LEN;
    }
#endif
#ifdef ACCEL_HIGHSENSITIVE_W_ADAPTIVE_BINNING_LEN
    if (adaptive_binning_sensor_channel_key == 17){
        binning_strategy = ACCEL_HIGHSENSITIVE_W_ADAPTIVE_BINNING_STRATEGY;
        adaptive_bins_len = ACCEL_HIGHSENSITIVE_W_ADAPTIVE_BINNING_LEN;
    }
#endif

    fval_t *pwr = fft->pwr;
    /* power spectrum summed into adaptive binning calculated offline dduring training. */
    float* adaptive_pwr = FLOAT_BUF;

/* power spectrum summed using adaptive binning strategy. */
#ifdef PYTHON_FEATS
    int num_digits = get_num_digits(adaptive_bins_len - 1);
#endif
    for (int i = 0; i < adaptive_bins_len - 1; i++) {
        adaptive_pwr[i] = float_sum(pwr, (int)binning_strategy[i], (int)binning_strategy[i+1]);
#ifdef PYTHON_FEATS
        set_feat_num(bin_num, i+1, num_digits);
#endif
        SET_FEAT(F_ptr, g_feature_prefix, "fftpowadaptivebinned_bin", bin_num, adaptive_pwr[i]);
    }
    float numerator;
    float denominator;
    float ratio;
#ifdef PYTHON_FEATS
    num_digits = get_num_digits(adaptive_bins_len - 2);
#endif

    for (int i = 0; i < adaptive_bins_len - 2; i++) {
#ifdef PYTHON_FEATS
        set_feat_num(bin_num, i+1, num_digits);
#endif
        numerator = adaptive_pwr[i];
        denominator = adaptive_pwr[i + 1];
        ratio =  numerator / (1 + denominator);
        SET_FEAT(F_ptr, g_feature_prefix, "fftpowadaptivebinned_bin_ratio", bin_num, ratio);
    }
}



static void set_feats_fft_octave(Features* F_ptr, Fft* fft, int len_sig) {
#ifdef PYTHON_FEATS
    char bin_num[10], bin_num2[10];
#endif
    fval_t *pwr = fft->pwr;
    int n = fft -> len;
    /* compute the number of octave bands */
    const int num_oct = min(log2_uint32(n), MAX_NUM_OCTAVE_BANDS);
    /* compute the number of bins for the first octave band (B0) */
    int first_band_num_bins = n >> (num_oct - 1);
    first_band_num_bins = max(first_band_num_bins, MIN_BINS_PER_OCTAVE_BAND);
    int num_bins = num_oct;
    float* oct_bins = FLOAT_BUF;
    int counter = 0;
    float idx = first_band_num_bins;
    /* compute the upper bin for each octave band using Bi = (2 ** i) * B0 */
    for (int i = 0; i < num_bins + 1; i++) {  /* num_bins + 1 to include the first bin with idx 0*/
        if (i == 0){
             oct_bins[counter] = 0;
             counter += 1;
        } else {
            if (idx > n){
                break;  /* avoid index out of bounds */
            }
            if (idx > oct_bins[counter-1]){
                oct_bins[counter] = idx;
                counter += 1;
            }
            idx *= 2;
        }
    }
    /* include last bin */
    if (oct_bins[counter - 1] < n){
        oct_bins[counter] = n;
        counter += 1;
    }

    /* counter points to next empty bin, number of available bins is one less */
    num_bins = counter - 1;

    /* power spectrum summed into log spaced bins (octaves) */
    float* pwr_oct = FLOAT_BUF + num_bins + 1;
#ifdef PYTHON_FEATS
    int num_digits = get_num_digits(num_bins);
#endif
    for (int i = 0; i < num_bins; i++) {
        pwr_oct[i] = float_sum(pwr, (int)oct_bins[i], (int)oct_bins[i+1]);
    /* Removing FFT OOCTAVE BIN features as they are subset of Thirds features. More analysis can be found in EM-1049. */

#ifdef PYTHON_FEATS
        set_feat_num(bin_num, i+1, num_digits);
#endif
    /*
        SET_FEAT(F_ptr, g_feature_prefix, "fftpowoctavebinned_bin", bin_num, pwr_oct[i]);
    */
    }

    /* log ratio of octaves */
#ifdef PYTHON_FEATS
    char buf[10+10+6]; /* str buffer space to hold "{bin_num[i]}_over_{bin_num[j]}" */
#endif

    fval_t v;
    for (int i = 0; i < num_bins; i++) {
        for (int j = 0; j < i; j++) {
            if (pwr_oct[i] <= 0.0 || pwr_oct[j] <= 0.0) { /* should not take log of neg or 0 */
                v = 0;
            } else {
                v = logf(pwr_oct[i] / pwr_oct[j]);
            }
#ifdef PYTHON_FEATS
            set_feat_num(bin_num, i+1, num_digits);
            set_feat_num(bin_num2, j+1, num_digits);
            sprintf(buf, "%s_over_%s", bin_num, bin_num2);
#endif
            SET_FEAT(F_ptr, g_feature_prefix, "fftpowoctavebinned_logratio", buf, v);
        }
    }
}


static void set_feats_fft_thirds(Features* F_ptr, Fft* fft, int len_sig) {
#ifdef PYTHON_FEATS
    char bin_num[10];
#endif

   /* make log spaced bins for thirds of power spectrum */
    fval_t *pwr = fft->pwr;
    int n = fft -> len;
    /* compute the number of octave bands */
    const int num_oct = min(log2_uint32(n), MAX_NUM_OCTAVE_BANDS);
    /* compute the number of bins for the first octave band (B0) */
    int first_band_num_bins = n >> (num_oct - 1);
    first_band_num_bins = max(first_band_num_bins, MIN_BINS_PER_OCTAVE_BAND);
    int num_bins = 3 * num_oct;

    sigval_t* trd_bins = FLOAT_BUF;

    int counter = 0;
    float idx_unrounded = first_band_num_bins;
    const fval_t cube_root_two = 1.25992104f; // 2.0**(1.0/3.0)
    /* compute the upper bin for each thirds band using Bi = (2 ** (i/3)) * B0 */
    for (int i = 0; i < num_bins; i++) {
        if (i == 0){
            trd_bins[counter] = 0.0f;
            counter += 1;
        } else {
            float idx = roundf(idx_unrounded);
            if (idx > n){ /* avoid index out of bounds */
                break;
            }
            if (idx > trd_bins[counter - 1]){
                trd_bins[counter] = idx;
                counter += 1;
            }
            idx_unrounded *= cube_root_two;
        }
    }

    if (trd_bins[counter - 1] < n){
        trd_bins[counter] = n;
        counter += 1;
    }
    num_bins = counter;
    fval_t v;

    /* power spectrum summed into log spaced bins (thirds) */
#ifdef PYTHON_FEATS
    int num_digits = get_num_digits(num_bins);
#endif
    for (int i = 0; i < num_bins - 1; i++) {
        v = float_sum(pwr, trd_bins[i], trd_bins[i+1]);
#ifdef PYTHON_FEATS
        set_feat_num(bin_num, i+1, num_digits);
#endif
        SET_FEAT(F_ptr, g_feature_prefix, "fftpowthirdsbinned_bin", bin_num, v);
    }
}


#ifdef PYTHON_FEATS
static Fft* init_fft(int n, float samp_freq) {
    int len_fft = n/2;
    Fft* fft = malloc(sizeof(Fft));
    fft -> pwr = malloc(len_fft * sizeof(fval_t));
    fft -> len = len_fft;
    fft -> smp_frq = samp_freq;
    return(fft);
}
#endif

/*
   This function does the work of turning an improper CFFT (i.e. a CFFT performed on a real signal)
   into a proper RFFT, and also the work
   of converting the RFFT into a power spectrum in one step, without using any additional memory.
   All details are in EM-1238

   cfft: output of CFFT. Sequence of sig_len/2 complex numbers. array length = sig_len
   twiddle: precomputed table with array length = sig_len/2.
      The twiddle factor is i * exp(-i*2*pi*k/N), and the real and imag parts are interleaved.
      So, twiddle[2*k] = sin(2*pi*k/N) and twiddle[2*k + 1] = cos(2*pi*k/N)
      Technically, elements 0 and 1 are not used, but we keep them for sanity.
   sig_len: must be a power of 2, and 32 <= sig_len <= 4096

   The output will overwrite cfft in the following format:
   cfft[0]: power of the 0th FFT bin (i.e. square of the DC component)
   cfft[1]: power of the nyquist frequency (=sig_len/2'th) bin
   cfft[2*n]: power of nth frequency bin, for 0 < n < sig_len/2
   All odd indices except for 1 are unused
 */
static void calc_power_in_place(float* cfft, const float* twiddle, int sig_len) {
    const int N = sig_len;
    // First handle special cases (DC and nyquist and N/4)
    float a = cfft[0] + cfft[1];
    float b = cfft[0] - cfft[1];
    cfft[0] = a*a;
    cfft[1] = b*b;
    cfft[N/2] = cfft[N/2]*cfft[N/2] + cfft[N/2 + 1]*cfft[N/2 + 1];

    for (int k=1; k<N/4; k++) {
        // Calculate the RFFT power for both the k'th bin and the m'th bin, where m = N/2 - k
        float cr = cfft[2*k]; // Real part of k'th CFFT coefficient
        float ci = cfft[2*k + 1];  // Imag part of k'th CFFT coefficient
        float dr = cfft[N - 2*k]; // Real part of m'th CFFT coefficient
        float di = cfft[N - 2*k + 1]; // Imag part of m'th CFFT coefficient

        // For k, the twiddle factor is i * exp(-i*2*pi*k/N), so that the
        // real part is sin(2*pi*k/N), imag part is cos(2*pi*k/N)
        float tr = twiddle[2*k];
        float ti = twiddle[2*k + 1];
        // For m, the twiddle is -i * exp(i*2*pi*k/N), so the real part is tr, and the imag part is -ti

        float real_diff = dr - cr;
        float imag_sum = ci + di;
        float real_sum = cr + dr;
        float imag_diff = ci - di;
        float temp1 = ti*real_diff - tr*imag_sum;
        float temp2 = tr*real_diff + ti*imag_sum;

        float real_k = real_sum + temp2;
        float imag_k = imag_diff + temp1;

        float real_m = real_sum - temp2;
        float imag_m = temp1 - imag_diff;

        cfft[2*k] = 0.25f*(real_k*real_k + imag_k*imag_k); // Magnitude squared of k'th RFFT coefficient
        cfft[N - 2*k] = 0.25f*(real_m*real_m + imag_m*imag_m); // Magnitude squared of m'th RFFT coefficient
    }
}


/*
 * Calculate FFT coefficients with running weight average.
 * Parameter fft contains old fft values.
 * new fft values = (1 - weight) * old fft values + weight * new fft values
 */
static void UNROLL_LOOPS calculate_fft(Fft* fft, sigval_t* accel, int n, float weight) {
	// fft->len tells us the length of arrays in Fft struct
	// Doesn't make sense unless signal_length is twice the fft->len
	if (n != 2*fft->len) {
		return;
	}

	fval_t* pwr = fft -> pwr;
	float samp_freq = fft -> smp_frq;

	fval_t one_over_samp_freq = 1.0f / samp_freq;

    int i;

    /** EM-1237: Storing only required FFT_TABLES
    Previously, we were using arm_rfft_fast_init_f32() to create FFT tables for our project.
    Doing so was leading to all possible FFT tables being stored, i.e. even if MAX_FFT_SIG_LEN=64,
    we stored FFT tables for 32, 64, 128, 256, 512, 1024, 2048, 4096 leading to large binary files.
    Now, we are directly referencing tables inside ifdef's such that only tables with <=MAX_FFT_SIG_LEN are stored.
    <= is because we only know the max limit variable MAX_FFT_SIG_LEN. int n might be lower than MAX_FFT_SIG_LEN
    **/
    // Since arm_cfft_f32 does calculation in-place, we implemented calc_power_in_place(), so that
    // we don't need to use any additional memory in this function.
    switch(n) {
#if MAX_FFT_SIG_LEN >= 32
    case 32:
        arm_cfft_f32(&arm_cfft_sR_f32_len16, accel, 0, 1);
        // Prefix these twiddle tables with qx_ so that we don't get confused with the
        // CMSIS-DSP twiddle tables. They are similar to the CMSIS-DSP tables, except only half the length
        // TODO: For cleanliness, these tables should be defined in a separate file.
        // This is slightly complicated because adding a new C file requires changing
        // a lot of packaging/build systems/python scripts.
        static const float qx_rfft_twiddle_32[] = {0.0f, 1.0f, 0.19509032201612825f, 0.9807852804032304f, 0.3826834323650898f, 0.9238795325112867f, 0.5555702330196022f, 0.8314696123025452f, 0.7071067811865475f, 0.7071067811865476f, 0.8314696123025451f, 0.5555702330196023f, 0.9238795325112867f, 0.38268343236508984f, 0.9807852804032304f, 0.1950903220161283f};
        calc_power_in_place(accel, &qx_rfft_twiddle_32, n);
        break;
#endif

#if MAX_FFT_SIG_LEN >= 64
    case 64:
        arm_cfft_f32(&arm_cfft_sR_f32_len32, accel, 0, 1);
        static const float qx_rfft_twiddle_64[] = {0.0f, 1.0f, 0.0980171403295606f, 0.9951847266721969f, 0.19509032201612825f, 0.9807852804032304f, 0.29028467725446233f, 0.9569403357322088f, 0.3826834323650898f, 0.9238795325112867f, 0.47139673682599764f, 0.881921264348355f, 0.5555702330196022f, 0.8314696123025452f, 0.6343932841636455f, 0.773010453362737f, 0.7071067811865475f, 0.7071067811865476f, 0.7730104533627369f, 0.6343932841636456f, 0.8314696123025451f, 0.5555702330196023f, 0.8819212643483549f, 0.4713967368259978f, 0.9238795325112867f, 0.38268343236508984f, 0.9569403357322089f, 0.2902846772544623f, 0.9807852804032304f, 0.1950903220161283f, 0.9951847266721968f, 0.09801714032956077f};
        calc_power_in_place(accel, &qx_rfft_twiddle_64, n);
        break;
#endif

#if MAX_FFT_SIG_LEN >= 128
    case 128:
        arm_cfft_f32(&arm_cfft_sR_f32_len64, accel, 0, 1);
        static const float qx_rfft_twiddle_128[] = {0.0f, 1.0f, 0.049067674327418015f, 0.9987954562051724f, 0.0980171403295606f, 0.9951847266721969f, 0.14673047445536175f, 0.989176509964781f, 0.19509032201612825f, 0.9807852804032304f, 0.24298017990326387f, 0.970031253194544f, 0.29028467725446233f, 0.9569403357322088f, 0.33688985339222005f, 0.9415440651830208f, 0.3826834323650898f, 0.9238795325112867f, 0.4275550934302821f, 0.9039892931234433f, 0.47139673682599764f, 0.881921264348355f, 0.5141027441932217f, 0.8577286100002721f, 0.5555702330196022f, 0.8314696123025452f, 0.5956993044924334f, 0.8032075314806449f, 0.6343932841636455f, 0.773010453362737f, 0.6715589548470183f, 0.7409511253549592f, 0.7071067811865475f, 0.7071067811865476f, 0.740951125354959f, 0.6715589548470184f, 0.7730104533627369f, 0.6343932841636456f, 0.8032075314806448f, 0.5956993044924335f, 0.8314696123025451f, 0.5555702330196023f, 0.8577286100002721f, 0.5141027441932218f, 0.8819212643483549f, 0.4713967368259978f, 0.9039892931234433f, 0.4275550934302822f, 0.9238795325112867f, 0.38268343236508984f, 0.9415440651830208f, 0.33688985339222005f, 0.9569403357322089f, 0.2902846772544623f, 0.970031253194544f, 0.24298017990326398f, 0.9807852804032304f, 0.1950903220161283f, 0.989176509964781f, 0.14673047445536175f, 0.9951847266721968f, 0.09801714032956077f, 0.9987954562051724f, 0.04906767432741813f};
        calc_power_in_place(accel, &qx_rfft_twiddle_128, n);
        break;
#endif

#if MAX_FFT_SIG_LEN >= 256
    case 256:
        arm_cfft_f32(&arm_cfft_sR_f32_len128, accel, 0, 1);
        static const float qx_rfft_twiddle_256[] = {0.0f, 1.0f, 0.024541228522912288f, 0.9996988186962042f, 0.049067674327418015f, 0.9987954562051724f, 0.07356456359966743f, 0.9972904566786902f, 0.0980171403295606f, 0.9951847266721969f, 0.1224106751992162f, 0.99247953459871f, 0.14673047445536175f, 0.989176509964781f, 0.17096188876030122f, 0.9852776423889412f, 0.19509032201612825f, 0.9807852804032304f, 0.2191012401568698f, 0.9757021300385286f, 0.24298017990326387f, 0.970031253194544f, 0.26671275747489837f, 0.9637760657954398f, 0.29028467725446233f, 0.9569403357322088f, 0.3136817403988915f, 0.9495281805930367f, 0.33688985339222005f, 0.9415440651830208f, 0.3598950365349881f, 0.932992798834739f, 0.3826834323650898f, 0.9238795325112867f, 0.40524131400498986f, 0.9142097557035307f, 0.4275550934302821f, 0.9039892931234433f, 0.44961132965460654f, 0.8932243011955153f, 0.47139673682599764f, 0.881921264348355f, 0.49289819222978404f, 0.8700869911087115f, 0.5141027441932217f, 0.8577286100002721f, 0.5349976198870972f, 0.8448535652497071f, 0.5555702330196022f, 0.8314696123025452f, 0.5758081914178453f, 0.8175848131515837f, 0.5956993044924334f, 0.8032075314806449f, 0.6152315905806268f, 0.7883464276266063f, 0.6343932841636455f, 0.773010453362737f, 0.6531728429537768f, 0.7572088465064846f, 0.6715589548470183f, 0.7409511253549592f, 0.6895405447370668f, 0.724247082951467f, 0.7071067811865475f, 0.7071067811865476f, 0.7242470829514669f, 0.6895405447370669f, 0.740951125354959f, 0.6715589548470184f, 0.7572088465064845f, 0.6531728429537769f, 0.7730104533627369f, 0.6343932841636456f, 0.7883464276266062f, 0.6152315905806269f, 0.8032075314806448f, 0.5956993044924335f, 0.8175848131515837f, 0.5758081914178454f, 0.8314696123025451f, 0.5555702330196023f, 0.844853565249707f, 0.5349976198870974f, 0.8577286100002721f, 0.5141027441932218f, 0.8700869911087113f, 0.49289819222978415f, 0.8819212643483549f, 0.4713967368259978f, 0.8932243011955153f, 0.4496113296546066f, 0.9039892931234433f, 0.4275550934302822f, 0.9142097557035307f, 0.4052413140049898f, 0.9238795325112867f, 0.38268343236508984f, 0.9329927988347388f, 0.3598950365349883f, 0.9415440651830208f, 0.33688985339222005f, 0.9495281805930367f, 0.3136817403988915f, 0.9569403357322089f, 0.2902846772544623f, 0.9637760657954398f, 0.2667127574748984f, 0.970031253194544f, 0.24298017990326398f, 0.9757021300385286f, 0.21910124015686977f, 0.9807852804032304f, 0.1950903220161283f, 0.9852776423889412f, 0.17096188876030136f, 0.989176509964781f, 0.14673047445536175f, 0.99247953459871f, 0.12241067519921628f, 0.9951847266721968f, 0.09801714032956077f, 0.9972904566786902f, 0.07356456359966745f, 0.9987954562051724f, 0.04906767432741813f, 0.9996988186962042f, 0.024541228522912267f};
        calc_power_in_place(accel, &qx_rfft_twiddle_256, n);
        break;
#endif

#if MAX_FFT_SIG_LEN >= 512
    case 512:
        arm_cfft_f32(&arm_cfft_sR_f32_len256, accel, 0, 1);
        static const float qx_rfft_twiddle_512[] = {0.0f, 1.0f, 0.012271538285719925f, 0.9999247018391445f, 0.024541228522912288f, 0.9996988186962042f, 0.03680722294135883f, 0.9993223845883495f, 0.049067674327418015f, 0.9987954562051724f, 0.06132073630220858f, 0.9981181129001492f, 0.07356456359966743f, 0.9972904566786902f, 0.0857973123444399f, 0.996312612182778f, 0.0980171403295606f, 0.9951847266721969f, 0.11022220729388306f, 0.9939069700023561f, 0.1224106751992162f, 0.99247953459871f, 0.13458070850712617f, 0.99090263542778f, 0.14673047445536175f, 0.989176509964781f, 0.15885814333386145f, 0.9873014181578584f, 0.17096188876030122f, 0.9852776423889412f, 0.18303988795514095f, 0.9831054874312163f, 0.19509032201612825f, 0.9807852804032304f, 0.20711137619221856f, 0.9783173707196277f, 0.2191012401568698f, 0.9757021300385286f, 0.2310581082806711f, 0.9729399522055602f, 0.24298017990326387f, 0.970031253194544f, 0.25486565960451457f, 0.9669764710448521f, 0.26671275747489837f, 0.9637760657954398f, 0.27851968938505306f, 0.9604305194155658f, 0.29028467725446233f, 0.9569403357322088f, 0.3020059493192281f, 0.9533060403541939f, 0.3136817403988915f, 0.9495281805930367f, 0.3253102921622629f, 0.9456073253805213f, 0.33688985339222005f, 0.9415440651830208f, 0.34841868024943456f, 0.937339011912575f, 0.3598950365349881f, 0.932992798834739f, 0.37131719395183754f, 0.9285060804732156f, 0.3826834323650898f, 0.9238795325112867f, 0.3939920400610481f, 0.9191138516900578f, 0.40524131400498986f, 0.9142097557035307f, 0.41642956009763715f, 0.9091679830905224f, 0.4275550934302821f, 0.9039892931234433f, 0.43861623853852766f, 0.8986744656939538f, 0.44961132965460654f, 0.8932243011955153f, 0.46053871095824f, 0.8876396204028539f, 0.47139673682599764f, 0.881921264348355f, 0.4821837720791227f, 0.8760700941954066f, 0.49289819222978404f, 0.8700869911087115f, 0.5035383837257176f, 0.8639728561215867f, 0.5141027441932217f, 0.8577286100002721f, 0.524589682678469f, 0.8513551931052652f, 0.5349976198870972f, 0.8448535652497071f, 0.5453249884220465f, 0.838224705554838f, 0.5555702330196022f, 0.8314696123025452f, 0.5657318107836131f, 0.8245893027850253f, 0.5758081914178453f, 0.8175848131515837f, 0.5857978574564389f, 0.8104571982525948f, 0.5956993044924334f, 0.8032075314806449f, 0.6055110414043255f, 0.7958369046088836f, 0.6152315905806268f, 0.7883464276266063f, 0.6248594881423863f, 0.7807372285720945f, 0.6343932841636455f, 0.773010453362737f, 0.6438315428897914f, 0.765167265622459f, 0.6531728429537768f, 0.7572088465064846f, 0.6624157775901718f, 0.7491363945234594f, 0.6715589548470183f, 0.7409511253549592f, 0.680600997795453f, 0.7326542716724128f, 0.6895405447370668f, 0.724247082951467f, 0.6983762494089729f, 0.7157308252838186f, 0.7071067811865475f, 0.7071067811865476f, 0.7157308252838186f, 0.6983762494089729f, 0.7242470829514669f, 0.6895405447370669f, 0.7326542716724127f, 0.6806009977954531f, 0.740951125354959f, 0.6715589548470184f, 0.7491363945234593f, 0.6624157775901718f, 0.7572088465064845f, 0.6531728429537769f, 0.7651672656224588f, 0.6438315428897915f, 0.7730104533627369f, 0.6343932841636456f, 0.7807372285720944f, 0.6248594881423865f, 0.7883464276266062f, 0.6152315905806269f, 0.7958369046088835f, 0.6055110414043255f, 0.8032075314806448f, 0.5956993044924335f, 0.8104571982525948f, 0.5857978574564389f, 0.8175848131515837f, 0.5758081914178454f, 0.8245893027850253f, 0.5657318107836132f, 0.8314696123025451f, 0.5555702330196023f, 0.838224705554838f, 0.5453249884220466f, 0.844853565249707f, 0.5349976198870974f, 0.8513551931052652f, 0.524589682678469f, 0.8577286100002721f, 0.5141027441932218f, 0.8639728561215867f, 0.5035383837257176f, 0.8700869911087113f, 0.49289819222978415f, 0.8760700941954065f, 0.4821837720791229f, 0.8819212643483549f, 0.4713967368259978f, 0.8876396204028539f, 0.46053871095824f, 0.8932243011955153f, 0.4496113296546066f, 0.8986744656939538f, 0.4386162385385277f, 0.9039892931234433f, 0.4275550934302822f, 0.9091679830905224f, 0.4164295600976373f, 0.9142097557035307f, 0.4052413140049898f, 0.9191138516900578f, 0.3939920400610481f, 0.9238795325112867f, 0.38268343236508984f, 0.9285060804732155f, 0.3713171939518376f, 0.9329927988347388f, 0.3598950365349883f, 0.937339011912575f, 0.3484186802494345f, 0.9415440651830208f, 0.33688985339222005f, 0.9456073253805213f, 0.325310292162263f, 0.9495281805930367f, 0.3136817403988915f, 0.9533060403541938f, 0.3020059493192282f, 0.9569403357322089f, 0.2902846772544623f, 0.9604305194155658f, 0.27851968938505306f, 0.9637760657954398f, 0.2667127574748984f, 0.9669764710448521f, 0.2548656596045146f, 0.970031253194544f, 0.24298017990326398f, 0.9729399522055601f, 0.23105810828067125f, 0.9757021300385286f, 0.21910124015686977f, 0.9783173707196277f, 0.20711137619221856f, 0.9807852804032304f, 0.1950903220161283f, 0.9831054874312163f, 0.18303988795514103f, 0.9852776423889412f, 0.17096188876030136f, 0.9873014181578584f, 0.1588581433338614f, 0.989176509964781f, 0.14673047445536175f, 0.99090263542778f, 0.13458070850712622f, 0.99247953459871f, 0.12241067519921628f, 0.9939069700023561f, 0.11022220729388318f, 0.9951847266721968f, 0.09801714032956077f, 0.996312612182778f, 0.08579731234443988f, 0.9972904566786902f, 0.07356456359966745f, 0.9981181129001492f, 0.061320736302208655f, 0.9987954562051724f, 0.04906767432741813f, 0.9993223845883495f, 0.03680722294135899f, 0.9996988186962042f, 0.024541228522912267f, 0.9999247018391445f, 0.012271538285719944f};
        calc_power_in_place(accel, &qx_rfft_twiddle_512, n);
        break;
#endif

#if MAX_FFT_SIG_LEN >= 1024
    case 1024:
        arm_cfft_f32(&arm_cfft_sR_f32_len512, accel, 0, 1);
        static const float qx_rfft_twiddle_1024[] = {0.0f, 1.0f, 0.006135884649154475f, 0.9999811752826011f, 0.012271538285719925f, 0.9999247018391445f, 0.01840672990580482f, 0.9998305817958234f, 0.024541228522912288f, 0.9996988186962042f, 0.030674803176636626f, 0.9995294175010931f, 0.03680722294135883f, 0.9993223845883495f, 0.04293825693494082f, 0.9990777277526454f, 0.049067674327418015f, 0.9987954562051724f, 0.05519524434968994f, 0.9984755805732948f, 0.06132073630220858f, 0.9981181129001492f, 0.06744391956366405f, 0.9977230666441916f, 0.07356456359966743f, 0.9972904566786902f, 0.07968243797143013f, 0.9968202992911657f, 0.0857973123444399f, 0.996312612182778f, 0.09190895649713272f, 0.9957674144676598f, 0.0980171403295606f, 0.9951847266721969f, 0.10412163387205459f, 0.9945645707342554f, 0.11022220729388306f, 0.9939069700023561f, 0.11631863091190475f, 0.9932119492347945f, 0.1224106751992162f, 0.99247953459871f, 0.12849811079379317f, 0.9917097536690995f, 0.13458070850712617f, 0.99090263542778f, 0.1406582393328492f, 0.9900582102622971f, 0.14673047445536175f, 0.989176509964781f, 0.15279718525844344f, 0.9882575677307495f, 0.15885814333386145f, 0.9873014181578584f, 0.16491312048996992f, 0.9863080972445987f, 0.17096188876030122f, 0.9852776423889412f, 0.17700422041214875f, 0.984210092386929f, 0.18303988795514095f, 0.9831054874312163f, 0.1890686641498062f, 0.9819638691095552f, 0.19509032201612825f, 0.9807852804032304f, 0.2011046348420919f, 0.9795697656854405f, 0.20711137619221856f, 0.9783173707196277f, 0.21311031991609136f, 0.9770281426577544f, 0.2191012401568698f, 0.9757021300385286f, 0.22508391135979283f, 0.9743393827855759f, 0.2310581082806711f, 0.9729399522055602f, 0.2370236059943672f, 0.9715038909862518f, 0.24298017990326387f, 0.970031253194544f, 0.24892760574572015f, 0.9685220942744173f, 0.25486565960451457f, 0.9669764710448521f, 0.2607941179152755f, 0.9653944416976894f, 0.26671275747489837f, 0.9637760657954398f, 0.272621355449949f, 0.9621214042690416f, 0.27851968938505306f, 0.9604305194155658f, 0.2844075372112719f, 0.9587034748958716f, 0.29028467725446233f, 0.9569403357322088f, 0.2961508882436238f, 0.9551411683057707f, 0.3020059493192281f, 0.9533060403541939f, 0.30784964004153487f, 0.9514350209690083f, 0.3136817403988915f, 0.9495281805930367f, 0.3195020308160157f, 0.9475855910177411f, 0.3253102921622629f, 0.9456073253805213f, 0.33110630575987643f, 0.9435934581619604f, 0.33688985339222005f, 0.9415440651830208f, 0.3426607173119944f, 0.9394592236021899f, 0.34841868024943456f, 0.937339011912575f, 0.35416352542049034f, 0.9351835099389476f, 0.3598950365349881f, 0.932992798834739f, 0.36561299780477385f, 0.9307669610789837f, 0.37131719395183754f, 0.9285060804732156f, 0.37700741021641826f, 0.9262102421383114f, 0.3826834323650898f, 0.9238795325112867f, 0.38834504669882625f, 0.9215140393420419f, 0.3939920400610481f, 0.9191138516900578f, 0.3996241998456468f, 0.9166790599210427f, 0.40524131400498986f, 0.9142097557035307f, 0.4108431710579039f, 0.9117060320054299f, 0.41642956009763715f, 0.9091679830905224f, 0.4220002707997997f, 0.9065957045149153f, 0.4275550934302821f, 0.9039892931234433f, 0.43309381885315196f, 0.901348847046022f, 0.43861623853852766f, 0.8986744656939538f, 0.4441221445704292f, 0.8959662497561852f, 0.44961132965460654f, 0.8932243011955153f, 0.45508358712634384f, 0.8904487232447579f, 0.46053871095824f, 0.8876396204028539f, 0.4659764957679662f, 0.8847970984309378f, 0.47139673682599764f, 0.881921264348355f, 0.4767992300633221f, 0.8790122264286335f, 0.4821837720791227f, 0.8760700941954066f, 0.487550160148436f, 0.8730949784182901f, 0.49289819222978404f, 0.8700869911087115f, 0.49822766697278187f, 0.8670462455156926f, 0.5035383837257176f, 0.8639728561215867f, 0.508830142543107f, 0.8608669386377673f, 0.5141027441932217f, 0.8577286100002721f, 0.5193559901655896f, 0.8545579883654005f, 0.524589682678469f, 0.8513551931052652f, 0.5298036246862946f, 0.8481203448032972f, 0.5349976198870972f, 0.8448535652497071f, 0.5401714727298929f, 0.8415549774368984f, 0.5453249884220465f, 0.838224705554838f, 0.5504579729366048f, 0.83486287498638f, 0.5555702330196022f, 0.8314696123025452f, 0.560661576197336f, 0.8280450452577558f, 0.5657318107836131f, 0.8245893027850253f, 0.5707807458869673f, 0.8211025149911046f, 0.5758081914178453f, 0.8175848131515837f, 0.5808139580957645f, 0.8140363297059484f, 0.5857978574564389f, 0.8104571982525948f, 0.5907597018588742f, 0.8068475535437993f, 0.5956993044924334f, 0.8032075314806449f, 0.600616479383869f, 0.799537269107905f, 0.6055110414043255f, 0.7958369046088836f, 0.6103828062763095f, 0.7921065773002124f, 0.6152315905806268f, 0.7883464276266063f, 0.6200572117632891f, 0.7845565971555752f, 0.6248594881423863f, 0.7807372285720945f, 0.629638238914927f, 0.7768884656732324f, 0.6343932841636455f, 0.773010453362737f, 0.6391244448637757f, 0.7691033376455797f, 0.6438315428897914f, 0.765167265622459f, 0.6485144010221124f, 0.7612023854842618f, 0.6531728429537768f, 0.7572088465064846f, 0.6578066932970786f, 0.7531867990436125f, 0.6624157775901718f, 0.7491363945234594f, 0.6669999223036375f, 0.745057785441466f, 0.6715589548470183f, 0.7409511253549592f, 0.6760927035753159f, 0.7368165688773698f, 0.680600997795453f, 0.7326542716724128f, 0.6850836677727004f, 0.7284643904482252f, 0.6895405447370668f, 0.724247082951467f, 0.6939714608896539f, 0.7200025079613817f, 0.6983762494089729f, 0.7157308252838186f, 0.7027547444572253f, 0.7114321957452164f, 0.7071067811865475f, 0.7071067811865476f, 0.7114321957452163f, 0.7027547444572254f, 0.7157308252838186f, 0.6983762494089729f, 0.7200025079613817f, 0.693971460889654f, 0.7242470829514669f, 0.6895405447370669f, 0.7284643904482252f, 0.6850836677727005f, 0.7326542716724127f, 0.6806009977954531f, 0.7368165688773698f, 0.676092703575316f, 0.740951125354959f, 0.6715589548470184f, 0.745057785441466f, 0.6669999223036376f, 0.7491363945234593f, 0.6624157775901718f, 0.7531867990436124f, 0.6578066932970787f, 0.7572088465064845f, 0.6531728429537769f, 0.7612023854842618f, 0.6485144010221126f, 0.7651672656224588f, 0.6438315428897915f, 0.7691033376455796f, 0.6391244448637758f, 0.7730104533627369f, 0.6343932841636456f, 0.7768884656732324f, 0.6296382389149271f, 0.7807372285720944f, 0.6248594881423865f, 0.7845565971555752f, 0.6200572117632892f, 0.7883464276266062f, 0.6152315905806269f, 0.7921065773002123f, 0.6103828062763095f, 0.7958369046088835f, 0.6055110414043255f, 0.799537269107905f, 0.600616479383869f, 0.8032075314806448f, 0.5956993044924335f, 0.8068475535437992f, 0.5907597018588743f, 0.8104571982525948f, 0.5857978574564389f, 0.8140363297059483f, 0.5808139580957646f, 0.8175848131515837f, 0.5758081914178454f, 0.8211025149911046f, 0.5707807458869674f, 0.8245893027850253f, 0.5657318107836132f, 0.8280450452577557f, 0.560661576197336f, 0.8314696123025451f, 0.5555702330196023f, 0.83486287498638f, 0.5504579729366049f, 0.838224705554838f, 0.5453249884220466f, 0.8415549774368983f, 0.540171472729893f, 0.844853565249707f, 0.5349976198870974f, 0.8481203448032971f, 0.5298036246862948f, 0.8513551931052652f, 0.524589682678469f, 0.8545579883654005f, 0.5193559901655896f, 0.8577286100002721f, 0.5141027441932218f, 0.8608669386377672f, 0.5088301425431071f, 0.8639728561215867f, 0.5035383837257176f, 0.8670462455156926f, 0.4982276669727819f, 0.8700869911087113f, 0.49289819222978415f, 0.87309497841829f, 0.4875501601484361f, 0.8760700941954065f, 0.4821837720791229f, 0.8790122264286334f, 0.47679923006332225f, 0.8819212643483549f, 0.4713967368259978f, 0.8847970984309378f, 0.4659764957679661f, 0.8876396204028539f, 0.46053871095824f, 0.8904487232447579f, 0.45508358712634384f, 0.8932243011955153f, 0.4496113296546066f, 0.8959662497561851f, 0.44412214457042926f, 0.8986744656939538f, 0.4386162385385277f, 0.901348847046022f, 0.433093818853152f, 0.9039892931234433f, 0.4275550934302822f, 0.9065957045149153f, 0.4220002707997998f, 0.9091679830905224f, 0.4164295600976373f, 0.9117060320054299f, 0.4108431710579039f, 0.9142097557035307f, 0.4052413140049898f, 0.9166790599210427f, 0.3996241998456468f, 0.9191138516900578f, 0.3939920400610481f, 0.9215140393420419f, 0.3883450466988263f, 0.9238795325112867f, 0.38268343236508984f, 0.9262102421383114f, 0.3770074102164183f, 0.9285060804732155f, 0.3713171939518376f, 0.9307669610789837f, 0.36561299780477396f, 0.9329927988347388f, 0.3598950365349883f, 0.9351835099389475f, 0.3541635254204905f, 0.937339011912575f, 0.3484186802494345f, 0.9394592236021899f, 0.3426607173119944f, 0.9415440651830208f, 0.33688985339222005f, 0.9435934581619604f, 0.33110630575987643f, 0.9456073253805213f, 0.325310292162263f, 0.9475855910177411f, 0.31950203081601575f, 0.9495281805930367f, 0.3136817403988915f, 0.9514350209690083f, 0.307849640041535f, 0.9533060403541938f, 0.3020059493192282f, 0.9551411683057707f, 0.29615088824362396f, 0.9569403357322089f, 0.2902846772544623f, 0.9587034748958716f, 0.2844075372112718f, 0.9604305194155658f, 0.27851968938505306f, 0.9621214042690416f, 0.272621355449949f, 0.9637760657954398f, 0.2667127574748984f, 0.9653944416976894f, 0.26079411791527557f, 0.9669764710448521f, 0.2548656596045146f, 0.9685220942744173f, 0.24892760574572026f, 0.970031253194544f, 0.24298017990326398f, 0.9715038909862518f, 0.23702360599436734f, 0.9729399522055601f, 0.23105810828067125f, 0.9743393827855759f, 0.22508391135979278f, 0.9757021300385286f, 0.21910124015686977f, 0.9770281426577544f, 0.21311031991609136f, 0.9783173707196277f, 0.20711137619221856f, 0.9795697656854405f, 0.20110463484209193f, 0.9807852804032304f, 0.1950903220161283f, 0.9819638691095552f, 0.18906866414980628f, 0.9831054874312163f, 0.18303988795514103f, 0.984210092386929f, 0.17700422041214886f, 0.9852776423889412f, 0.17096188876030136f, 0.9863080972445987f, 0.16491312048997006f, 0.9873014181578584f, 0.1588581433338614f, 0.9882575677307495f, 0.1527971852584434f, 0.989176509964781f, 0.14673047445536175f, 0.9900582102622971f, 0.14065823933284924f, 0.99090263542778f, 0.13458070850712622f, 0.9917097536690995f, 0.12849811079379322f, 0.99247953459871f, 0.12241067519921628f, 0.9932119492347945f, 0.11631863091190486f, 0.9939069700023561f, 0.11022220729388318f, 0.9945645707342554f, 0.10412163387205473f, 0.9951847266721968f, 0.09801714032956077f, 0.9957674144676598f, 0.0919089564971327f, 0.996312612182778f, 0.08579731234443988f, 0.9968202992911657f, 0.07968243797143013f, 0.9972904566786902f, 0.07356456359966745f, 0.9977230666441916f, 0.0674439195636641f, 0.9981181129001492f, 0.061320736302208655f, 0.9984755805732948f, 0.05519524434969003f, 0.9987954562051724f, 0.04906767432741813f, 0.9990777277526454f, 0.04293825693494096f, 0.9993223845883495f, 0.03680722294135899f, 0.9995294175010931f, 0.030674803176636584f, 0.9996988186962042f, 0.024541228522912267f, 0.9998305817958234f, 0.01840672990580482f, 0.9999247018391445f, 0.012271538285719944f, 0.9999811752826011f, 0.006135884649154516f};
        calc_power_in_place(accel, &qx_rfft_twiddle_1024, n);
        break;
#endif

#if MAX_FFT_SIG_LEN >= 2048
    case 2048:
        arm_cfft_f32(&arm_cfft_sR_f32_len1024, accel, 0, 1);
        static const float qx_rfft_twiddle_2048[] = {0.0f, 1.0f, 0.003067956762965976f, 0.9999952938095762f, 0.006135884649154475f, 0.9999811752826011f, 0.00920375478205982f, 0.9999576445519639f, 0.012271538285719925f, 0.9999247018391445f, 0.0153392062849881f, 0.9998823474542126f, 0.01840672990580482f, 0.9998305817958234f, 0.021474080275469508f, 0.9997694053512153f, 0.024541228522912288f, 0.9996988186962042f, 0.02760814577896574f, 0.9996188224951786f, 0.030674803176636626f, 0.9995294175010931f, 0.03374117185137758f, 0.9994306045554617f, 0.03680722294135883f, 0.9993223845883495f, 0.03987292758773981f, 0.9992047586183639f, 0.04293825693494082f, 0.9990777277526454f, 0.04600318213091462f, 0.9989412931868569f, 0.049067674327418015f, 0.9987954562051724f, 0.052131704680283324f, 0.9986402181802653f, 0.05519524434968994f, 0.9984755805732948f, 0.05825826450043575f, 0.9983015449338929f, 0.06132073630220858f, 0.9981181129001492f, 0.06438263092985747f, 0.997925286198596f, 0.06744391956366405f, 0.9977230666441916f, 0.07050457338961386f, 0.9975114561403035f, 0.07356456359966743f, 0.9972904566786902f, 0.07662386139203149f, 0.997060070339483f, 0.07968243797143013f, 0.9968202992911657f, 0.08274026454937569f, 0.9965711457905548f, 0.0857973123444399f, 0.996312612182778f, 0.0888535525825246f, 0.996044700901252f, 0.09190895649713272f, 0.9957674144676598f, 0.09496349532963899f, 0.9954807554919269f, 0.0980171403295606f, 0.9951847266721969f, 0.10106986275482782f, 0.9948793307948056f, 0.10412163387205459f, 0.9945645707342554f, 0.10717242495680884f, 0.9942404494531879f, 0.11022220729388306f, 0.9939069700023561f, 0.11327095217756435f, 0.9935641355205953f, 0.11631863091190475f, 0.9932119492347945f, 0.11936521481099135f, 0.9928504144598651f, 0.1224106751992162f, 0.99247953459871f, 0.12545498341154623f, 0.9920993131421918f, 0.12849811079379317f, 0.9917097536690995f, 0.13154002870288312f, 0.9913108598461154f, 0.13458070850712617f, 0.99090263542778f, 0.13762012158648604f, 0.9904850842564571f, 0.1406582393328492f, 0.9900582102622971f, 0.14369503315029447f, 0.9896220174632009f, 0.14673047445536175f, 0.989176509964781f, 0.1497645346773215f, 0.9887216919603238f, 0.15279718525844344f, 0.9882575677307495f, 0.15582839765426523f, 0.9877841416445722f, 0.15885814333386145f, 0.9873014181578584f, 0.16188639378011183f, 0.9868094018141855f, 0.16491312048996992f, 0.9863080972445987f, 0.16793829497473117f, 0.9857975091675675f, 0.17096188876030122f, 0.9852776423889412f, 0.17398387338746382f, 0.9847485018019042f, 0.17700422041214875f, 0.984210092386929f, 0.18002290140569951f, 0.9836624192117303f, 0.18303988795514095f, 0.9831054874312163f, 0.18605515166344663f, 0.9825393022874412f, 0.1890686641498062f, 0.9819638691095552f, 0.19208039704989244f, 0.9813791933137546f, 0.19509032201612825f, 0.9807852804032304f, 0.19809841071795356f, 0.9801821359681174f, 0.2011046348420919f, 0.9795697656854405f, 0.20410896609281687f, 0.9789481753190622f, 0.20711137619221856f, 0.9783173707196277f, 0.2101118368804696f, 0.9776773578245099f, 0.21311031991609136f, 0.9770281426577544f, 0.21610679707621952f, 0.9763697313300211f, 0.2191012401568698f, 0.9757021300385286f, 0.2220936209732035f, 0.9750253450669941f, 0.22508391135979283f, 0.9743393827855759f, 0.22807208317088573f, 0.973644249650812f, 0.2310581082806711f, 0.9729399522055602f, 0.23404195858354343f, 0.9722264970789363f, 0.2370236059943672f, 0.9715038909862518f, 0.2400030224487415f, 0.9707721407289504f, 0.24298017990326387f, 0.970031253194544f, 0.2459550503357946f, 0.9692812353565485f, 0.24892760574572015f, 0.9685220942744173f, 0.25189781815421697f, 0.9677538370934755f, 0.25486565960451457f, 0.9669764710448521f, 0.257831102162159f, 0.9661900034454125f, 0.2607941179152755f, 0.9653944416976894f, 0.26375467897483135f, 0.9645897932898128f, 0.26671275747489837f, 0.9637760657954398f, 0.2696683255729151f, 0.9629532668736839f, 0.272621355449949f, 0.9621214042690416f, 0.27557181931095814f, 0.9612804858113206f, 0.27851968938505306f, 0.9604305194155658f, 0.28146493792575794f, 0.9595715130819845f, 0.2844075372112719f, 0.9587034748958716f, 0.2873474595447295f, 0.9578264130275329f, 0.29028467725446233f, 0.9569403357322088f, 0.29321916269425863f, 0.9560452513499964f, 0.2961508882436238f, 0.9551411683057707f, 0.2990798263080405f, 0.9542280951091057f, 0.3020059493192281f, 0.9533060403541939f, 0.3049292297354024f, 0.9523750127197659f, 0.30784964004153487f, 0.9514350209690083f, 0.3107671527496115f, 0.9504860739494817f, 0.3136817403988915f, 0.9495281805930367f, 0.31659337555616585f, 0.9485613499157303f, 0.3195020308160157f, 0.9475855910177411f, 0.32240767880106985f, 0.9466009130832835f, 0.3253102921622629f, 0.9456073253805213f, 0.3282098435790925f, 0.9446048372614803f, 0.33110630575987643f, 0.9435934581619604f, 0.3339996514420094f, 0.9425731976014469f, 0.33688985339222005f, 0.9415440651830208f, 0.33977688440682685f, 0.9405060705932683f, 0.3426607173119944f, 0.9394592236021899f, 0.3455413249639891f, 0.9384035340631081f, 0.34841868024943456f, 0.937339011912575f, 0.3512927560855671f, 0.9362656671702783f, 0.35416352542049034f, 0.9351835099389476f, 0.35703096123343f, 0.934092550404259f, 0.3598950365349881f, 0.932992798834739f, 0.3627557243673972f, 0.9318842655816681f, 0.36561299780477385f, 0.9307669610789837f, 0.3684668299533723f, 0.9296408958431812f, 0.37131719395183754f, 0.9285060804732156f, 0.37416406297145793f, 0.9273625256504011f, 0.37700741021641826f, 0.9262102421383114f, 0.37984720892405116f, 0.9250492407826776f, 0.3826834323650898f, 0.9238795325112867f, 0.38551605384391885f, 0.9227011283338786f, 0.38834504669882625f, 0.9215140393420419f, 0.39117038430225387f, 0.9203182767091106f, 0.3939920400610481f, 0.9191138516900578f, 0.3968099874167103f, 0.9179007756213905f, 0.3996241998456468f, 0.9166790599210427f, 0.40243465085941843f, 0.9154487160882678f, 0.40524131400498986f, 0.9142097557035307f, 0.4080441628649787f, 0.9129621904283982f, 0.4108431710579039f, 0.9117060320054299f, 0.41363831223843456f, 0.9104412922580672f, 0.41642956009763715f, 0.9091679830905224f, 0.4192168883632239f, 0.9078861164876663f, 0.4220002707997997f, 0.9065957045149153f, 0.4247796812091088f, 0.9052967593181188f, 0.4275550934302821f, 0.9039892931234433f, 0.4303264813400826f, 0.9026733182372588f, 0.43309381885315196f, 0.901348847046022f, 0.4358570799222555f, 0.9000158920161603f, 0.43861623853852766f, 0.8986744656939538f, 0.44137126873171667f, 0.8973245807054183f, 0.4441221445704292f, 0.8959662497561852f, 0.44686884016237416f, 0.8945994856313827f, 0.44961132965460654f, 0.8932243011955153f, 0.4523495872337709f, 0.8918407093923427f, 0.45508358712634384f, 0.8904487232447579f, 0.45781330359887723f, 0.8890483558546646f, 0.46053871095824f, 0.8876396204028539f, 0.46325978355186015f, 0.8862225301488806f, 0.4659764957679662f, 0.8847970984309378f, 0.46868882203582796f, 0.8833633386657316f, 0.47139673682599764f, 0.881921264348355f, 0.47410021465054997f, 0.8804708890521608f, 0.4767992300633221f, 0.8790122264286335f, 0.479493757660153f, 0.8775452902072614f, 0.4821837720791227f, 0.8760700941954066f, 0.48486924800079106f, 0.8745866522781762f, 0.487550160148436f, 0.8730949784182901f, 0.49022648328829116f, 0.871595086655951f, 0.49289819222978404f, 0.8700869911087115f, 0.49556526182577254f, 0.8685707059713409f, 0.49822766697278187f, 0.8670462455156926f, 0.5008853826112407f, 0.8655136240905691f, 0.5035383837257176f, 0.8639728561215867f, 0.5061866453451552f, 0.8624239561110405f, 0.508830142543107f, 0.8608669386377673f, 0.5114688504379703f, 0.8593018183570085f, 0.5141027441932217f, 0.8577286100002721f, 0.5167317990176499f, 0.8561473283751945f, 0.5193559901655896f, 0.8545579883654005f, 0.5219752929371544f, 0.8529606049303636f, 0.524589682678469f, 0.8513551931052652f, 0.5271991347819014f, 0.8497417680008525f, 0.5298036246862946f, 0.8481203448032972f, 0.5324031278771979f, 0.8464909387740521f, 0.5349976198870972f, 0.8448535652497071f, 0.5375870762956454f, 0.8432082396418454f, 0.5401714727298929f, 0.8415549774368984f, 0.5427507848645159f, 0.8398937941959995f, 0.5453249884220465f, 0.838224705554838f, 0.5478940591731002f, 0.836547727223512f, 0.5504579729366048f, 0.83486287498638f, 0.5530167055800275f, 0.8331701647019132f, 0.5555702330196022f, 0.8314696123025452f, 0.5581185312205561f, 0.829761233794523f, 0.560661576197336f, 0.8280450452577558f, 0.5631993440138341f, 0.8263210628456635f, 0.5657318107836131f, 0.8245893027850253f, 0.5682589526701315f, 0.8228497813758264f, 0.5707807458869673f, 0.8211025149911046f, 0.5732971666980422f, 0.819347520076797f, 0.5758081914178453f, 0.8175848131515837f, 0.5783137964116556f, 0.8158144108067338f, 0.5808139580957645f, 0.8140363297059484f, 0.5833086529376983f, 0.8122505865852039f, 0.5857978574564389f, 0.8104571982525948f, 0.5882815482226452f, 0.808656181588175f, 0.5907597018588742f, 0.8068475535437993f, 0.5932322950397998f, 0.8050313311429637f, 0.5956993044924334f, 0.8032075314806449f, 0.5981607069963424f, 0.8013761717231402f, 0.600616479383869f, 0.799537269107905f, 0.6030665985403482f, 0.7976908409433912f, 0.6055110414043255f, 0.7958369046088836f, 0.6079497849677736f, 0.7939754775543372f, 0.6103828062763095f, 0.7921065773002124f, 0.6128100824294097f, 0.79023022143731f, 0.6152315905806268f, 0.7883464276266063f, 0.6176473079378039f, 0.7864552135990858f, 0.6200572117632891f, 0.7845565971555752f, 0.62246127937415f, 0.7826505961665757f, 0.6248594881423863f, 0.7807372285720945f, 0.6272518154951441f, 0.778816512381476f, 0.629638238914927f, 0.7768884656732324f, 0.6320187359398091f, 0.7749531065948738f, 0.6343932841636455f, 0.773010453362737f, 0.6367618612362842f, 0.7710605242618138f, 0.6391244448637757f, 0.7691033376455797f, 0.6414810128085832f, 0.7671389119358204f, 0.6438315428897914f, 0.765167265622459f, 0.6461760129833163f, 0.7631884172633814f, 0.6485144010221124f, 0.7612023854842618f, 0.650846684996381f, 0.759209188978388f, 0.6531728429537768f, 0.7572088465064846f, 0.6554928529996153f, 0.7552013768965365f, 0.6578066932970786f, 0.7531867990436125f, 0.6601143420674205f, 0.7511651319096864f, 0.6624157775901718f, 0.7491363945234594f, 0.6647109782033448f, 0.7471006059801801f, 0.6669999223036375f, 0.745057785441466f, 0.669282588346636f, 0.7430079521351217f, 0.6715589548470183f, 0.7409511253549592f, 0.673829000378756f, 0.7388873244606151f, 0.6760927035753159f, 0.7368165688773698f, 0.6783500431298615f, 0.7347388780959634f, 0.680600997795453f, 0.7326542716724128f, 0.6828455463852481f, 0.7305627692278276f, 0.6850836677727004f, 0.7284643904482252f, 0.687315340891759f, 0.726359155084346f, 0.6895405447370668f, 0.724247082951467f, 0.6917592583641577f, 0.7221281939292153f, 0.6939714608896539f, 0.7200025079613817f, 0.696177131491463f, 0.7178700450557317f, 0.6983762494089729f, 0.7157308252838186f, 0.7005687939432483f, 0.7135848687807935f, 0.7027547444572253f, 0.7114321957452164f, 0.7049340803759049f, 0.7092728264388657f, 0.7071067811865475f, 0.7071067811865476f, 0.7092728264388656f, 0.704934080375905f, 0.7114321957452163f, 0.7027547444572254f, 0.7135848687807935f, 0.7005687939432484f, 0.7157308252838186f, 0.6983762494089729f, 0.7178700450557316f, 0.6961771314914631f, 0.7200025079613817f, 0.693971460889654f, 0.7221281939292152f, 0.6917592583641579f, 0.7242470829514669f, 0.6895405447370669f, 0.7263591550843459f, 0.6873153408917592f, 0.7284643904482252f, 0.6850836677727005f, 0.7305627692278276f, 0.6828455463852481f, 0.7326542716724127f, 0.6806009977954531f, 0.7347388780959634f, 0.6783500431298616f, 0.7368165688773698f, 0.676092703575316f, 0.7388873244606151f, 0.6738290003787561f, 0.740951125354959f, 0.6715589548470184f, 0.7430079521351216f, 0.6692825883466361f, 0.745057785441466f, 0.6669999223036376f, 0.7471006059801801f, 0.6647109782033449f, 0.7491363945234593f, 0.6624157775901718f, 0.7511651319096864f, 0.6601143420674206f, 0.7531867990436124f, 0.6578066932970787f, 0.7552013768965364f, 0.6554928529996155f, 0.7572088465064845f, 0.6531728429537769f, 0.759209188978388f, 0.650846684996381f, 0.7612023854842618f, 0.6485144010221126f, 0.7631884172633813f, 0.6461760129833164f, 0.7651672656224588f, 0.6438315428897915f, 0.7671389119358203f, 0.6414810128085832f, 0.7691033376455796f, 0.6391244448637758f, 0.7710605242618137f, 0.6367618612362843f, 0.7730104533627369f, 0.6343932841636456f, 0.7749531065948738f, 0.6320187359398091f, 0.7768884656732324f, 0.6296382389149271f, 0.7788165123814759f, 0.6272518154951442f, 0.7807372285720944f, 0.6248594881423865f, 0.7826505961665756f, 0.6224612793741501f, 0.7845565971555752f, 0.6200572117632892f, 0.7864552135990858f, 0.617647307937804f, 0.7883464276266062f, 0.6152315905806269f, 0.79023022143731f, 0.6128100824294098f, 0.7921065773002123f, 0.6103828062763095f, 0.7939754775543371f, 0.6079497849677737f, 0.7958369046088835f, 0.6055110414043255f, 0.797690840943391f, 0.6030665985403483f, 0.799537269107905f, 0.600616479383869f, 0.8013761717231401f, 0.5981607069963424f, 0.8032075314806448f, 0.5956993044924335f, 0.8050313311429635f, 0.5932322950397999f, 0.8068475535437992f, 0.5907597018588743f, 0.808656181588175f, 0.5882815482226453f, 0.8104571982525948f, 0.5857978574564389f, 0.8122505865852039f, 0.5833086529376984f, 0.8140363297059483f, 0.5808139580957646f, 0.8158144108067338f, 0.5783137964116557f, 0.8175848131515837f, 0.5758081914178454f, 0.8193475200767969f, 0.5732971666980423f, 0.8211025149911046f, 0.5707807458869674f, 0.8228497813758263f, 0.5682589526701316f, 0.8245893027850253f, 0.5657318107836132f, 0.8263210628456634f, 0.5631993440138342f, 0.8280450452577557f, 0.560661576197336f, 0.829761233794523f, 0.5581185312205562f, 0.8314696123025451f, 0.5555702330196023f, 0.8331701647019132f, 0.5530167055800276f, 0.83486287498638f, 0.5504579729366049f, 0.8365477272235119f, 0.5478940591731003f, 0.838224705554838f, 0.5453249884220466f, 0.8398937941959994f, 0.542750784864516f, 0.8415549774368983f, 0.540171472729893f, 0.8432082396418454f, 0.5375870762956455f, 0.844853565249707f, 0.5349976198870974f, 0.846490938774052f, 0.532403127877198f, 0.8481203448032971f, 0.5298036246862948f, 0.8497417680008524f, 0.5271991347819014f, 0.8513551931052652f, 0.524589682678469f, 0.8529606049303636f, 0.5219752929371544f, 0.8545579883654005f, 0.5193559901655896f, 0.8561473283751944f, 0.51673179901765f, 0.8577286100002721f, 0.5141027441932218f, 0.8593018183570083f, 0.5114688504379705f, 0.8608669386377672f, 0.5088301425431071f, 0.8624239561110405f, 0.5061866453451555f, 0.8639728561215867f, 0.5035383837257176f, 0.865513624090569f, 0.5008853826112409f, 0.8670462455156926f, 0.4982276669727819f, 0.8685707059713409f, 0.49556526182577254f, 0.8700869911087113f, 0.49289819222978415f, 0.871595086655951f, 0.49022648328829116f, 0.87309497841829f, 0.4875501601484361f, 0.8745866522781761f, 0.4848692480007911f, 0.8760700941954065f, 0.4821837720791229f, 0.8775452902072612f, 0.47949375766015306f, 0.8790122264286334f, 0.47679923006332225f, 0.8804708890521608f, 0.47410021465055f, 0.8819212643483549f, 0.4713967368259978f, 0.8833633386657316f, 0.46868882203582796f, 0.8847970984309378f, 0.4659764957679661f, 0.8862225301488806f, 0.46325978355186026f, 0.8876396204028539f, 0.46053871095824f, 0.8890483558546645f, 0.4578133035988773f, 0.8904487232447579f, 0.45508358712634384f, 0.8918407093923427f, 0.452349587233771f, 0.8932243011955153f, 0.4496113296546066f, 0.8945994856313826f, 0.4468688401623743f, 0.8959662497561851f, 0.44412214457042926f, 0.8973245807054183f, 0.4413712687317166f, 0.8986744656939538f, 0.4386162385385277f, 0.9000158920161603f, 0.4358570799222555f, 0.901348847046022f, 0.433093818853152f, 0.9026733182372588f, 0.4303264813400826f, 0.9039892931234433f, 0.4275550934302822f, 0.9052967593181188f, 0.4247796812091088f, 0.9065957045149153f, 0.4220002707997998f, 0.9078861164876662f, 0.41921688836322396f, 0.9091679830905224f, 0.4164295600976373f, 0.9104412922580671f, 0.41363831223843456f, 0.9117060320054299f, 0.4108431710579039f, 0.9129621904283981f, 0.40804416286497874f, 0.9142097557035307f, 0.4052413140049898f, 0.9154487160882678f, 0.40243465085941854f, 0.9166790599210427f, 0.3996241998456468f, 0.9179007756213904f, 0.3968099874167104f, 0.9191138516900578f, 0.3939920400610481f, 0.9203182767091105f, 0.391170384302254f, 0.9215140393420419f, 0.3883450466988263f, 0.9227011283338785f, 0.385516053843919f, 0.9238795325112867f, 0.38268343236508984f, 0.9250492407826776f, 0.3798472089240511f, 0.9262102421383114f, 0.3770074102164183f, 0.9273625256504011f, 0.37416406297145793f, 0.9285060804732155f, 0.3713171939518376f, 0.9296408958431812f, 0.3684668299533723f, 0.9307669610789837f, 0.36561299780477396f, 0.9318842655816681f, 0.3627557243673972f, 0.9329927988347388f, 0.3598950365349883f, 0.9340925504042589f, 0.35703096123343003f, 0.9351835099389475f, 0.3541635254204905f, 0.9362656671702783f, 0.35129275608556715f, 0.937339011912575f, 0.3484186802494345f, 0.9384035340631081f, 0.34554132496398915f, 0.9394592236021899f, 0.3426607173119944f, 0.9405060705932683f, 0.33977688440682696f, 0.9415440651830208f, 0.33688985339222005f, 0.9425731976014469f, 0.3339996514420095f, 0.9435934581619604f, 0.33110630575987643f, 0.9446048372614803f, 0.32820984357909266f, 0.9456073253805213f, 0.325310292162263f, 0.9466009130832835f, 0.32240767880106996f, 0.9475855910177411f, 0.31950203081601575f, 0.9485613499157303f, 0.31659337555616585f, 0.9495281805930367f, 0.3136817403988915f, 0.9504860739494817f, 0.3107671527496115f, 0.9514350209690083f, 0.307849640041535f, 0.9523750127197659f, 0.3049292297354024f, 0.9533060403541938f, 0.3020059493192282f, 0.9542280951091057f, 0.2990798263080405f, 0.9551411683057707f, 0.29615088824362396f, 0.9560452513499964f, 0.2932191626942587f, 0.9569403357322089f, 0.2902846772544623f, 0.9578264130275329f, 0.28734745954472957f, 0.9587034748958716f, 0.2844075372112718f, 0.9595715130819845f, 0.28146493792575805f, 0.9604305194155658f, 0.27851968938505306f, 0.9612804858113206f, 0.27557181931095825f, 0.9621214042690416f, 0.272621355449949f, 0.9629532668736839f, 0.2696683255729152f, 0.9637760657954398f, 0.2667127574748984f, 0.9645897932898126f, 0.2637546789748315f, 0.9653944416976894f, 0.26079411791527557f, 0.9661900034454126f, 0.25783110216215893f, 0.9669764710448521f, 0.2548656596045146f, 0.9677538370934755f, 0.2518978181542169f, 0.9685220942744173f, 0.24892760574572026f, 0.9692812353565485f, 0.2459550503357946f, 0.970031253194544f, 0.24298017990326398f, 0.9707721407289504f, 0.2400030224487415f, 0.9715038909862518f, 0.23702360599436734f, 0.9722264970789363f, 0.23404195858354346f, 0.9729399522055601f, 0.23105810828067125f, 0.9736442496508119f, 0.2280720831708858f, 0.9743393827855759f, 0.22508391135979278f, 0.9750253450669941f, 0.2220936209732036f, 0.9757021300385286f, 0.21910124015686977f, 0.9763697313300211f, 0.2161067970762196f, 0.9770281426577544f, 0.21311031991609136f, 0.9776773578245099f, 0.21011183688046972f, 0.9783173707196277f, 0.20711137619221856f, 0.9789481753190622f, 0.204108966092817f, 0.9795697656854405f, 0.20110463484209193f, 0.9801821359681173f, 0.19809841071795373f, 0.9807852804032304f, 0.1950903220161283f, 0.9813791933137546f, 0.19208039704989238f, 0.9819638691095552f, 0.18906866414980628f, 0.9825393022874412f, 0.1860551516634466f, 0.9831054874312163f, 0.18303988795514103f, 0.9836624192117303f, 0.18002290140569951f, 0.984210092386929f, 0.17700422041214886f, 0.9847485018019042f, 0.17398387338746385f, 0.9852776423889412f, 0.17096188876030136f, 0.9857975091675674f, 0.1679382949747312f, 0.9863080972445987f, 0.16491312048997006f, 0.9868094018141854f, 0.16188639378011188f, 0.9873014181578584f, 0.1588581433338614f, 0.9877841416445722f, 0.15582839765426532f, 0.9882575677307495f, 0.1527971852584434f, 0.9887216919603238f, 0.14976453467732162f, 0.989176509964781f, 0.14673047445536175f, 0.9896220174632008f, 0.14369503315029458f, 0.9900582102622971f, 0.14065823933284924f, 0.990485084256457f, 0.13762012158648618f, 0.99090263542778f, 0.13458070850712622f, 0.9913108598461154f, 0.13154002870288325f, 0.9917097536690995f, 0.12849811079379322f, 0.9920993131421918f, 0.1254549834115462f, 0.99247953459871f, 0.12241067519921628f, 0.9928504144598651f, 0.11936521481099134f, 0.9932119492347945f, 0.11631863091190486f, 0.9935641355205953f, 0.11327095217756435f, 0.9939069700023561f, 0.11022220729388318f, 0.9942404494531879f, 0.10717242495680887f, 0.9945645707342554f, 0.10412163387205473f, 0.9948793307948056f, 0.10106986275482786f, 0.9951847266721968f, 0.09801714032956077f, 0.9954807554919269f, 0.09496349532963906f, 0.9957674144676598f, 0.0919089564971327f, 0.996044700901252f, 0.08885355258252468f, 0.996312612182778f, 0.08579731234443988f, 0.9965711457905548f, 0.0827402645493758f, 0.9968202992911657f, 0.07968243797143013f, 0.997060070339483f, 0.07662386139203162f, 0.9972904566786902f, 0.07356456359966745f, 0.9975114561403035f, 0.07050457338961401f, 0.9977230666441916f, 0.0674439195636641f, 0.997925286198596f, 0.06438263092985741f, 0.9981181129001492f, 0.061320736302208655f, 0.9983015449338929f, 0.05825826450043573f, 0.9984755805732948f, 0.05519524434969003f, 0.9986402181802653f, 0.05213170468028332f, 0.9987954562051724f, 0.04906767432741813f, 0.9989412931868569f, 0.046003182130914644f, 0.9990777277526454f, 0.04293825693494096f, 0.9992047586183639f, 0.039872927587739845f, 0.9993223845883495f, 0.03680722294135899f, 0.9994306045554617f, 0.03374117185137764f, 0.9995294175010931f, 0.030674803176636584f, 0.9996188224951786f, 0.02760814577896582f, 0.9996988186962042f, 0.024541228522912267f, 0.9997694053512153f, 0.02147408027546961f, 0.9998305817958234f, 0.01840672990580482f, 0.9998823474542126f, 0.01533920628498822f, 0.9999247018391445f, 0.012271538285719944f, 0.9999576445519639f, 0.00920375478205996f, 0.9999811752826011f, 0.006135884649154516f, 0.9999952938095762f, 0.003067956762966138f};
        calc_power_in_place(accel, &qx_rfft_twiddle_2048, n);
        break;
#endif

#if MAX_FFT_SIG_LEN >= 4096
    case 4096:
        arm_cfft_f32(&arm_cfft_sR_f32_len2048, accel, 0, 1);
        static const float qx_rfft_twiddle_4096[] = {0.0f, 1.0f, 0.0015339801862847655f, 0.9999988234517019f, 0.003067956762965976f, 0.9999952938095762f, 0.0046019261204485705f, 0.9999894110819284f, 0.006135884649154475f, 0.9999811752826011f, 0.007669828739531097f, 0.9999705864309741f, 0.00920375478205982f, 0.9999576445519639f, 0.01073765916726449f, 0.9999423496760239f, 0.012271538285719925f, 0.9999247018391445f, 0.01380538852806039f, 0.9999047010828529f, 0.0153392062849881f, 0.9998823474542126f, 0.01687298794728171f, 0.9998576410058239f, 0.01840672990580482f, 0.9998305817958234f, 0.01994042855151444f, 0.9998011698878843f, 0.021474080275469508f, 0.9997694053512153f, 0.02300768146883937f, 0.9997352882605617f, 0.024541228522912288f, 0.9996988186962042f, 0.0260747178291039f, 0.9996599967439592f, 0.02760814577896574f, 0.9996188224951786f, 0.029141508764193722f, 0.9995752960467492f, 0.030674803176636626f, 0.9995294175010931f, 0.032208025408304586f, 0.999481186966167f, 0.03374117185137758f, 0.9994306045554617f, 0.03527423889821395f, 0.9993776703880028f, 0.03680722294135883f, 0.9993223845883495f, 0.038340120373552694f, 0.9992647472865944f, 0.03987292758773981f, 0.9992047586183639f, 0.04140564097707674f, 0.9991424187248169f, 0.04293825693494082f, 0.9990777277526454f, 0.04447077185493867f, 0.9990106858540734f, 0.04600318213091462f, 0.9989412931868569f, 0.0475354841569593f, 0.9988695499142836f, 0.049067674327418015f, 0.9987954562051724f, 0.05059974903689928f, 0.9987190122338729f, 0.052131704680283324f, 0.9986402181802653f, 0.05366353765273052f, 0.9985590742297593f, 0.05519524434968994f, 0.9984755805732948f, 0.05672682116690775f, 0.9983897374073402f, 0.05825826450043575f, 0.9983015449338929f, 0.05978957074663987f, 0.9982110033604782f, 0.06132073630220858f, 0.9981181129001492f, 0.0628517575641614f, 0.9980228737714862f, 0.06438263092985747f, 0.997925286198596f, 0.0659133527970038f, 0.9978253504111116f, 0.06744391956366405f, 0.9977230666441916f, 0.06897432762826675f, 0.9976184351385196f, 0.07050457338961386f, 0.9975114561403035f, 0.07203465324688933f, 0.9974021299012753f, 0.07356456359966743f, 0.9972904566786902f, 0.0750943008479213f, 0.9971764367353262f, 0.07662386139203149f, 0.997060070339483f, 0.07815324163279423f, 0.9969413577649822f, 0.07968243797143013f, 0.9968202992911657f, 0.08121144680959244f, 0.9966968952028961f, 0.08274026454937569f, 0.9965711457905548f, 0.08426888759332407f, 0.9964430513500426f, 0.0857973123444399f, 0.996312612182778f, 0.08732553520619206f, 0.996179828595697f, 0.0888535525825246f, 0.996044700901252f, 0.09038136087786498f, 0.9959072294174117f, 0.09190895649713272f, 0.9957674144676598f, 0.09343633584574779f, 0.9956252563809943f, 0.09496349532963899f, 0.9954807554919269f, 0.09649043135525259f, 0.9953339121404823f, 0.0980171403295606f, 0.9951847266721969f, 0.09954361866006932f, 0.9950331994381186f, 0.10106986275482782f, 0.9948793307948056f, 0.10259586902243628f, 0.9947231211043257f, 0.10412163387205459f, 0.9945645707342554f, 0.10564715371341062f, 0.9944036800576791f, 0.10717242495680884f, 0.9942404494531879f, 0.10869744401313872f, 0.9940748793048794f, 0.11022220729388306f, 0.9939069700023561f, 0.11174671121112659f, 0.9937367219407246f, 0.11327095217756435f, 0.9935641355205953f, 0.11479492660651008f, 0.9933892111480807f, 0.11631863091190475f, 0.9932119492347945f, 0.11784206150832498f, 0.9930323501978514f, 0.11936521481099135f, 0.9928504144598651f, 0.12088808723577708f, 0.992666142448948f, 0.1224106751992162f, 0.99247953459871f, 0.12393297511851216f, 0.9922905913482574f, 0.12545498341154623f, 0.9920993131421918f, 0.12697669649688587f, 0.9919057004306093f, 0.12849811079379317f, 0.9917097536690995f, 0.13001922272223335f, 0.9915114733187439f, 0.13154002870288312f, 0.9913108598461154f, 0.13306052515713906f, 0.9911079137232769f, 0.13458070850712617f, 0.99090263542778f, 0.1361005751757062f, 0.9906950254426646f, 0.13762012158648604f, 0.9904850842564571f, 0.1391393441638262f, 0.9902728123631691f, 0.1406582393328492f, 0.9900582102622971f, 0.14217680351944803f, 0.9898412784588205f, 0.14369503315029447f, 0.9896220174632009f, 0.14521292465284746f, 0.9894004277913804f, 0.14673047445536175f, 0.989176509964781f, 0.14824767898689603f, 0.988950264510303f, 0.1497645346773215f, 0.9887216919603238f, 0.15128103795733022f, 0.9884907928526966f, 0.15279718525844344f, 0.9882575677307495f, 0.1543129730130201f, 0.9880220171432835f, 0.15582839765426523f, 0.9877841416445722f, 0.15734345561623825f, 0.9875439417943592f, 0.15885814333386145f, 0.9873014181578584f, 0.16037245724292828f, 0.987056571305751f, 0.16188639378011183f, 0.9868094018141855f, 0.16339994938297323f, 0.9865599102647754f, 0.16491312048996992f, 0.9863080972445987f, 0.1664259035404641f, 0.9860539633461954f, 0.16793829497473117f, 0.9857975091675675f, 0.16945029123396796f, 0.9855387353121761f, 0.17096188876030122f, 0.9852776423889412f, 0.17247308399679595f, 0.9850142310122398f, 0.17398387338746382f, 0.9847485018019042f, 0.17549425337727143f, 0.9844804553832209f, 0.17700422041214875f, 0.984210092386929f, 0.1785137709389975f, 0.9839374134492189f, 0.18002290140569951f, 0.9836624192117303f, 0.18153160826112497f, 0.9833851103215512f, 0.18303988795514095f, 0.9831054874312163f, 0.18454773693861962f, 0.9828235511987052f, 0.18605515166344663f, 0.9825393022874412f, 0.1875621285825296f, 0.9822527413662894f, 0.1890686641498062f, 0.9819638691095552f, 0.19057475482025274f, 0.9816726861969831f, 0.19208039704989244f, 0.9813791933137546f, 0.1935855872958036f, 0.9810833911504867f, 0.19509032201612825f, 0.9807852804032304f, 0.19659459767008022f, 0.9804848617734694f, 0.19809841071795356f, 0.9801821359681174f, 0.19960175762113097f, 0.9798771036995176f, 0.2011046348420919f, 0.9795697656854405f, 0.20260703884442113f, 0.979260122649082f, 0.20410896609281687f, 0.9789481753190622f, 0.20561041305309924f, 0.9786339244294232f, 0.20711137619221856f, 0.9783173707196277f, 0.20861185197826349f, 0.9779985149345571f, 0.2101118368804696f, 0.9776773578245099f, 0.21161132736922755f, 0.9773539001452001f, 0.21311031991609136f, 0.9770281426577544f, 0.21460881099378676f, 0.9767000861287118f, 0.21610679707621952f, 0.9763697313300211f, 0.21760427463848364f, 0.976037079039039f, 0.2191012401568698f, 0.9757021300385286f, 0.2205976901088735f, 0.975364885116657f, 0.2220936209732035f, 0.9750253450669941f, 0.22358902922979f, 0.9746835106885107f, 0.22508391135979283f, 0.9743393827855759f, 0.22657826384561f, 0.9739929621679558f, 0.22807208317088573f, 0.973644249650812f, 0.22956536582051887f, 0.9732932460546982f, 0.2310581082806711f, 0.9729399522055602f, 0.23255030703877524f, 0.9725843689347322f, 0.23404195858354343f, 0.9722264970789363f, 0.2355330594049755f, 0.9718663374802794f, 0.2370236059943672f, 0.9715038909862518f, 0.23851359484431842f, 0.9711391584497251f, 0.2400030224487415f, 0.9707721407289504f, 0.24149188530286933f, 0.9704028386875555f, 0.24298017990326387f, 0.970031253194544f, 0.24446790274782415f, 0.9696573851242924f, 0.2459550503357946f, 0.9692812353565485f, 0.24744161916777327f, 0.9689028047764289f, 0.24892760574572015f, 0.9685220942744173f, 0.2504130065729652f, 0.9681391047463624f, 0.25189781815421697f, 0.9677538370934755f, 0.25338203699557016f, 0.9673662922223285f, 0.25486565960451457f, 0.9669764710448521f, 0.2563486824899429f, 0.9665843744783331f, 0.257831102162159f, 0.9661900034454125f, 0.25931291513288623f, 0.9657933588740837f, 0.2607941179152755f, 0.9653944416976894f, 0.2622747070239136f, 0.9649932528549203f, 0.26375467897483135f, 0.9645897932898128f, 0.2652340302855118f, 0.9641840639517458f, 0.26671275747489837f, 0.9637760657954398f, 0.2681908570634032f, 0.963365799780954f, 0.2696683255729151f, 0.9629532668736839f, 0.271145159526808f, 0.9625384680443592f, 0.272621355449949f, 0.9621214042690416f, 0.2740969098687064f, 0.9617020765291225f, 0.27557181931095814f, 0.9612804858113206f, 0.2770460803060999f, 0.9608566331076797f, 0.27851968938505306f, 0.9604305194155658f, 0.2799926430802732f, 0.960002145737666f, 0.28146493792575794f, 0.9595715130819845f, 0.2829365704570554f, 0.9591386224618419f, 0.2844075372112719f, 0.9587034748958716f, 0.2858778347270806f, 0.9582660714080177f, 0.2873474595447295f, 0.9578264130275329f, 0.2888164082060495f, 0.9573845007889759f, 0.29028467725446233f, 0.9569403357322088f, 0.29175226323498926f, 0.9564939189023951f, 0.29321916269425863f, 0.9560452513499964f, 0.2946853721805143f, 0.9555943341307711f, 0.2961508882436238f, 0.9551411683057707f, 0.2976157074350862f, 0.9546857549413383f, 0.2990798263080405f, 0.9542280951091057f, 0.30054324141727345f, 0.9537681898859903f, 0.3020059493192281f, 0.9533060403541939f, 0.3034679465720113f, 0.9528416476011987f, 0.3049292297354024f, 0.9523750127197659f, 0.3063897953708609f, 0.9519061368079323f, 0.30784964004153487f, 0.9514350209690083f, 0.3093087603122687f, 0.9509616663115751f, 0.3107671527496115f, 0.9504860739494817f, 0.3122248139218249f, 0.950008245001843f, 0.3136817403988915f, 0.9495281805930367f, 0.31513792875252244f, 0.9490458818527006f, 0.31659337555616585f, 0.9485613499157303f, 0.31804807738501495f, 0.9480745859222762f, 0.3195020308160157f, 0.9475855910177411f, 0.3209552324278752f, 0.9470943663527772f, 0.32240767880106985f, 0.9466009130832835f, 0.32385936651785285f, 0.9461052323704034f, 0.3253102921622629f, 0.9456073253805213f, 0.32676045232013173f, 0.9451071932852606f, 0.3282098435790925f, 0.9446048372614803f, 0.3296584625285875f, 0.9441002584912727f, 0.33110630575987643f, 0.9435934581619604f, 0.3325533698660442f, 0.9430844374660935f, 0.3339996514420094f, 0.9425731976014469f, 0.3354451470845316f, 0.9420597397710173f, 0.33688985339222005f, 0.9415440651830208f, 0.3383337669655411f, 0.9410261750508893f, 0.33977688440682685f, 0.9405060705932683f, 0.34121920232028236f, 0.939983753034014f, 0.3426607173119944f, 0.9394592236021899f, 0.3441014259899388f, 0.9389324835320645f, 0.3455413249639891f, 0.9384035340631081f, 0.3469804108459237f, 0.9378723764399899f, 0.34841868024943456f, 0.937339011912575f, 0.3498561297901349f, 0.9368034417359216f, 0.3512927560855671f, 0.9362656671702783f, 0.3527285557552107f, 0.9357256894810804f, 0.35416352542049034f, 0.9351835099389476f, 0.35559766170478385f, 0.9346391298196808f, 0.35703096123343f, 0.934092550404259f, 0.35846342063373654f, 0.9335437729788362f, 0.3598950365349881f, 0.932992798834739f, 0.3613258055684543f, 0.9324396292684624f, 0.3627557243673972f, 0.9318842655816681f, 0.3641847895670799f, 0.9313267090811804f, 0.36561299780477385f, 0.9307669610789837f, 0.3670403457197672f, 0.9302050228922191f, 0.3684668299533723f, 0.9296408958431812f, 0.3698924471489341f, 0.9290745812593157f, 0.37131719395183754f, 0.9285060804732156f, 0.37274106700951576f, 0.9279353948226179f, 0.37416406297145793f, 0.9273625256504011f, 0.3755861784892172f, 0.9267874743045817f, 0.37700741021641826f, 0.9262102421383114f, 0.37842775480876556f, 0.9256308305098727f, 0.37984720892405116f, 0.9250492407826776f, 0.3812657692221624f, 0.9244654743252626f, 0.3826834323650898f, 0.9238795325112867f, 0.38410019501693504f, 0.9232914167195276f, 0.38551605384391885f, 0.9227011283338786f, 0.3869310055143886f, 0.9221086687433451f, 0.38834504669882625f, 0.9215140393420419f, 0.3897581740698564f, 0.9209172415291895f, 0.39117038430225387f, 0.9203182767091106f, 0.39258167407295147f, 0.9197171462912274f, 0.3939920400610481f, 0.9191138516900578f, 0.39540147894781635f, 0.9185083943252123f, 0.3968099874167103f, 0.9179007756213905f, 0.39821756215337356f, 0.9172909970083779f, 0.3996241998456468f, 0.9166790599210427f, 0.4010298971835756f, 0.9160649657993317f, 0.40243465085941843f, 0.9154487160882678f, 0.4038384575676541f, 0.9148303122379462f, 0.40524131400498986f, 0.9142097557035307f, 0.40664321687036903f, 0.9135870479452508f, 0.4080441628649787f, 0.9129621904283982f, 0.4094441486922576f, 0.9123351846233227f, 0.4108431710579039f, 0.9117060320054299f, 0.4122412266698829f, 0.9110747340551764f, 0.41363831223843456f, 0.9104412922580672f, 0.41503442447608163f, 0.9098057081046522f, 0.41642956009763715f, 0.9091679830905224f, 0.41782371582021227f, 0.9085281187163061f, 0.4192168883632239f, 0.9078861164876663f, 0.4206090744484025f, 0.9072419779152958f, 0.4220002707997997f, 0.9065957045149153f, 0.42339047414379605f, 0.9059472978072685f, 0.4247796812091088f, 0.9052967593181188f, 0.4261678887267996f, 0.9046440905782462f, 0.4275550934302821f, 0.9039892931234433f, 0.4289412920553295f, 0.9033323684945118f, 0.4303264813400826f, 0.9026733182372588f, 0.43171065802505726f, 0.9020121439024932f, 0.43309381885315196f, 0.901348847046022f, 0.43447596056965565f, 0.900683429228647f, 0.4358570799222555f, 0.9000158920161603f, 0.4372371736610441f, 0.8993462369793416f, 0.43861623853852766f, 0.8986744656939538f, 0.43999427130963326f, 0.8980005797407399f, 0.44137126873171667f, 0.8973245807054183f, 0.44274722756457f, 0.8966464701786803f, 0.4441221445704292f, 0.8959662497561852f, 0.44549601651398174f, 0.8952839210385576f, 0.44686884016237416f, 0.8945994856313827f, 0.4482406122852199f, 0.8939129451452033f, 0.44961132965460654f, 0.8932243011955153f, 0.45098098904510386f, 0.8925335554027646f, 0.4523495872337709f, 0.8918407093923427f, 0.45371712100016387f, 0.8911457647945832f, 0.45508358712634384f, 0.8904487232447579f, 0.4564489823968839f, 0.8897495863830729f, 0.45781330359887723f, 0.8890483558546646f, 0.4591765475219441f, 0.8883450333095964f, 0.46053871095824f, 0.8876396204028539f, 0.46189979070246273f, 0.8869321187943422f, 0.46325978355186015f, 0.8862225301488806f, 0.4646186863062378f, 0.8855108561362f, 0.4659764957679662f, 0.8847970984309378f, 0.46733320874198847f, 0.884081258712635f, 0.46868882203582796f, 0.8833633386657316f, 0.4700433324595956f, 0.8826433399795628f, 0.47139673682599764f, 0.881921264348355f, 0.4727490319503428f, 0.881197113471222f, 0.47410021465054997f, 0.8804708890521608f, 0.47545028174715587f, 0.8797425928000474f, 0.4767992300633221f, 0.8790122264286335f, 0.478147056424843f, 0.8782797916565416f, 0.479493757660153f, 0.8775452902072614f, 0.48083933060033396f, 0.8768087238091457f, 0.4821837720791227f, 0.8760700941954066f, 0.48352707893291874f, 0.8753294031041109f, 0.48486924800079106f, 0.8745866522781762f, 0.4862102761244864f, 0.8738418434653669f, 0.487550160148436f, 0.8730949784182901f, 0.48888889691976317f, 0.8723460588943915f, 0.49022648328829116f, 0.871595086655951f, 0.4915629161065499f, 0.8708420634700789f, 0.49289819222978404f, 0.8700869911087115f, 0.4942323085159597f, 0.8693298713486068f, 0.49556526182577254f, 0.8685707059713409f, 0.49689704902265447f, 0.8678094967633033f, 0.49822766697278187f, 0.8670462455156926f, 0.49955711254508184f, 0.866280954024513f, 0.5008853826112407f, 0.8655136240905691f, 0.5022124740457108f, 0.8647442575194624f, 0.5035383837257176f, 0.8639728561215867f, 0.5048631085312676f, 0.8631994217121242f, 0.5061866453451552f, 0.8624239561110405f, 0.5075089910529709f, 0.8616464611430813f, 0.508830142543107f, 0.8608669386377673f, 0.5101500967067668f, 0.8600853904293901f, 0.5114688504379703f, 0.8593018183570085f, 0.512786400633563f, 0.8585162242644429f, 0.5141027441932217f, 0.8577286100002721f, 0.5154178780194629f, 0.8569389774178288f, 0.5167317990176499f, 0.8561473283751945f, 0.5180445040959993f, 0.855353664735196f, 0.5193559901655896f, 0.8545579883654005f, 0.5206662541403672f, 0.8537603011381114f, 0.5219752929371544f, 0.8529606049303636f, 0.5232831034756564f, 0.8521589016239198f, 0.524589682678469f, 0.8513551931052652f, 0.5258950274710846f, 0.8505494812656035f, 0.5271991347819014f, 0.8497417680008525f, 0.5285020015422285f, 0.8489320552116396f, 0.5298036246862946f, 0.8481203448032972f, 0.531104001151255f, 0.8473066386858583f, 0.5324031278771979f, 0.8464909387740521f, 0.533701001807153f, 0.8456732469872991f, 0.5349976198870972f, 0.8448535652497071f, 0.5362929790659632f, 0.8440318954900664f, 0.5375870762956454f, 0.8432082396418454f, 0.5388799085310084f, 0.8423825996431858f, 0.5401714727298929f, 0.8415549774368984f, 0.5414617658531234f, 0.8407253749704581f, 0.5427507848645159f, 0.8398937941959995f, 0.5440385267308838f, 0.8390602370703127f, 0.5453249884220465f, 0.838224705554838f, 0.5466101669108349f, 0.8373872016156619f, 0.5478940591731002f, 0.836547727223512f, 0.5491766621877197f, 0.8357062843537526f, 0.5504579729366048f, 0.83486287498638f, 0.5517379884047073f, 0.8340175011060181f, 0.5530167055800275f, 0.8331701647019132f, 0.55429412145362f, 0.8323208677679297f, 0.5555702330196022f, 0.8314696123025452f, 0.5568450372751601f, 0.8306164003088463f, 0.5581185312205561f, 0.829761233794523f, 0.5593907118591361f, 0.8289041147718649f, 0.560661576197336f, 0.8280450452577558f, 0.5619311212446895f, 0.8271840272736691f, 0.5631993440138341f, 0.8263210628456635f, 0.5644662415205195f, 0.8254561540043776f, 0.5657318107836131f, 0.8245893027850253f, 0.5669960488251087f, 0.8237205112273914f, 0.5682589526701315f, 0.8228497813758264f, 0.5695205193469471f, 0.8219771152792416f, 0.5707807458869673f, 0.8211025149911046f, 0.572039629324757f, 0.8202259825694347f, 0.5732971666980422f, 0.819347520076797f, 0.5745533550477158f, 0.8184671295802987f, 0.5758081914178453f, 0.8175848131515837f, 0.5770616728556794f, 0.8167005728668278f, 0.5783137964116556f, 0.8158144108067338f, 0.5795645591394056f, 0.8149263290565266f, 0.5808139580957645f, 0.8140363297059484f, 0.5820619903407754f, 0.8131444148492536f, 0.5833086529376983f, 0.8122505865852039f, 0.5845539429530153f, 0.8113548470170637f, 0.5857978574564389f, 0.8104571982525948f, 0.587040393520918f, 0.8095576424040513f, 0.5882815482226452f, 0.808656181588175f, 0.5895213186410639f, 0.8077528179261904f, 0.5907597018588742f, 0.8068475535437993f, 0.591996694962041f, 0.8059403905711763f, 0.5932322950397998f, 0.8050313311429637f, 0.5944664991846644f, 0.8041203773982657f, 0.5956993044924334f, 0.8032075314806449f, 0.5969307080621965f, 0.8022927955381157f, 0.5981607069963424f, 0.8013761717231402f, 0.5993892984005645f, 0.8004576621926228f, 0.600616479383869f, 0.799537269107905f, 0.60184224705858f, 0.7986149946347609f, 0.6030665985403482f, 0.7976908409433912f, 0.604289530948156f, 0.7967648102084188f, 0.6055110414043255f, 0.7958369046088836f, 0.6067311270345245f, 0.794907126328237f, 0.6079497849677736f, 0.7939754775543372f, 0.6091670123364532f, 0.7930419604794436f, 0.6103828062763095f, 0.7921065773002124f, 0.6115971639264619f, 0.7911693302176901f, 0.6128100824294097f, 0.79023022143731f, 0.6140215589310385f, 0.7892892531688857f, 0.6152315905806268f, 0.7883464276266063f, 0.6164401745308536f, 0.7874017470290313f, 0.6176473079378039f, 0.7864552135990858f, 0.6188529879609763f, 0.7855068295640539f, 0.6200572117632891f, 0.7845565971555752f, 0.6212599765110876f, 0.7836045186096383f, 0.62246127937415f, 0.7826505961665757f, 0.6236611175256945f, 0.7816948320710595f, 0.6248594881423863f, 0.7807372285720945f, 0.6260563884043435f, 0.7797777879230146f, 0.6272518154951441f, 0.778816512381476f, 0.6284457666018327f, 0.7778534042094531f, 0.629638238914927f, 0.7768884656732324f, 0.6308292296284245f, 0.7759216990434077f, 0.6320187359398091f, 0.7749531065948738f, 0.6332067550500572f, 0.7739826906068229f, 0.6343932841636455f, 0.773010453362737f, 0.6355783204885561f, 0.7720363971503845f, 0.6367618612362842f, 0.7710605242618138f, 0.637943903621844f, 0.7700828369933479f, 0.6391244448637757f, 0.7691033376455797f, 0.6403034821841517f, 0.7681220285233654f, 0.6414810128085832f, 0.7671389119358204f, 0.6426570339662269f, 0.7661539901963129f, 0.6438315428897914f, 0.765167265622459f, 0.6450045368155439f, 0.7641787405361167f, 0.6461760129833163f, 0.7631884172633814f, 0.6473459686365121f, 0.7621962981345789f, 0.6485144010221124f, 0.7612023854842618f, 0.6496813073906832f, 0.7602066816512024f, 0.650846684996381f, 0.759209188978388f, 0.6520105310969595f, 0.7582099098130153f, 0.6531728429537768f, 0.7572088465064846f, 0.6543336178318004f, 0.7562060014143945f, 0.6554928529996153f, 0.7552013768965365f, 0.656650545729429f, 0.7541949753168892f, 0.6578066932970786f, 0.7531867990436125f, 0.6589612929820373f, 0.7521768504490427f, 0.6601143420674205f, 0.7511651319096864f, 0.6612658378399923f, 0.7501516458062151f, 0.6624157775901718f, 0.7491363945234594f, 0.6635641586120398f, 0.7481193804504036f, 0.6647109782033448f, 0.7471006059801801f, 0.6658562336655097f, 0.7460800735100638f, 0.6669999223036375f, 0.745057785441466f, 0.6681420414265185f, 0.7440337441799293f, 0.669282588346636f, 0.7430079521351217f, 0.6704215603801731f, 0.7419804117208311f, 0.6715589548470183f, 0.7409511253549592f, 0.6726947690707729f, 0.7399200954595162f, 0.673829000378756f, 0.7388873244606151f, 0.674961646102012f, 0.737852814788466f, 0.6760927035753159f, 0.7368165688773698f, 0.6772221701371803f, 0.7357785891657136f, 0.6783500431298615f, 0.7347388780959634f, 0.679476319899365f, 0.7336974381146604f, 0.680600997795453f, 0.7326542716724128f, 0.6817240741716497f, 0.7316093812238926f, 0.6828455463852481f, 0.7305627692278276f, 0.6839654117973155f, 0.729514438146997f, 0.6850836677727004f, 0.7284643904482252f, 0.6862003116800386f, 0.7274126286023758f, 0.687315340891759f, 0.726359155084346f, 0.6884287527840904f, 0.7253039723730608f, 0.6895405447370668f, 0.724247082951467f, 0.6906507141345346f, 0.7231884893065275f, 0.6917592583641577f, 0.7221281939292153f, 0.6928661748174246f, 0.7210661993145081f, 0.6939714608896539f, 0.7200025079613817f, 0.6950751139800009f, 0.7189371223728045f, 0.696177131491463f, 0.7178700450557317f, 0.6972775108308865f, 0.7168012785210995f, 0.6983762494089729f, 0.7157308252838186f, 0.6994733446402838f, 0.7146586878627691f, 0.7005687939432483f, 0.7135848687807935f, 0.7016625947401685f, 0.7125093705646923f, 0.7027547444572253f, 0.7114321957452164f, 0.7038452405244849f, 0.7103533468570624f, 0.7049340803759049f, 0.7092728264388657f, 0.7060212614493397f, 0.7081906370331954f, 0.7071067811865475f, 0.7071067811865476f, 0.7081906370331953f, 0.7060212614493399f, 0.7092728264388656f, 0.704934080375905f, 0.7103533468570623f, 0.703845240524485f, 0.7114321957452163f, 0.7027547444572254f, 0.7125093705646923f, 0.7016625947401686f, 0.7135848687807935f, 0.7005687939432484f, 0.714658687862769f, 0.6994733446402839f, 0.7157308252838186f, 0.6983762494089729f, 0.7168012785210994f, 0.6972775108308866f, 0.7178700450557316f, 0.6961771314914631f, 0.7189371223728044f, 0.6950751139800009f, 0.7200025079613817f, 0.693971460889654f, 0.721066199314508f, 0.6928661748174247f, 0.7221281939292152f, 0.6917592583641579f, 0.7231884893065272f, 0.6906507141345347f, 0.7242470829514669f, 0.6895405447370669f, 0.7253039723730607f, 0.6884287527840905f, 0.7263591550843459f, 0.6873153408917592f, 0.7274126286023757f, 0.6862003116800387f, 0.7284643904482252f, 0.6850836677727005f, 0.7295144381469969f, 0.6839654117973155f, 0.7305627692278276f, 0.6828455463852481f, 0.7316093812238925f, 0.6817240741716498f, 0.7326542716724127f, 0.6806009977954531f, 0.7336974381146603f, 0.6794763198993651f, 0.7347388780959634f, 0.6783500431298616f, 0.7357785891657135f, 0.6772221701371804f, 0.7368165688773698f, 0.676092703575316f, 0.737852814788466f, 0.674961646102012f, 0.7388873244606151f, 0.6738290003787561f, 0.7399200954595161f, 0.672694769070773f, 0.740951125354959f, 0.6715589548470184f, 0.741980411720831f, 0.6704215603801732f, 0.7430079521351216f, 0.6692825883466361f, 0.7440337441799292f, 0.6681420414265186f, 0.745057785441466f, 0.6669999223036376f, 0.7460800735100637f, 0.6658562336655097f, 0.7471006059801801f, 0.6647109782033449f, 0.7481193804504035f, 0.6635641586120399f, 0.7491363945234593f, 0.6624157775901718f, 0.750151645806215f, 0.6612658378399923f, 0.7511651319096864f, 0.6601143420674206f, 0.7521768504490427f, 0.6589612929820373f, 0.7531867990436124f, 0.6578066932970787f, 0.7541949753168892f, 0.656650545729429f, 0.7552013768965364f, 0.6554928529996155f, 0.7562060014143945f, 0.6543336178318006f, 0.7572088465064845f, 0.6531728429537769f, 0.7582099098130153f, 0.6520105310969596f, 0.759209188978388f, 0.650846684996381f, 0.7602066816512023f, 0.6496813073906833f, 0.7612023854842618f, 0.6485144010221126f, 0.7621962981345789f, 0.6473459686365122f, 0.7631884172633813f, 0.6461760129833164f, 0.7641787405361167f, 0.645004536815544f, 0.7651672656224588f, 0.6438315428897915f, 0.7661539901963128f, 0.642657033966227f, 0.7671389119358203f, 0.6414810128085832f, 0.7681220285233653f, 0.6403034821841518f, 0.7691033376455796f, 0.6391244448637758f, 0.7700828369933479f, 0.6379439036218442f, 0.7710605242618137f, 0.6367618612362843f, 0.7720363971503844f, 0.6355783204885562f, 0.7730104533627369f, 0.6343932841636456f, 0.7739826906068228f, 0.6332067550500573f, 0.7749531065948738f, 0.6320187359398091f, 0.7759216990434076f, 0.6308292296284246f, 0.7768884656732324f, 0.6296382389149271f, 0.777853404209453f, 0.6284457666018327f, 0.7788165123814759f, 0.6272518154951442f, 0.7797777879230144f, 0.6260563884043436f, 0.7807372285720944f, 0.6248594881423865f, 0.7816948320710594f, 0.6236611175256946f, 0.7826505961665756f, 0.6224612793741501f, 0.7836045186096382f, 0.6212599765110877f, 0.7845565971555752f, 0.6200572117632892f, 0.7855068295640539f, 0.6188529879609764f, 0.7864552135990858f, 0.617647307937804f, 0.7874017470290313f, 0.6164401745308536f, 0.7883464276266062f, 0.6152315905806269f, 0.7892892531688855f, 0.6140215589310385f, 0.79023022143731f, 0.6128100824294098f, 0.7911693302176901f, 0.611597163926462f, 0.7921065773002123f, 0.6103828062763095f, 0.7930419604794436f, 0.6091670123364533f, 0.7939754775543371f, 0.6079497849677737f, 0.794907126328237f, 0.6067311270345246f, 0.7958369046088835f, 0.6055110414043255f, 0.7967648102084187f, 0.6042895309481561f, 0.797690840943391f, 0.6030665985403483f, 0.7986149946347608f, 0.6018422470585801f, 0.799537269107905f, 0.600616479383869f, 0.8004576621926227f, 0.5993892984005647f, 0.8013761717231401f, 0.5981607069963424f, 0.8022927955381156f, 0.5969307080621965f, 0.8032075314806448f, 0.5956993044924335f, 0.8041203773982657f, 0.5944664991846645f, 0.8050313311429635f, 0.5932322950397999f, 0.8059403905711763f, 0.591996694962041f, 0.8068475535437992f, 0.5907597018588743f, 0.8077528179261902f, 0.5895213186410639f, 0.808656181588175f, 0.5882815482226453f, 0.8095576424040511f, 0.5870403935209181f, 0.8104571982525948f, 0.5857978574564389f, 0.8113548470170637f, 0.5845539429530154f, 0.8122505865852039f, 0.5833086529376984f, 0.8131444148492535f, 0.5820619903407755f, 0.8140363297059483f, 0.5808139580957646f, 0.8149263290565265f, 0.5795645591394057f, 0.8158144108067338f, 0.5783137964116557f, 0.8167005728668277f, 0.5770616728556796f, 0.8175848131515837f, 0.5758081914178454f, 0.8184671295802987f, 0.5745533550477159f, 0.8193475200767969f, 0.5732971666980423f, 0.8202259825694347f, 0.5720396293247572f, 0.8211025149911046f, 0.5707807458869674f, 0.8219771152792414f, 0.5695205193469473f, 0.8228497813758263f, 0.5682589526701316f, 0.8237205112273913f, 0.5669960488251087f, 0.8245893027850253f, 0.5657318107836132f, 0.8254561540043774f, 0.5644662415205195f, 0.8263210628456634f, 0.5631993440138342f, 0.827184027273669f, 0.5619311212446895f, 0.8280450452577557f, 0.560661576197336f, 0.8289041147718648f, 0.5593907118591361f, 0.829761233794523f, 0.5581185312205562f, 0.8306164003088462f, 0.5568450372751602f, 0.8314696123025451f, 0.5555702330196023f, 0.8323208677679297f, 0.5542941214536201f, 0.8331701647019132f, 0.5530167055800276f, 0.834017501106018f, 0.5517379884047074f, 0.83486287498638f, 0.5504579729366049f, 0.8357062843537525f, 0.5491766621877198f, 0.8365477272235119f, 0.5478940591731003f, 0.8373872016156618f, 0.546610166910835f, 0.838224705554838f, 0.5453249884220466f, 0.8390602370703126f, 0.5440385267308839f, 0.8398937941959994f, 0.542750784864516f, 0.840725374970458f, 0.5414617658531236f, 0.8415549774368983f, 0.540171472729893f, 0.8423825996431858f, 0.5388799085310084f, 0.8432082396418454f, 0.5375870762956455f, 0.8440318954900664f, 0.5362929790659632f, 0.844853565249707f, 0.5349976198870974f, 0.8456732469872991f, 0.533701001807153f, 0.846490938774052f, 0.532403127877198f, 0.8473066386858583f, 0.5311040011512551f, 0.8481203448032971f, 0.5298036246862948f, 0.8489320552116396f, 0.5285020015422285f, 0.8497417680008524f, 0.5271991347819014f, 0.8505494812656034f, 0.5258950274710849f, 0.8513551931052652f, 0.524589682678469f, 0.8521589016239198f, 0.5232831034756564f, 0.8529606049303636f, 0.5219752929371544f, 0.8537603011381113f, 0.5206662541403673f, 0.8545579883654005f, 0.5193559901655896f, 0.855353664735196f, 0.5180445040959994f, 0.8561473283751944f, 0.51673179901765f, 0.8569389774178287f, 0.5154178780194631f, 0.8577286100002721f, 0.5141027441932218f, 0.8585162242644427f, 0.5127864006335631f, 0.8593018183570083f, 0.5114688504379705f, 0.8600853904293901f, 0.5101500967067668f, 0.8608669386377672f, 0.5088301425431071f, 0.8616464611430813f, 0.507508991052971f, 0.8624239561110405f, 0.5061866453451555f, 0.8631994217121242f, 0.5048631085312676f, 0.8639728561215867f, 0.5035383837257176f, 0.8647442575194624f, 0.5022124740457109f, 0.865513624090569f, 0.5008853826112409f, 0.866280954024513f, 0.4995571125450819f, 0.8670462455156926f, 0.4982276669727819f, 0.8678094967633032f, 0.4968970490226547f, 0.8685707059713409f, 0.49556526182577254f, 0.8693298713486067f, 0.4942323085159598f, 0.8700869911087113f, 0.49289819222978415f, 0.8708420634700789f, 0.4915629161065501f, 0.871595086655951f, 0.49022648328829116f, 0.8723460588943914f, 0.4888888969197632f, 0.87309497841829f, 0.4875501601484361f, 0.8738418434653668f, 0.4862102761244866f, 0.8745866522781761f, 0.4848692480007911f, 0.8753294031041108f, 0.4835270789329188f, 0.8760700941954065f, 0.4821837720791229f, 0.8768087238091457f, 0.48083933060033396f, 0.8775452902072612f, 0.47949375766015306f, 0.8782797916565415f, 0.4781470564248431f, 0.8790122264286334f, 0.47679923006332225f, 0.8797425928000474f, 0.47545028174715587f, 0.8804708890521608f, 0.47410021465055f, 0.881197113471222f, 0.4727490319503429f, 0.8819212643483549f, 0.4713967368259978f, 0.8826433399795628f, 0.4700433324595956f, 0.8833633386657316f, 0.46868882203582796f, 0.884081258712635f, 0.4673332087419885f, 0.8847970984309378f, 0.4659764957679661f, 0.8855108561362f, 0.4646186863062378f, 0.8862225301488806f, 0.46325978355186026f, 0.8869321187943421f, 0.46189979070246284f, 0.8876396204028539f, 0.46053871095824f, 0.8883450333095964f, 0.45917654752194415f, 0.8890483558546645f, 0.4578133035988773f, 0.8897495863830729f, 0.45644898239688386f, 0.8904487232447579f, 0.45508358712634384f, 0.8911457647945832f, 0.4537171210001639f, 0.8918407093923427f, 0.452349587233771f, 0.8925335554027647f, 0.4509809890451038f, 0.8932243011955153f, 0.4496113296546066f, 0.8939129451452033f, 0.44824061228522f, 0.8945994856313826f, 0.4468688401623743f, 0.8952839210385576f, 0.44549601651398174f, 0.8959662497561851f, 0.44412214457042926f, 0.8966464701786802f, 0.44274722756457013f, 0.8973245807054183f, 0.4413712687317166f, 0.8980005797407399f, 0.43999427130963326f, 0.8986744656939538f, 0.4386162385385277f, 0.8993462369793415f, 0.4372371736610442f, 0.9000158920161603f, 0.4358570799222555f, 0.9006834292286469f, 0.4344759605696557f, 0.901348847046022f, 0.433093818853152f, 0.9020121439024931f, 0.43171065802505737f, 0.9026733182372588f, 0.4303264813400826f, 0.9033323684945118f, 0.42894129205532955f, 0.9039892931234433f, 0.4275550934302822f, 0.9046440905782462f, 0.42616788872679956f, 0.9052967593181188f, 0.4247796812091088f, 0.9059472978072685f, 0.4233904741437961f, 0.9065957045149153f, 0.4220002707997998f, 0.9072419779152958f, 0.4206090744484025f, 0.9078861164876662f, 0.41921688836322396f, 0.9085281187163061f, 0.4178237158202124f, 0.9091679830905224f, 0.4164295600976373f, 0.9098057081046522f, 0.41503442447608163f, 0.9104412922580671f, 0.41363831223843456f, 0.9110747340551762f, 0.412241226669883f, 0.9117060320054299f, 0.4108431710579039f, 0.9123351846233227f, 0.4094441486922576f, 0.9129621904283981f, 0.40804416286497874f, 0.9135870479452508f, 0.40664321687036914f, 0.9142097557035307f, 0.4052413140049898f, 0.9148303122379462f, 0.40383845756765413f, 0.9154487160882678f, 0.40243465085941854f, 0.9160649657993316f, 0.4010298971835758f, 0.9166790599210427f, 0.3996241998456468f, 0.9172909970083779f, 0.3982175621533736f, 0.9179007756213904f, 0.3968099874167104f, 0.9185083943252123f, 0.3954014789478163f, 0.9191138516900578f, 0.3939920400610481f, 0.9197171462912274f, 0.3925816740729515f, 0.9203182767091105f, 0.391170384302254f, 0.9209172415291895f, 0.3897581740698564f, 0.9215140393420419f, 0.3883450466988263f, 0.9221086687433451f, 0.3869310055143887f, 0.9227011283338785f, 0.385516053843919f, 0.9232914167195276f, 0.38410019501693504f, 0.9238795325112867f, 0.38268343236508984f, 0.9244654743252626f, 0.3812657692221625f, 0.9250492407826776f, 0.3798472089240511f, 0.9256308305098727f, 0.37842775480876556f, 0.9262102421383114f, 0.3770074102164183f, 0.9267874743045817f, 0.3755861784892173f, 0.9273625256504011f, 0.37416406297145793f, 0.9279353948226179f, 0.3727410670095158f, 0.9285060804732155f, 0.3713171939518376f, 0.9290745812593157f, 0.3698924471489342f, 0.9296408958431812f, 0.3684668299533723f, 0.9302050228922191f, 0.36704034571976724f, 0.9307669610789837f, 0.36561299780477396f, 0.9313267090811805f, 0.36418478956707984f, 0.9318842655816681f, 0.3627557243673972f, 0.9324396292684624f, 0.36132580556845434f, 0.9329927988347388f, 0.3598950365349883f, 0.9335437729788362f, 0.35846342063373654f, 0.9340925504042589f, 0.35703096123343003f, 0.9346391298196808f, 0.35559766170478396f, 0.9351835099389475f, 0.3541635254204905f, 0.9357256894810804f, 0.3527285557552107f, 0.9362656671702783f, 0.35129275608556715f, 0.9368034417359216f, 0.34985612979013503f, 0.937339011912575f, 0.3484186802494345f, 0.9378723764399899f, 0.3469804108459237f, 0.9384035340631081f, 0.34554132496398915f, 0.9389324835320645f, 0.344101425989939f, 0.9394592236021899f, 0.3426607173119944f, 0.9399837530340139f, 0.3412192023202824f, 0.9405060705932683f, 0.33977688440682696f, 0.9410261750508893f, 0.3383337669655413f, 0.9415440651830208f, 0.33688985339222005f, 0.9420597397710173f, 0.33544514708453166f, 0.9425731976014469f, 0.3339996514420095f, 0.9430844374660935f, 0.33255336986604417f, 0.9435934581619604f, 0.33110630575987643f, 0.9441002584912727f, 0.32965846252858755f, 0.9446048372614803f, 0.32820984357909266f, 0.9451071932852606f, 0.32676045232013173f, 0.9456073253805213f, 0.325310292162263f, 0.9461052323704033f, 0.32385936651785296f, 0.9466009130832835f, 0.32240767880106996f, 0.9470943663527772f, 0.3209552324278752f, 0.9475855910177411f, 0.31950203081601575f, 0.9480745859222762f, 0.31804807738501506f, 0.9485613499157303f, 0.31659337555616585f, 0.9490458818527006f, 0.31513792875252244f, 0.9495281805930367f, 0.3136817403988915f, 0.950008245001843f, 0.31222481392182505f, 0.9504860739494817f, 0.3107671527496115f, 0.9509616663115751f, 0.3093087603122687f, 0.9514350209690083f, 0.307849640041535f, 0.9519061368079322f, 0.3063897953708611f, 0.9523750127197659f, 0.3049292297354024f, 0.9528416476011987f, 0.30346794657201137f, 0.9533060403541938f, 0.3020059493192282f, 0.9537681898859903f, 0.3005432414172734f, 0.9542280951091057f, 0.2990798263080405f, 0.9546857549413383f, 0.2976157074350863f, 0.9551411683057707f, 0.29615088824362396f, 0.9555943341307711f, 0.2946853721805143f, 0.9560452513499964f, 0.2932191626942587f, 0.9564939189023951f, 0.2917522632349894f, 0.9569403357322089f, 0.2902846772544623f, 0.9573845007889759f, 0.2888164082060495f, 0.9578264130275329f, 0.28734745954472957f, 0.9582660714080177f, 0.2858778347270807f, 0.9587034748958716f, 0.2844075372112718f, 0.9591386224618419f, 0.2829365704570554f, 0.9595715130819845f, 0.28146493792575805f, 0.9600021457376658f, 0.2799926430802734f, 0.9604305194155658f, 0.27851968938505306f, 0.9608566331076797f, 0.27704608030609995f, 0.9612804858113206f, 0.27557181931095825f, 0.9617020765291225f, 0.27409690986870633f, 0.9621214042690416f, 0.272621355449949f, 0.9625384680443592f, 0.27114515952680807f, 0.9629532668736839f, 0.2696683255729152f, 0.963365799780954f, 0.2681908570634031f, 0.9637760657954398f, 0.2667127574748984f, 0.9641840639517457f, 0.2652340302855119f, 0.9645897932898126f, 0.2637546789748315f, 0.9649932528549203f, 0.2622747070239136f, 0.9653944416976894f, 0.26079411791527557f, 0.9657933588740837f, 0.25931291513288635f, 0.9661900034454126f, 0.25783110216215893f, 0.9665843744783331f, 0.2563486824899429f, 0.9669764710448521f, 0.2548656596045146f, 0.9673662922223285f, 0.25338203699557027f, 0.9677538370934755f, 0.2518978181542169f, 0.9681391047463624f, 0.2504130065729653f, 0.9685220942744173f, 0.24892760574572026f, 0.9689028047764289f, 0.24744161916777344f, 0.9692812353565485f, 0.2459550503357946f, 0.9696573851242924f, 0.2444679027478242f, 0.970031253194544f, 0.24298017990326398f, 0.9704028386875555f, 0.24149188530286927f, 0.9707721407289504f, 0.2400030224487415f, 0.9711391584497251f, 0.2385135948443185f, 0.9715038909862518f, 0.23702360599436734f, 0.9718663374802794f, 0.23553305940497546f, 0.9722264970789363f, 0.23404195858354346f, 0.9725843689347322f, 0.23255030703877533f, 0.9729399522055601f, 0.23105810828067125f, 0.9732932460546982f, 0.22956536582051887f, 0.9736442496508119f, 0.2280720831708858f, 0.9739929621679558f, 0.2265782638456101f, 0.9743393827855759f, 0.22508391135979278f, 0.9746835106885107f, 0.22358902922979f, 0.9750253450669941f, 0.2220936209732036f, 0.9753648851166569f, 0.22059769010887365f, 0.9757021300385286f, 0.21910124015686977f, 0.976037079039039f, 0.21760427463848367f, 0.9763697313300211f, 0.2161067970762196f, 0.9767000861287118f, 0.21460881099378692f, 0.9770281426577544f, 0.21311031991609136f, 0.9773539001452f, 0.2116113273692276f, 0.9776773578245099f, 0.21011183688046972f, 0.9779985149345571f, 0.20861185197826343f, 0.9783173707196277f, 0.20711137619221856f, 0.9786339244294231f, 0.20561041305309932f, 0.9789481753190622f, 0.204108966092817f, 0.979260122649082f, 0.2026070388444211f, 0.9795697656854405f, 0.20110463484209193f, 0.9798771036995176f, 0.19960175762113105f, 0.9801821359681173f, 0.19809841071795373f, 0.9804848617734694f, 0.19659459767008022f, 0.9807852804032304f, 0.1950903220161283f, 0.9810833911504866f, 0.19358558729580372f, 0.9813791933137546f, 0.19208039704989238f, 0.9816726861969831f, 0.19057475482025277f, 0.9819638691095552f, 0.18906866414980628f, 0.9822527413662894f, 0.18756212858252974f, 0.9825393022874412f, 0.1860551516634466f, 0.9828235511987052f, 0.18454773693861964f, 0.9831054874312163f, 0.18303988795514103f, 0.9833851103215512f, 0.18153160826112513f, 0.9836624192117303f, 0.18002290140569951f, 0.9839374134492189f, 0.17851377093899756f, 0.984210092386929f, 0.17700422041214886f, 0.9844804553832209f, 0.17549425337727137f, 0.9847485018019042f, 0.17398387338746385f, 0.9850142310122398f, 0.17247308399679603f, 0.9852776423889412f, 0.17096188876030136f, 0.9855387353121761f, 0.16945029123396793f, 0.9857975091675674f, 0.1679382949747312f, 0.9860539633461954f, 0.16642590354046422f, 0.9863080972445987f, 0.16491312048997006f, 0.9865599102647754f, 0.16339994938297323f, 0.9868094018141854f, 0.16188639378011188f, 0.987056571305751f, 0.1603724572429284f, 0.9873014181578584f, 0.1588581433338614f, 0.9875439417943592f, 0.15734345561623828f, 0.9877841416445722f, 0.15582839765426532f, 0.9880220171432835f, 0.15431297301302024f, 0.9882575677307495f, 0.1527971852584434f, 0.9884907928526966f, 0.15128103795733025f, 0.9887216919603238f, 0.14976453467732162f, 0.988950264510303f, 0.1482476789868962f, 0.989176509964781f, 0.14673047445536175f, 0.9894004277913804f, 0.14521292465284752f, 0.9896220174632008f, 0.14369503315029458f, 0.9898412784588205f, 0.142176803519448f, 0.9900582102622971f, 0.14065823933284924f, 0.9902728123631691f, 0.13913934416382628f, 0.990485084256457f, 0.13762012158648618f, 0.9906950254426646f, 0.13610057517570617f, 0.99090263542778f, 0.13458070850712622f, 0.9911079137232768f, 0.13306052515713918f, 0.9913108598461154f, 0.13154002870288325f, 0.9915114733187439f, 0.13001922272223335f, 0.9917097536690995f, 0.12849811079379322f, 0.9919057004306093f, 0.12697669649688598f, 0.9920993131421918f, 0.1254549834115462f, 0.9922905913482574f, 0.12393297511851219f, 0.99247953459871f, 0.12241067519921628f, 0.992666142448948f, 0.12088808723577722f, 0.9928504144598651f, 0.11936521481099134f, 0.9930323501978514f, 0.11784206150832502f, 0.9932119492347945f, 0.11631863091190486f, 0.9933892111480807f, 0.11479492660651025f, 0.9935641355205953f, 0.11327095217756435f, 0.9937367219407246f, 0.11174671121112666f, 0.9939069700023561f, 0.11022220729388318f, 0.9940748793048794f, 0.10869744401313867f, 0.9942404494531879f, 0.10717242495680887f, 0.9944036800576791f, 0.1056471537134107f, 0.9945645707342554f, 0.10412163387205473f, 0.9947231211043257f, 0.10259586902243627f, 0.9948793307948056f, 0.10106986275482786f, 0.9950331994381186f, 0.09954361866006943f, 0.9951847266721968f, 0.09801714032956077f, 0.9953339121404823f, 0.09649043135525259f, 0.9954807554919269f, 0.09496349532963906f, 0.9956252563809943f, 0.09343633584574791f, 0.9957674144676598f, 0.0919089564971327f, 0.9959072294174117f, 0.09038136087786501f, 0.996044700901252f, 0.08885355258252468f, 0.9961798285956969f, 0.08732553520619221f, 0.996312612182778f, 0.08579731234443988f, 0.9964430513500426f, 0.08426888759332411f, 0.9965711457905548f, 0.0827402645493758f, 0.9966968952028961f, 0.08121144680959239f, 0.9968202992911657f, 0.07968243797143013f, 0.9969413577649822f, 0.0781532416327943f, 0.997060070339483f, 0.07662386139203162f, 0.9971764367353262f, 0.07509430084792128f, 0.9972904566786902f, 0.07356456359966745f, 0.9974021299012753f, 0.07203465324688942f, 0.9975114561403035f, 0.07050457338961401f, 0.9976184351385196f, 0.06897432762826673f, 0.9977230666441916f, 0.0674439195636641f, 0.9978253504111116f, 0.06591335279700392f, 0.997925286198596f, 0.06438263092985741f, 0.9980228737714862f, 0.06285175756416142f, 0.9981181129001492f, 0.061320736302208655f, 0.9982110033604782f, 0.05978957074664001f, 0.9983015449338929f, 0.05825826450043573f, 0.9983897374073402f, 0.05672682116690778f, 0.9984755805732948f, 0.05519524434969003f, 0.9985590742297593f, 0.05366353765273068f, 0.9986402181802653f, 0.05213170468028332f, 0.9987190122338729f, 0.05059974903689934f, 0.9987954562051724f, 0.04906767432741813f, 0.9988695499142836f, 0.04753548415695926f, 0.9989412931868569f, 0.046003182130914644f, 0.9990106858540734f, 0.044470771854938744f, 0.9990777277526454f, 0.04293825693494096f, 0.9991424187248169f, 0.04140564097707672f, 0.9992047586183639f, 0.039872927587739845f, 0.9992647472865944f, 0.03834012037355279f, 0.9993223845883495f, 0.03680722294135899f, 0.9993776703880028f, 0.03527423889821395f, 0.9994306045554617f, 0.03374117185137764f, 0.999481186966167f, 0.032208025408304704f, 0.9995294175010931f, 0.030674803176636584f, 0.9995752960467492f, 0.029141508764193743f, 0.9996188224951786f, 0.02760814577896582f, 0.9996599967439592f, 0.02607471782910404f, 0.9996988186962042f, 0.024541228522912267f, 0.9997352882605617f, 0.02300768146883941f, 0.9997694053512153f, 0.02147408027546961f, 0.9998011698878843f, 0.0199404285515146f, 0.9998305817958234f, 0.01840672990580482f, 0.9998576410058239f, 0.016872987947281773f, 0.9998823474542126f, 0.01533920628498822f, 0.9999047010828529f, 0.013805388528060349f, 0.9999247018391445f, 0.012271538285719944f, 0.9999423496760239f, 0.01073765916726457f, 0.9999576445519639f, 0.00920375478205996f, 0.9999705864309741f, 0.007669828739531077f, 0.9999811752826011f, 0.006135884649154516f, 0.9999894110819284f, 0.004601926120448672f, 0.9999952938095762f, 0.003067956762966138f, 0.9999988234517019f, 0.001533980186284766f};
        calc_power_in_place(accel, &qx_rfft_twiddle_4096, n);
        break;
#endif

    default:
        /* You screwed up and provided an unsupported length! */
        while(1);
    }

    // Store power, and take weighted average
    for (i = 2; i < n; i += 2) {
        pwr[i/2-1] = accel[i] * one_over_samp_freq * weight + pwr[i/2-1] * (1 - weight);
    }
    // The nyquist (last) frequency coefficient is packed in with the DC component
    pwr[n/2-1] = accel[1] * one_over_samp_freq * weight + pwr[n/2-1] * (1 - weight);
}

#ifdef PYTHON_FEATS
static void free_fft(Fft* fft) {
    free(fft -> pwr);
    free(fft);
}
#endif
#endif

/*
 *   SURROUNDING PEAK SIGNAL
 */


static void set_feats_peak(Features* F_ptr, sigval_t* sig, int n, float samp_freq) {

    int i;

    /* index of signal with maximum */
    int imax = 0;
    fval_t smax = fabsf(sig[imax]);
    fval_t abs_sig;
    for (i = 0; i < n; i++) {
        abs_sig = fabsf(sig[i]);
        if (abs_sig > smax) {
            imax = i;
            smax = abs_sig;
        }
    }

    int istop = n-2;
    int nzc = 0; /* number of zero crosses */
    int nxt_sign;
    float* ipeaks = FLOAT_BUF;
    int num_peaks = 0;
    fval_t avg_peak = 0, avg_ipeak = 0; /* average peak signal and index */
    for (i = imax; i < istop; i++) {
        nxt_sign = sign(sig[i+1]);

        /* zero-crossing detection */
        if ( (sign(sig[i]) != nxt_sign) && (nxt_sign != 0) ) {
            nzc += 1;
        }

        /* peak detection */
        if ((i > imax) && is_peak(sig[i-1], sig[i], sig[i+1]) && (num_peaks < MAX_NUM_PEAKS)) {
            ipeaks[num_peaks] = i;
            num_peaks += 1;

            /* avg peak and avg peak index */
            avg_peak += sig[i];
            avg_ipeak += (fval_t)i;
        }
    }

    fval_t zcr = 0;
    if (nzc > 0) {
        zcr = (fval_t)nzc / ( (fval_t)(istop-imax+1) / (fval_t)samp_freq );
    }
    SET_FEAT(F_ptr, g_feature_prefix, "rawpeak_zcr", "", zcr);

    /* linear fit to peaks not defined if no peaks */
    /* cannot fit line with only one data point */
    if (num_peaks <= 1) {
        SET_FEAT(F_ptr, g_feature_prefix, "rawpeak_lin_fit", "slope", 0);
        SET_FEAT(F_ptr, g_feature_prefix, "rawpeak_lin_fit", "resid_avg", 0);
        SET_FEAT(F_ptr, g_feature_prefix, "rawpeak_lin_fit", "resid_std", 0);
        return;
    }

    avg_peak /= (fval_t)num_peaks;
    avg_ipeak /= (fval_t)num_peaks;

    /* fit line to peaks */

    /* calc variance in peak indexes and covariance between peaks and indexes */
    fval_t var = 0, cov = 0; /* covariance between signal and time and variance in time */
    fval_t v;
    for (i = 0; i < num_peaks; i++) {
        v = ipeaks[i] - avg_ipeak;
        var += v * v;
        cov += ipeaks[i] * sig[(int)ipeaks[i]];
    }

    cov = cov - (fval_t)num_peaks * avg_ipeak * avg_peak;
    fval_t m = cov / var; /* slope of line */
    fval_t b = avg_peak - m * avg_ipeak;
    SET_FEAT(F_ptr, g_feature_prefix, "rawpeak_lin_fit", "slope", m);
    if(num_peaks == 2) {
        /* perfect line */
        SET_FEAT(F_ptr, g_feature_prefix, "rawpeak_lin_fit", "resid_avg", 0);
        SET_FEAT(F_ptr, g_feature_prefix, "rawpeak_lin_fit", "resid_std", 0);
        return;
    }

    /* residuals in linear fit */
    fval_t *resids = FLOAT_BUF + MAX_NUM_PEAKS;
    if(resids == NULL) {
        SET_FEAT(F_ptr, g_feature_prefix, "rawpeak_lin_fit", "resid_avg", 0);
        SET_FEAT(F_ptr, g_feature_prefix, "rawpeak_lin_fit", "resid_std", 0);
        return;
    }
    fval_t avg_resid = 0;
    for (i = 0; i < num_peaks; i++) {
        resids[i] = fabsf(m*ipeaks[i]+b - sig[(int)ipeaks[i]]);
        avg_resid += resids[i];
    }
    avg_resid /= (fval_t)num_peaks;

    /* standard deviation of  */
    fval_t std_resid = 0;
    for (i = 0; i < num_peaks; i++) {
        v = resids[i] - avg_resid;
        std_resid += v * v;
    }
    std_resid = sqrtf(std_resid / (fval_t)num_peaks);

    SET_FEAT(F_ptr, g_feature_prefix, "rawpeak_lin_fit", "resid_avg", avg_resid);
    SET_FEAT(F_ptr, g_feature_prefix, "rawpeak_lin_fit", "resid_std", std_resid);
}

/* * * * * * * * * *
 *   H E L P E R   *
 * * * * * * * * * */

/*
 *   input: stats, pointer to Stats struct to fill in stat values
 *          sig, array of fval_t containing signal
 *          n, length of sig
 */
static void set_stats(Stats* stats, sigval_t* sig, int n, bool calc_higher_order) {
    fval_t mina = sig[0]; /* minimum */
    fval_t maxa = sig[0]; /* maximum */
    fval_t cura; /* current signal */

    fval_t m1 = 0; /* first  moment */

    fval_t len = (fval_t)n; /* length of signal as a float */
    fval_t inv_len = 1.0f / len;

    int i;
    /* calculate max, min, and moments */
    for (i = 0; i < n; i++) {
        cura = sig[i];
        if (cura < mina) {
            mina = cura;
        } else if (cura > maxa) {
            maxa = cura;
        }
        m1 += cura;
    }
    fval_t sum = m1;
    m1 *= inv_len;

    fval_t rng = maxa - mina;
    if (rng == 0) rng = 1;

    stats -> min = mina;
    stats -> max = maxa;
    stats -> rng = rng;
    stats -> sum = sum;
    stats -> avg = m1;

    /* This separate block is used to calculate these higher-order stats only for the raw
     * signal. See EM-1236 for more details. */
    if (calc_higher_order) {
        fval_t mm2 = 0, mm3 = 0; /* second and third momements centered around the mean */

        /* calculate moments centered around mean */
        fval_t v, v2;
        for (i = 0; i < n; i++) {
            v = sig[i] - m1;
            v2 = v * v;
            mm2 += v2;
            mm3 += v2 * v;
        }
        mm2 *= inv_len;
        mm3 *= inv_len;

        fval_t std = sqrtf(mm2);
        if (std == 0) std = 1;

        stats -> std = std;
        if (mm2 * std == 0) {
            stats -> skew = 0;
        } else {
            stats -> skew = mm3 / (mm2 * std);
        }
    } else {
        stats -> std = 0;
        stats -> skew = 0;
    }
}

/* subtracts mean from input accel signal */
static inline void UNROLL_LOOPS mean_subtract(sigval_t* accel, fval_t mean, size_t n) {
    for (size_t i = 0; i < n; i++) {
        accel[i] -= mean;
    }
}

/* sums all values from nums[lo] to nums[hi-1]. does not include nums[hi].
 *
 * Previously we had implemented the more accurate Kahan summation method here.
 * However, due to the -ffast-math optimization, it was
 * being optimized away to a regular sum. Based on our feature comparison
 * the more precise summation method was not necessary anyway.
 *
 * If, in the future, this method does not provide the required accuracy,
 * we can re-implement Kahan summation, but care must taken that
 * -ffast-math is not enabled.*/
static float float_sum(float* nums, int lo, int hi) {
    float sum = 0;
    for (int i = lo; i < hi; i++) {
      sum += nums[i];
    }
    return sum;
}

#else /* !USE_FIXEDPOINT */

/** Fixed-point feature computation.
 *
 * Although all features are stored as fixed-point, some internal
 * computations are carried out using floating-point arithmetic.
 *
 * Basic rules:
 * - float->int is ~50 cycles, int->float is ~150 cycles
 * - floating-point addition and multiplication are expensive
 * - floating point division is on par or cheaper than integer division
 * - the original signal input is understood to be in the range [-1, 1) in Q0.15 format.
 */

/* Temporary buffer for various operations.
 * Used for:
 * - peaks in set_feat_peaks (as int array)
 * - for temporary computation of the FFT
 * - to store the FFT power spectrum
 */
static int32_t TMP_BUF[MAX_FFT_SIG_LEN + 2] __attribute__((aligned(8)));

/*** Reimplementations of built-in functions.
 * Why do we have to reimplement functions that already exist in GCC?
 * Because GCC built-in functions are amazingly slow and were not carefully optimized.
 * We can also optimize for less accuracy in corner cases in exchange for much higher speeds
 * in the general case, or implement behaviour that is more useful for us.
 * Improvements here can be very substantial, ranging from 2x (on __aeabi_lmul) to over 8x (in
 * i64_to_f32 vs. __aeabi_l2f).
 */
#if defined(__arm__) && (defined(ARM_MATH_CM0) || defined(ARM_MATH_CM0PLUS))
/* Only enable these assembly functions for Cortex-M0. These should *not* be enabled for any
 * CPU with 64-bit multiply instructions (smull/umull).
 */

/* Faster 64x64-bit multiplies, from http://www.purposeful.co.uk/arm_rtabi/arm_rtabi.html
 * Used under a public-domain license: http://www.purposeful.co.uk/tfl/
 * This version is 2x as fast as GCC's default implementation (~37 cyc vs. ~70 cyc including call overhead)
 */
int64_t __attribute__((naked)) __aeabi_lmul(int64_t a, int64_t b) {
  __asm__ volatile (
    ".syntax unified\n"
    "muls    r1, r2\n"
    "muls    r3, r0\n"
    "adds    r1, r3\n"
    "mov     r12, r1\n"
    "lsrs    r1, r2, 16\n"
    "uxth    r3, r0\n"
    "muls    r3, r1\n"
    "push    {r4}\n"
    "lsrs    r4, r0, 16\n"
    "muls    r1, r4\n"
    "uxth    r2, r2\n"
    "uxth    r0, r0\n"
    "muls    r0, r2\n"
    "muls    r2, r4\n"
    "lsls    r4, r3, 16\n"
    "lsrs    r3, 16\n"
    "adds    r0, r4\n"
    "pop     {r4}\n"
    "adcs    r1, r3\n"
    "lsls    r3, r2, 16\n"
    "lsrs    r2, 16\n"
    "adds    r0, r3\n"
    "adcs    r1, r2\n"
    "add     r1, r12\n"
    "bx lr\n"
  );
}

static uint64_t __attribute__((naked)) square64(int64_t x) {
  __asm__ volatile (
    ".syntax unified\n"
    "muls r1, r0\n"
    "lsls r1, r1, 1\n"
    "lsrs r2, r0, 16\n"
    "muls r2, r2\n"
    "adds r1, r2\n"
    "lsrs r2, r0, 16\n"
    "uxth r0, r0\n"
    "muls r2, r0\n"
    "muls r0, r0\n"
    "lsls r3, r2, 16\n"
    "lsrs r2, r2, 16\n"
    "adds r0, r3\n"
    "adcs r1, r2\n"
    "adds r0, r3\n"
    "adcs r1, r2\n"
    "bx lr\n"
  );
}
#else
static inline uint64_t square64(int64_t x) {
    return x * x;
}
#endif

static int32_t u32_to_f32_helper(uint32_t x, int bias, int e) {
  uint32_t res;
  uint32_t highbit = 0x80000000U >> e;
  x -= highbit;
  if(e < 8) {
    x += 1 << (7 - e);
    if(__builtin_expect(x >= highbit, 0)) {
      x -= highbit;
      e--;
    }
    res = (x >> (8 - e));
  } else {
    res = (x << (e - 8));
  }
  res |= (unsigned)(127 + 31 + bias - e) << 23;
  return (int32_t)res;
}

/* int32_t to float conversion with built-in bias (saves a call to scalbnf)
 * and round-half-away-from-zero rounding.
 * This takes ~71 cycles in the typical path on the Nano 33 IoT.
 * Compare this to ~170 cycles using GCC __aeabi_i2f + scalbnf. */
static float i32_to_f32(int32_t v, int bias) {
  if(!v) return v;

  int32_t res = 0;
  if(v < 0) {
    res |= 0x80000000;
    v = -v;
  }
  int e = __builtin_clz((uint32_t)v);
  res |= u32_to_f32_helper((uint32_t)v, bias, e);

  /* memcpy is the only safe way to interpret the bits of an integer
   * as a float.
   * Thankfully, any sane compiler (ARM-GCC included) optimizes this
   * to a no-op.
   */
  float rf;
  memcpy(&rf, &res, sizeof(float));
  return rf;
}

/* int64_t to float conversion with built-in bias (saves a call to scalbnf).
 * This takes ~125 cycles in the typical path on the Nano 33 IoT.
 * Compare this to ~1033 cycles (!) using GCC __aeabi_l2f + scalbnf. */
static float i64_to_f32(int64_t v, int bias) {
  if(!v) return v;

  int32_t res = 0;
  if(v < 0) {
    res |= 0x80000000;
    v = -v;
  }
  int e = __builtin_clzll((uint64_t)v);

  uint32_t x;
  if(e < 32) {
    /* Push all non-zero bits into the low half.
       This is slightly sloppy - we should round for perfect
       accuracy, but not rounding here introduces only 2^-8
       relative error while avoiding potential double-rounding */
    x = (uint32_t)(((uint64_t)v) >> (32 - e));
    bias += (32 - e);
    e = 0;
  } else {
    /* All 32 high bits are already zero */
    x = (uint32_t)v;
    e -= 32;
  }

  res |= u32_to_f32_helper(x, bias, e);

  float rf;
  memcpy(&rf, &res, sizeof(float));
  return rf;
}

/* float to int32_t conversion with built-in bias and rounding (saving calls to
 * roundf and scalbnf). This implementation also clamps outputs to [-2^32, 2^32);
 * the normal float->int conversion outputs -2^32 for all out-of-range floats
 * which is unhelpful.
 * This takes ~33 cycles in the typical path on the Nano 33 IoT.
 * Compare this to ~128 cycles using GCC __aeabi_f2i + scalbnf + roundf. */
static int32_t f32_to_i32(float f, int bias) {
  int32_t fi;
  memcpy(&fi, &f, sizeof(int32_t));

  int e = ((fi >> 23) & 0xff) - 127 + bias;
  if(e <= -2) {
    return 0;
  } else if(e >= 31) {
    /* clamp to max/min value */
    if(fi >> 31) {
      return (int32_t)-0x80000000;
    } else {
      return 0x7fffffff;
    }
  }

  uint32_t m = (1 << 23) | (fi & 0x7fffff);
  if(e >= 22) {
    m = m << (e - 22);
  } else {
    m = m >> (22 - e);
  }
  m = (m + 1) >> 1;
  if(fi >> 31)
    return -(int32_t)m;
  return (int32_t)m;
}

#define ALWAYS_INLINE inline __attribute__((always_inline))
static ALWAYS_INLINE fval_t fvabs(fval_t v) {
    if(v < 0) return -v;
    return v;
}

/* Convert Qa.b into Qa.(b-x) */
#define RSHIFT(n,x) (((n) + (1 << ((x) - 1))) >> (x))

/*
 *   input: stats, pointer to Stats struct to fill in stat values
 *          sig, array of Q0.15 sigval_t containing signal
 *          sig_mean, Q0.30 signal mean value
 *          n, length of sig
 */
static void set_stats(Stats* stats, sigval_t* sig, fval_t sig_mean, int n) {
    fval_t mina = 0; /* minimum, mean-subtracted; Q1.30 */
    fval_t maxa = 0; /* maximum, mean-subtracted; Q1.30 */
    int64_t sum2 = 0; /* Q2.30 */
    int64_t sum3 = 0; /* Q3.45 */

    fval_t sig_mean_hi = RSHIFT(sig_mean, 15); /* Q0.15 */
    fval_t sig_mean_lo = sig_mean - (sig_mean_hi << 15); /* Q0.30, |rawval| <= 2**14 */

    /* calculate max, min, and moments */
    for (int i = 0; i < n; i++) {
        fval_t cura = sig[i] - sig_mean_hi; /* Q1.15 */
        if (cura < mina) {
            mina = cura;
        } else if (cura > maxa) {
            maxa = cura;
        }

        uint32_t cura_sq = cura * cura; /* Q2.30, unsigned */
        sum2 += cura_sq;
        /* current value cubed, Q3.45 */
        int64_t cura_cb = ((uint64_t)cura_sq) * cura;
        sum3 += cura_cb;
    }
    /* correct for sig_mean_lo error */
    mina = (mina << 15) - sig_mean_lo; /* Q1.30 */
    maxa = (maxa << 15) - sig_mean_lo; /* Q1.30 */
    stats -> max_dev_from_mean = max(fvabs(mina), fvabs(maxa));
    /* maxa - mina = (sig[i] << 15) - (sig[j] << 15) = (Q0.30 - Q0.30) so this doesn't overflow */
    fval_t rng = maxa - mina;
    if (rng == 0) rng = 1 << 15; /* signal epsilon: 1 unit in Q0.15 */
    stats -> min = mina; /* Q1.30 */
    stats -> max = maxa; /* Q1.30 */
    stats -> rng = rng; /* Q1.30 */

    /* correct for sig_mean_lo via algebraic expansion
       note that sig_mean_lo is the mean of (sig - sig_mean_hi) */
    sum3 -= RSHIFT(3 * sum2 * sig_mean_lo, 15);
    /* note: the following two terms are likely insignificant and could be dropped,
       but don't cost much because they're not in a loop */
    int64_t e_sq = sig_mean_lo * sig_mean_lo; /* Q0.60 */
    sum3 += RSHIFT(e_sq * (sig_mean_lo * 2 * n), 45);
    sum2 -= RSHIFT(e_sq * n, 30);

    /* perform remaining calculations in floating-point for accuracy
       this is relatively cheap because these calculations are not in a loop */
    float invn = 1.0f / n;
    float mm2 = i64_to_f32(sum2, -30) * invn;
    float mm3 = i64_to_f32(sum3, -45) * invn;

    float std = sqrtf(mm2);
    if (std == 0) std = (1.0f / 32768); /* signal epsilon */

    stats -> sum = 0; /* by definition, once sig_mean is removed */
    stats -> avg = 0; /* by definition, once sig_mean is removed */
    stats -> std = f32_to_i32(std, 30); /* Q1.30 */
    float den = mm2 * std;
    if (den == 0) {
        stats -> skew = 0;
    } else {
        stats -> skew = f32_to_i32(mm3 / den, 15); /* Q16.15 */
    }
    stats -> std_by_range = f32_to_i32(std / i32_to_f32(rng, -30), 30); /* Q1.30 */
}

static void set_feats_rawstat_summary(Features* F_ptr, sigval_t* accel, fval_t sig_mean, int n) {
#ifdef PYTHON_FEATS
    char* infix = "rawstat";
#endif

    Stats s;
    set_stats(&s, accel, sig_mean, n);

    SET_FEAT(F_ptr, g_feature_prefix, infix, "range", s.rng); /* Q1.30 */
    SET_FEAT(F_ptr, g_feature_prefix, infix, "std", s.std); /* Q2.30 */
    SET_FEAT(F_ptr, g_feature_prefix, infix, "skew", s.skew); /* Q16.15 */
    SET_FEAT(F_ptr, g_feature_prefix, infix, "std_by_range", s.std_by_range); /* Q16.15 */
    SET_FEAT(F_ptr, g_feature_prefix, infix, "max_dev_from_mean", s.max_dev_from_mean); /* Q1.30 */
}

static int64_t autocorr_helper(sigval_t *sig, int i, int n) {
    int64_t autsum = 0;
#ifdef __arm__
    /* This is a critical inner loop that is called once per autocorr feature,
      per axis, across the entire signal. It must be fast. GCC does not optimize
      this well - the best I got was ~22 cycles per iteration.
      The version below is 12 cycles per iteration on M0+. */
    fval_t a, b;
    __asm__ (
        ".syntax unified        \n"
        "1:                     \n"
        // Load sig[j] and sig[i+j] (note: ldrsh for Thumb only allows [rB, rI] syntax)
        "ldrsh  %[a], [%[ptr], %[zero]]\n"
        "ldrsh  %[b], [%[ptr], %[off]]\n"
        // Increment base pointer (equiv. to incrementing j)
        "adds    %[ptr], #2     \n"
        // Multiply a*b, store result in a
        "muls %[a], %[b]        \n"
        // Add signed product to 64-bit autsum
        "asrs %[b], %[a], #31   \n"
        "adds %Q[autsum], %[a]  \n"
        "adcs %R[autsum], %[b]  \n"
        // Loop while we haven't hit the end
        "cmp %[end], %[ptr]     \n"
        "bne 1b                 \n"
        // Most of our registers need "l" constraints so they end up in r0..r7
        // (many instructions only take "lo" registers)
        : [autsum]"+l"(autsum),
          [ptr]"+l"(sig),
          [a]"=&l"(a),
          [b]"=&l"(b)
        : [end]"r"(&sig[n-i]),
          [off]"l"(2*i),
          [zero]"l"(0)
        : "cc"
    );
#else
    for (int j = 0; i+j < n; j++) {
        fval_t a = sig[i+j];
        fval_t b = sig[j];
        autsum += a * b;
    }
#endif
    return autsum;
}

static void set_feats_autocorr(Features* F_ptr, sigval_t* sig, fval_t sig_mean, int n) {
    int len_aut = f32_to_i32(i32_to_f32(n, 0) * AUTOCORR_RATIO, 0);
    if (len_aut > MAX_N_AUTOCORR_FEATS){
        len_aut = MAX_N_AUTOCORR_FEATS;
    } else if (len_aut < MIN_N_AUTOCORR_FEATS){
        len_aut = MIN_N_AUTOCORR_FEATS;
    }

    int64_t sigsum2 = 0; /* Qx.30 */
    int32_t sigsum = 0; /* Qx.15 */
    for(int i=0; i<n; i++) {
        fval_t cura = sig[i];
        sigsum2 += cura * cura;
        sigsum += cura;
    }
    /* correct for mean */
    sigsum2 = sigsum2 * n - square64(sigsum); /* still Qx.30, but multiplied by n */
    float inv_max_aut = (sigsum2 == 0) ? 0 : i32_to_f32(n, 0) / i64_to_f32(sigsum2, -30);

#ifdef PYTHON_FEATS
    int num_digits = get_num_digits(len_aut);
    char feat_num[10];
#endif
    int32_t autadj = 2 * sigsum; /* Qx.15 */
    for (int i = 1; i <= len_aut; i++) {
        /* calculate autocorrelation using algebraic expansion so we can use 16-bit multiplication
           in the critical inner loop. */
        /* autadj = sum{a + b} */
        autadj -= sig[i - 1];
        autadj -= sig[n - i];

        int64_t autsum = autocorr_helper(sig, i, n); /* Qx.30 */
        /* correct for mean */
        autsum <<= 15; /* Qx.45 */
        /* middle term */
        autsum -= ((int64_t)sig_mean) * autadj;
        /* mean^2 term */
        autsum += RSHIFT(square64(sig_mean), 15) * (n - i);
        float cur_aut = i64_to_f32(autsum, -45) * inv_max_aut;

#ifdef PYTHON_FEATS
        set_feat_num(feat_num, i, num_digits);
#endif
        SET_FEAT(F_ptr, g_feature_prefix, "autocorr", feat_num, f32_to_i32(cur_aut, 30)); /* Q1.30 */
    }
}

static void linfit_helper(sigval_t *sig, int32_t *ipeaks, int n, float *m, float *b, int32_t *sxp) {
    int32_t sx = 0; /* Qx.0 */
    int32_t sy = 0; /* Qx.15 */
    int64_t sxy = 0; /* Qx.15 */
    int64_t sxx = 0; /* Qx.0 */
    for(int i=0; i<n; i++) {
        int x = ipeaks[i]; /* Qx.0 */
        fval_t y = sig[x]; /* Qx.15 */
        sx += x;
        sy += y;
        /* raw values of x and y are <=2^15 in abs value */
        sxy += x * y;
        sxx += x * x;
    }
    float num = i64_to_f32(n * sxy - ((int64_t)sx) * sy, 0);
    float den = i64_to_f32(n * sxx - ((int64_t)sx) * sx, 0);
    *m = num / den; /* slope, scaled by 2^15 */
    *b = sy - *m * sx; /* intercept, multiplied by n, scaled by 2^15 */
    *sxp = sx;
}

static void set_feats_peak(Features* F_ptr, sigval_t* sig, fval_t sig_mean, int n, float samp_freq) {
    /* index of signal with maximum */
    int imax = 0;
    fval_t smax = 0; /* Q1.30 */
    for (int i = 0; i < n; i++) {
        fval_t si = fvabs((sig[i] << 15) - sig_mean);
        if (si > smax) {
            imax = i;
            smax = si;
        }
    }

    int istop = n-2;

    int nzc = 0; /* number of zero crosses */
    int32_t* ipeaks = TMP_BUF;
    int num_peaks = 0;

    fval_t prev = (sig[imax] << 15) - sig_mean; /* Q1.30 */
    fval_t cur = prev; /* Q1.30 */
    fval_t cur_sign = sign(cur);

    for (int i = imax; i < istop; i++) {
        fval_t next = (sig[i+1] << 15) - sig_mean; /* Q1.30 */
        int next_sign = sign(next);

        /* zero-crossing detection */
        if ( (cur_sign != next_sign) && (next_sign != 0) ) {
            nzc += 1;
        }

        /* peak detection */
        if ((i > imax) && is_peak(prev, cur, next) && (num_peaks < MAX_NUM_PEAKS)){
            ipeaks[num_peaks++] = i;
        }
        prev = cur;
        cur = next;
        cur_sign = next_sign;
    }

    float zcr = 0;
    if(nzc > 0) {
        zcr = samp_freq * i32_to_f32(nzc, 0) / i32_to_f32(istop - imax + 1, 0);
    }
    SET_FEAT(F_ptr, g_feature_prefix, "rawpeak_zcr", "", f32_to_i32(zcr, 15)); /* Q16.15 */

    /* linear fit to peaks not defined if no peaks */
    /* cannot fit line with only one data point */
    if (num_peaks <= 1) {
        SET_FEAT(F_ptr, g_feature_prefix, "rawpeak_lin_fit", "slope", 0);
        SET_FEAT(F_ptr, g_feature_prefix, "rawpeak_lin_fit", "resid_avg", 0);
        SET_FEAT(F_ptr, g_feature_prefix, "rawpeak_lin_fit", "resid_std", 0);
        return;
    }

    float m, b;
    int32_t sx;
    linfit_helper(sig, ipeaks, num_peaks, &m, &b, &sx);
    float inv_num_peaks = 1.0f / i32_to_f32(num_peaks, 0);
    /* Adjust b to the intercept in the middle of the dataset.
       This ensures that b is within the Qx.15 range. */
    int32_t xadj = sx / num_peaks;
    b = (b * inv_num_peaks) + m * i32_to_f32(xadj, 0);
    fval_t mi = f32_to_i32(m, 15);
    fval_t bi = f32_to_i32(b, 15);
    SET_FEAT(F_ptr, g_feature_prefix, "rawpeak_lin_fit", "slope", mi);
    if(num_peaks == 2) {
        /* perfect line */
        SET_FEAT(F_ptr, g_feature_prefix, "rawpeak_lin_fit", "resid_avg", 0);
        SET_FEAT(F_ptr, g_feature_prefix, "rawpeak_lin_fit", "resid_std", 0);
        return;
    }

    /* residuals in linear fit */
    int64_t sum_resid = 0; /* Q0.30 */
    int64_t sum_resid2 = 0; /* Q0.60 */
    for (int i = 0; i < num_peaks; i++) {
        int peak = ipeaks[i];
        int64_t resid = ((int64_t)mi) * (peak - xadj) + bi - (sig[peak] << 15);
        if(resid < 0)
            resid = -resid;
        sum_resid += resid;
        sum_resid2 += square64(resid);
    }
    int64_t avg_resid = (sum_resid + (num_peaks >> 1)) / num_peaks;
    SET_FEAT(F_ptr, g_feature_prefix, "rawpeak_lin_fit", "resid_avg", avg_resid);

    float std_resid = i64_to_f32(sum_resid2 - avg_resid * sum_resid, -60);
    std_resid = sqrtf(std_resid * inv_num_peaks);
    /* standard deviation of residuals */
    SET_FEAT(F_ptr, g_feature_prefix, "rawpeak_lin_fit", "resid_std", f32_to_i32(std_resid, 30));
}

/*** FFT ***/
#if defined(FFT_FEATURES) || defined(MFCC_FEATURES)

#ifdef WEIGHTED_AVERAGE
#error "WEIGHTED_AVERAGE is not supported for fixed-point feature pipeline"
#endif

typedef uint64_t pwr_t;

static int64_t fft_multab(int32_t val, uint32_t tab) {
    return RSHIFT(((int64_t)val) * tab, 32);
}

/* In-place conversion of an N-point complex FFT into a real power spectrum for the corresponding RFFT.
   Input: complex-valued FFT, N complex values (int32_t x int32_t)
   Output: real-valued power spectrum, N+1 real values (uint64_t) */
static uint64_t *fftpow_calc(int32_t *cfft, const int n, const uint32_t *tab1, const uint32_t *tab2) {
    uint64_t *out = (uint64_t *)cfft;
    for(int i=1; i<n/2; i++) {
        int32_t ar = cfft[i*2];
        int32_t ai = cfft[i*2 + 1];
        int32_t br = cfft[n*2 - i*2];
        int32_t bi = cfft[n*2 - i*2 + 1];

        uint32_t Mi = *tab2++;
        int64_t mid = fft_multab(ar, Mi) * bi + fft_multab(ai, Mi) * br;

        uint32_t Bi = *tab1++;
        uint64_t amag = square64(fft_multab(ar, Bi)) + square64(fft_multab(ai, Bi));
        uint64_t bmag = square64(fft_multab(br, Bi)) + square64(fft_multab(bi, Bi));

        int64_t res;
        res = (square64(ar) + square64(ai) - amag) + bmag + mid;
        res = MAX(res, 1);
        out[i] = res;
        res = amag + (square64(br) + square64(bi) - bmag) - mid;
        res = MAX(res, 1);
        out[n-i] = res;
    }

    /* special case for n/2 */
    {
        int32_t ar = cfft[n];
        int32_t ai = cfft[n+1];
        out[n/2] = square64(ar) + square64(ai);
    }

    /* special case for 0 and nyquist */
    {
        int64_t ar = cfft[0];
        int32_t ai = cfft[1];
        out[0] = square64(ar + ai);
        out[n] = square64(ar - ai);
    }
    return out;
}

/* Calculate the power spectrum of the FFT of a signal, *in-place*. The buffer must be
   two int32_t's longer than the actual signal.
   Note that we use the *complex FFT* CMSIS routines, then convert the result into a real
   power spectrum by performing a split RFFT and power spectrum computation (fftpow_calc),
   because the default CMSIS fixed-point RFFT routines don't support in-place operation
   (they also calculate 2x as many outputs as necessary for RFFT).
   n = padded signal length; must be a power of two.
   The new signal is also returned in TMP_BUF as uint64_t units (n/2+1 elements).
   The main difference between this function and the floating-point calculate_fft is that
   this version does *not* divide by samp_rate. */
static pwr_t *calculate_fft(int32_t *isig, int n) {
    pwr_t *res;

    /* The #if statements here help us compile only the tables we need - saving memory on the device. */
    switch(n) {
    /* Tables were generated as follows:

    import numpy as np

    W = lambda k, n: np.exp(-1j * 2 * np.pi * k / n)
    A = lambda k, n: 0.5 * (1 - 1j * W(k, 2*n))
    B = lambda k, n: 0.5 * (1 + 1j * W(k, 2*n))
    M = lambda k, n: -(2 * A(k, n) * B(k, n).conj()).imag

    tab1 = [int(round(np.abs(B(i, N)) * (1 << 32))) for i in range(1, N//2)]
    tab2 = [int(round(M(i, N) * (1 << 32))) for i in range(1, N//2)]
    */

#if MAX_FFT_SIG_LEN >= 32
    case 32:
        arm_cfft_q31(&arm_cfft_sR_q31_len16, isig, 0 /* ifftFlagR */, 1 /* bitReverseFlagR */);
        static const uint32_t fftpow_16_tab1[] = {3320054617, 3571134792, 3787822988, 3968032378, 4110027446, 4212440704, 4274285855};
        static const uint32_t fftpow_16_tab2[] = {4212440704, 3968032378, 3571134792, 3037000500, 2386155981, 1643612827, 837906553};
        res = fftpow_calc(isig, 16, fftpow_16_tab1, fftpow_16_tab2);
        for(int i=0; i<=16; i++)
            res[i] = RSHIFT(res[i], 24);
        break;
#endif

#if MAX_FFT_SIG_LEN >= 64
    case 64:
        arm_cfft_q31(&arm_cfft_sR_q31_len32, isig, 0 /* ifftFlagR */, 1 /* bitReverseFlagR */);
        static const uint32_t fftpow_32_tab1[] = {3182360851, 3320054617, 3449750080, 3571134792, 3683916329, 3787822988, 3882604450, 3968032378, 4043900968, 4110027446, 4166252509, 4212440704, 4248480760, 4274285855, 4289793820};
        static const uint32_t fftpow_32_tab2[] = {4274285855, 4212440704, 4110027446, 3968032378, 3787822988, 3571134792, 3320054617, 3037000500, 2724698408, 2386155981, 2024633568, 1643612827, 1246763195, 837906553, 420980412};
        res = fftpow_calc(isig, 32, fftpow_32_tab1, fftpow_32_tab2);
        for(int i=0; i<=32; i++)
            res[i] = RSHIFT(res[i], 22);
        break;
#endif

#if MAX_FFT_SIG_LEN >= 128
    case 128:
        arm_cfft_q31(&arm_cfft_sR_q31_len64, isig, 0 /* ifftFlagR */, 1 /* bitReverseFlagR */);
        static const uint32_t fftpow_64_tab1[] = {3110617535, 3182360851, 3252187232, 3320054617, 3385922125, 3449750080, 3511500034, 3571134792, 3628618433, 3683916329, 3736995171, 3787822988, 3836369162, 3882604450, 3926501002, 3968032378, 4007173558, 4043900968, 4078192482, 4110027446, 4139386683, 4166252509, 4190608739, 4212440704, 4231735252, 4248480760, 4262667143, 4274285855, 4283329896, 4289793820, 4293673732};
        static const uint32_t fftpow_64_tab2[] = {4289793820, 4274285855, 4248480760, 4212440704, 4166252509, 4110027446, 4043900968, 3968032378, 3882604450, 3787822988, 3683916329, 3571134792, 3449750080, 3320054617, 3182360851, 3037000500, 2884323748, 2724698408, 2558509031, 2386155981, 2208054473, 2024633568, 1836335144, 1643612827, 1446930903, 1246763195, 1043591926, 837906553, 630202589, 420980412, 210744057};
        res = fftpow_calc(isig, 64, fftpow_64_tab1, fftpow_64_tab2);
        for(int i=0; i<=64; i++)
            res[i] = RSHIFT(res[i], 20);
        break;
#endif

#if MAX_FFT_SIG_LEN >= 256
    case 256:
        arm_cfft_q31(&arm_cfft_sR_q31_len128, isig, 0 /* ifftFlagR */, 1 /* bitReverseFlagR */);
        static const uint32_t fftpow_128_tab1[] = {3074040487, 3110617535, 3146726136, 3182360851, 3217516315, 3252187232, 3286368382, 3320054617, 3353240863, 3385922125, 3418093478, 3449750080, 3480887161, 3511500034, 3541584088, 3571134792, 3600147697, 3628618433, 3656542712, 3683916329, 3710735162, 3736995171, 3762692404, 3787822988, 3812383140, 3836369162, 3859777440, 3882604450, 3904846754, 3926501002, 3947563934, 3968032378, 3987903250, 4007173558, 4025840401, 4043900968, 4061352537, 4078192482, 4094418266, 4110027446, 4125017671, 4139386683, 4153132319, 4166252509, 4178745276, 4190608739, 4201841112, 4212440704, 4222405917, 4231735252, 4240427302, 4248480760, 4255894413, 4262667143, 4268797931, 4274285855, 4279130086, 4283329896, 4286884652, 4289793820, 4292056960, 4293673732, 4294643893};
        static const uint32_t fftpow_128_tab2[] = {4293673732, 4289793820, 4283329896, 4274285855, 4262667143, 4248480760, 4231735252, 4212440704, 4190608739, 4166252509, 4139386683, 4110027446, 4078192482, 4043900968, 4007173558, 3968032378, 3926501002, 3882604450, 3836369162, 3787822988, 3736995171, 3683916329, 3628618433, 3571134792, 3511500034, 3449750080, 3385922125, 3320054617, 3252187232, 3182360851, 3110617535, 3037000500, 2961554089, 2884323748, 2805355999, 2724698408, 2642399561, 2558509031, 2473077351, 2386155981, 2297797281, 2208054473, 2116981616, 2024633568, 1931065957, 1836335144, 1740498191, 1643612827, 1545737412, 1446930903, 1347252816, 1246763195, 1145522571, 1043591926, 941032661, 837906553, 734275721, 630202589, 525749847, 420980412, 315957395, 210744057, 105403774};
        res = fftpow_calc(isig, 128, fftpow_128_tab1, fftpow_128_tab2);
        for(int i=0; i<=128; i++)
            res[i] = RSHIFT(res[i], 18);
        break;
#endif

#if MAX_FFT_SIG_LEN >= 512
    case 512:
        arm_cfft_q31(&arm_cfft_sR_q31_len256, isig, 0 /* ifftFlagR */, 1 /* bitReverseFlagR */);
        static const uint32_t fftpow_256_tab1[] = {3055578014, 3074040487, 3092387225, 3110617535, 3128730733, 3146726136, 3164603066, 3182360851, 3199998822, 3217516315, 3234912670, 3252187232, 3269339351, 3286368382, 3303273682, 3320054617, 3336710553, 3353240863, 3369644927, 3385922125, 3402071844, 3418093478, 3433986423, 3449750080, 3465383855, 3480887161, 3496259414, 3511500034, 3526608449, 3541584088, 3556426389, 3571134792, 3585708745, 3600147697, 3614451106, 3628618433, 3642649144, 3656542712, 3670298613, 3683916329, 3697395348, 3710735162, 3723935269, 3736995171, 3749914379, 3762692404, 3775328765, 3787822988, 3800174601, 3812383140, 3824448145, 3836369162, 3848145741, 3859777440, 3871263820, 3882604450, 3893798902, 3904846754, 3915747591, 3926501002, 3937106583, 3947563934, 3957872662, 3968032378, 3978042699, 3987903250, 3997613658, 4007173558, 4016582591, 4025840401, 4034946641, 4043900968, 4052703044, 4061352537, 4069849124, 4078192482, 4086382299, 4094418266, 4102300081, 4110027446, 4117600071, 4125017671, 4132279966, 4139386683, 4146337555, 4153132319, 4159770720, 4166252509, 4172577440, 4178745276, 4184755784, 4190608739, 4196303920, 4201841112, 4207220108, 4212440704, 4217502704, 4222405917, 4227150159, 4231735252, 4236161021, 4240427302, 4244533933, 4248480760, 4252267634, 4255894413, 4259360959, 4262667143, 4265812840, 4268797931, 4271622305, 4274285855, 4276788480, 4279130086, 4281310585, 4283329896, 4285187942, 4286884652, 4288419964, 4289793820, 4291006167, 4292056960, 4292946160, 4293673732, 4294239650, 4294643893, 4294886444};
        static const uint32_t fftpow_256_tab2[] = {4294643893, 4293673732, 4292056960, 4289793820, 4286884652, 4283329896, 4279130086, 4274285855, 4268797931, 4262667143, 4255894413, 4248480760, 4240427302, 4231735252, 4222405917, 4212440704, 4201841112, 4190608739, 4178745276, 4166252509, 4153132319, 4139386683, 4125017671, 4110027446, 4094418266, 4078192482, 4061352537, 4043900968, 4025840401, 4007173558, 3987903250, 3968032378, 3947563934, 3926501002, 3904846754, 3882604450, 3859777440, 3836369162, 3812383140, 3787822988, 3762692404, 3736995171, 3710735162, 3683916329, 3656542712, 3628618433, 3600147697, 3571134792, 3541584088, 3511500034, 3480887161, 3449750080, 3418093478, 3385922125, 3353240863, 3320054617, 3286368382, 3252187232, 3217516315, 3182360851, 3146726136, 3110617535, 3074040487, 3037000500, 2999503152, 2961554089, 2923159027, 2884323748, 2845054101, 2805355999, 2765235421, 2724698408, 2683751066, 2642399561, 2600650120, 2558509031, 2515982640, 2473077351, 2429799626, 2386155981, 2342152991, 2297797281, 2253095531, 2208054473, 2162680890, 2116981616, 2070963532, 2024633568, 1977998702, 1931065957, 1883842400, 1836335144, 1788551342, 1740498191, 1692182927, 1643612827, 1594795204, 1545737412, 1496446837, 1446930903, 1397197066, 1347252816, 1297105676, 1246763195, 1196232957, 1145522571, 1094639673, 1043591926, 992387019, 941032661, 889536587, 837906553, 786150333, 734275721, 682290530, 630202589, 578019742, 525749847, 473400776, 420980412, 368496651, 315957395, 263370557, 210744057, 158085819, 105403774, 52705856};
        res = fftpow_calc(isig, 256, fftpow_256_tab1, fftpow_256_tab2);
        for(int i=0; i<=256; i++)
            res[i] = RSHIFT(res[i], 16);
        break;
#endif

#if MAX_FFT_SIG_LEN >= 1024
    case 1024:
        arm_cfft_q31(&arm_cfft_sR_q31_len512, isig, 0 /* ifftFlagR */, 1 /* bitReverseFlagR */);
        static const uint32_t fftpow_512_tab1[] = {3046303593, 3055578014, 3064823674, 3074040487, 3083228366, 3092387225, 3101516976, 3110617535, 3119688816, 3128730733, 3137743202, 3146726136, 3155679453, 3164603066, 3173496894, 3182360851, 3191194855, 3199998822, 3208772670, 3217516315, 3226229675, 3234912670, 3243565216, 3252187232, 3260778637, 3269339351, 3277869293, 3286368382, 3294836538, 3303273682, 3311679735, 3320054617, 3328398249, 3336710553, 3344991450, 3353240863, 3361458715, 3369644927, 3377799422, 3385922125, 3394012957, 3402071844, 3410098710, 3418093478, 3426056074, 3433986423, 3441884449, 3449750080, 3457583240, 3465383855, 3473151854, 3480887161, 3488589706, 3496259414, 3503896214, 3511500034, 3519070803, 3526608449, 3534112901, 3541584088, 3549021941, 3556426389, 3563797363, 3571134792, 3578438609, 3585708745, 3592945130, 3600147697, 3607316378, 3614451106, 3621551813, 3628618433, 3635650898, 3642649144, 3649613104, 3656542712, 3663437903, 3670298613, 3677124776, 3683916329, 3690673207, 3697395348, 3704082687, 3710735162, 3717352710, 3723935269, 3730482776, 3736995171, 3743472393, 3749914379, 3756321069, 3762692404, 3769028322, 3775328765, 3781593674, 3787822988, 3794016650, 3800174601, 3806296784, 3812383140, 3818433613, 3824448145, 3830426680, 3836369162, 3842275534, 3848145741, 3853979728, 3859777440, 3865538822, 3871263820, 3876952381, 3882604450, 3888219974, 3893798902, 3899341179, 3904846754, 3910315575, 3915747591, 3921142750, 3926501002, 3931822297, 3937106583, 3942353812, 3947563934, 3952736900, 3957872662, 3962971170, 3968032378, 3973056236, 3978042699, 3982991719, 3987903250, 3992777245, 3997613658, 4002412444, 4007173558, 4011896955, 4016582591, 4021230421, 4025840401, 4030412489, 4034946641, 4039442815, 4043900968, 4048321058, 4052703044, 4057046884, 4061352537, 4065619964, 4069849124, 4074039976, 4078192482, 4082306603, 4086382299, 4090419533, 4094418266, 4098378461, 4102300081, 4106183088, 4110027446, 4113833119, 4117600071, 4121328267, 4125017671, 4128668249, 4132279966, 4135852789, 4139386683, 4142881616, 4146337555, 4149754467, 4153132319, 4156471081, 4159770720, 4163031206, 4166252509, 4169434596, 4172577440, 4175681009, 4178745276, 4181770210, 4184755784, 4187701970, 4190608739, 4193476065, 4196303920, 4199092278, 4201841112, 4204550397, 4207220108, 4209850218, 4212440704, 4214991540, 4217502704, 4219974170, 4222405917, 4224797921, 4227150159, 4229462610, 4231735252, 4233968062, 4236161021, 4238314108, 4240427302, 4242500584, 4244533933, 4246527332, 4248480760, 4250394200, 4252267634, 4254101044, 4255894413, 4257647723, 4259360959, 4261034104, 4262667143, 4264260060, 4265812840, 4267325469, 4268797931, 4270230215, 4271622305, 4272974189, 4274285855, 4275557289, 4276788480, 4277979416, 4279130086, 4280240479, 4281310585, 4282340394, 4283329896, 4284279082, 4285187942, 4286056468, 4286884652, 4287672487, 4288419964, 4289127078, 4289793820, 4290420185, 4291006167, 4291551760, 4292056960, 4292521761, 4292946160, 4293330151, 4293673732, 4293976900, 4294239650, 4294461982, 4294643893, 4294785381, 4294886444, 4294947083};
        static const uint32_t fftpow_512_tab2[] = {4294886444, 4294643893, 4294239650, 4293673732, 4292946160, 4292056960, 4291006167, 4289793820, 4288419964, 4286884652, 4285187942, 4283329896, 4281310585, 4279130086, 4276788480, 4274285855, 4271622305, 4268797931, 4265812840, 4262667143, 4259360959, 4255894413, 4252267634, 4248480760, 4244533933, 4240427302, 4236161021, 4231735252, 4227150159, 4222405917, 4217502704, 4212440704, 4207220108, 4201841112, 4196303920, 4190608739, 4184755784, 4178745276, 4172577440, 4166252509, 4159770720, 4153132319, 4146337555, 4139386683, 4132279966, 4125017671, 4117600071, 4110027446, 4102300081, 4094418266, 4086382299, 4078192482, 4069849124, 4061352537, 4052703044, 4043900968, 4034946641, 4025840401, 4016582591, 4007173558, 3997613658, 3987903250, 3978042699, 3968032378, 3957872662, 3947563934, 3937106583, 3926501002, 3915747591, 3904846754, 3893798902, 3882604450, 3871263820, 3859777440, 3848145741, 3836369162, 3824448145, 3812383140, 3800174601, 3787822988, 3775328765, 3762692404, 3749914379, 3736995171, 3723935269, 3710735162, 3697395348, 3683916329, 3670298613, 3656542712, 3642649144, 3628618433, 3614451106, 3600147697, 3585708745, 3571134792, 3556426389, 3541584088, 3526608449, 3511500034, 3496259414, 3480887161, 3465383855, 3449750080, 3433986423, 3418093478, 3402071844, 3385922125, 3369644927, 3353240863, 3336710553, 3320054617, 3303273682, 3286368382, 3269339351, 3252187232, 3234912670, 3217516315, 3199998822, 3182360851, 3164603066, 3146726136, 3128730733, 3110617535, 3092387225, 3074040487, 3055578014, 3037000500, 3018308645, 2999503152, 2980584729, 2961554089, 2942411948, 2923159027, 2903796051, 2884323748, 2864742853, 2845054101, 2825258235, 2805355999, 2785348143, 2765235421, 2745018589, 2724698408, 2704275644, 2683751066, 2663125446, 2642399561, 2621574191, 2600650120, 2579628136, 2558509031, 2537293599, 2515982640, 2494576955, 2473077351, 2451484637, 2429799626, 2408023134, 2386155981, 2364198992, 2342152991, 2320018810, 2297797281, 2275489241, 2253095531, 2230616993, 2208054473, 2185408821, 2162680890, 2139871536, 2116981616, 2094011993, 2070963532, 2047837100, 2024633568, 2001353810, 1977998702, 1954569124, 1931065957, 1907490086, 1883842400, 1860123788, 1836335144, 1812477362, 1788551342, 1764557983, 1740498191, 1716372869, 1692182927, 1667929275, 1643612827, 1619234497, 1594795204, 1570295869, 1545737412, 1521120759, 1496446837, 1471716574, 1446930903, 1422090755, 1397197066, 1372250773, 1347252816, 1322204136, 1297105676, 1271958380, 1246763195, 1221521071, 1196232957, 1170899806, 1145522571, 1120102207, 1094639673, 1069135926, 1043591926, 1018008636, 992387019, 966728038, 941032661, 915301854, 889536587, 863737830, 837906553, 812043729, 786150333, 760227338, 734275721, 708296459, 682290530, 656258914, 630202589, 604122538, 578019742, 551895183, 525749847, 499584716, 473400776, 447199012, 420980412, 394745962, 368496651, 342233465, 315957395, 289669429, 263370557, 237061769, 210744057, 184418409, 158085819, 131747276, 105403774, 79056303, 52705856, 26353424};
        res = fftpow_calc(isig, 512, fftpow_512_tab1, fftpow_512_tab2);
        for(int i=0; i<=512; i++)
            res[i] = RSHIFT(res[i], 14);
        break;
#endif

#if MAX_FFT_SIG_LEN >= 2048
    case 2048:
        arm_cfft_q31(&arm_cfft_sR_q31_len1024, isig, 0 /* ifftFlagR */, 1 /* bitReverseFlagR */);
        static const uint32_t fftpow_1024_tab1[] = {3041655625, 3046303593, 3050944393, 3055578014, 3060204445, 3064823674, 3069435692, 3074040487, 3078638049, 3083228366, 3087811428, 3092387225, 3096955744, 3101516976, 3106070910, 3110617535, 3115156841, 3119688816, 3124213451, 3128730733, 3133240654, 3137743202, 3142238366, 3146726136, 3151206502, 3155679453, 3160144978, 3164603066, 3169053709, 3173496894, 3177932612, 3182360851, 3186781603, 3191194855, 3195600598, 3199998822, 3204389516, 3208772670, 3213148273, 3217516315, 3221876786, 3226229675, 3230574973, 3234912670, 3239242754, 3243565216, 3247880045, 3252187232, 3256486766, 3260778637, 3265062836, 3269339351, 3273608174, 3277869293, 3282122699, 3286368382, 3290606332, 3294836538, 3299058992, 3303273682, 3307480600, 3311679735, 3315871077, 3320054617, 3324230344, 3328398249, 3332558322, 3336710553, 3340854932, 3344991450, 3349120097, 3353240863, 3357353739, 3361458715, 3365555780, 3369644927, 3373726144, 3377799422, 3381864752, 3385922125, 3389971529, 3394012957, 3398046399, 3402071844, 3406089285, 3410098710, 3414100111, 3418093478, 3422078802, 3426056074, 3430025284, 3433986423, 3437939481, 3441884449, 3445821319, 3449750080, 3453670723, 3457583240, 3461487620, 3465383855, 3469271936, 3473151854, 3477023598, 3480887161, 3484742533, 3488589706, 3492428669, 3496259414, 3500081932, 3503896214, 3507702251, 3511500034, 3515289554, 3519070803, 3522843770, 3526608449, 3530364828, 3534112901, 3537852657, 3541584088, 3545307186, 3549021941, 3552728345, 3556426389, 3560116064, 3563797363, 3567470275, 3571134792, 3574790907, 3578438609, 3582077892, 3585708745, 3589331160, 3592945130, 3596550645, 3600147697, 3603736278, 3607316378, 3610887990, 3614451106, 3618005716, 3621551813, 3625089388, 3628618433, 3632138939, 3635650898, 3639154303, 3642649144, 3646135414, 3649613104, 3653082206, 3656542712, 3659994613, 3663437903, 3666872572, 3670298613, 3673716017, 3677124776, 3680524883, 3683916329, 3687299106, 3690673207, 3694038624, 3697395348, 3700743371, 3704082687, 3707413286, 3710735162, 3714048305, 3717352710, 3720648367, 3723935269, 3727213408, 3730482776, 3733743367, 3736995171, 3740238183, 3743472393, 3746697794, 3749914379, 3753122140, 3756321069, 3759511160, 3762692404, 3765864794, 3769028322, 3772182982, 3775328765, 3778465665, 3781593674, 3784712784, 3787822988, 3790924279, 3794016650, 3797100093, 3800174601, 3803240167, 3806296784, 3809344444, 3812383140, 3815412866, 3818433613, 3821445375, 3824448145, 3827441916, 3830426680, 3833402431, 3836369162, 3839326865, 3842275534, 3845215161, 3848145741, 3851067265, 3853979728, 3856883122, 3859777440, 3862662676, 3865538822, 3868405873, 3871263820, 3874112659, 3876952381, 3879782980, 3882604450, 3885416784, 3888219974, 3891014016, 3893798902, 3896574625, 3899341179, 3902098557, 3904846754, 3907585762, 3910315575, 3913036187, 3915747591, 3918449781, 3921142750, 3923826493, 3926501002, 3929166272, 3931822297, 3934469069, 3937106583, 3939734833, 3942353812, 3944963515, 3947563934, 3950155065, 3952736900, 3955309435, 3957872662, 3960426576, 3962971170, 3965506439, 3968032378, 3970548979, 3973056236, 3975554145, 3978042699, 3980521892, 3982991719, 3985452174, 3987903250, 3990344942, 3992777245, 3995200152, 3997613658, 4000017757, 4002412444, 4004797713, 4007173558, 4009539974, 4011896955, 4014244496, 4016582591, 4018911234, 4021230421, 4023540145, 4025840401, 4028131185, 4030412489, 4032684310, 4034946641, 4037199478, 4039442815, 4041676647, 4043900968, 4046115773, 4048321058, 4050516816, 4052703044, 4054879734, 4057046884, 4059204486, 4061352537, 4063491032, 4065619964, 4067739330, 4069849124, 4071949341, 4074039976, 4076121025, 4078192482, 4080254343, 4082306603, 4084349257, 4086382299, 4088405726, 4090419533, 4092423715, 4094418266, 4096403184, 4098378461, 4100344095, 4102300081, 4104246413, 4106183088, 4108110101, 4110027446, 4111935121, 4113833119, 4115721438, 4117600071, 4119469016, 4121328267, 4123177820, 4125017671, 4126847815, 4128668249, 4130478967, 4132279966, 4134071241, 4135852789, 4137624604, 4139386683, 4141139022, 4142881616, 4144614462, 4146337555, 4148050891, 4149754467, 4151448277, 4153132319, 4154806588, 4156471081, 4158125793, 4159770720, 4161405860, 4163031206, 4164646757, 4166252509, 4167848456, 4169434596, 4171010925, 4172577440, 4174134136, 4175681009, 4177218057, 4178745276, 4180262661, 4181770210, 4183267919, 4184755784, 4186233802, 4187701970, 4189160283, 4190608739, 4192047334, 4193476065, 4194894928, 4196303920, 4197703038, 4199092278, 4200471637, 4201841112, 4203200700, 4204550397, 4205890201, 4207220108, 4208540114, 4209850218, 4211150416, 4212440704, 4213721080, 4214991540, 4216252083, 4217502704, 4218743401, 4219974170, 4221195010, 4222405917, 4223606888, 4224797921, 4225979012, 4227150159, 4228311359, 4229462610, 4230603908, 4231735252, 4232856637, 4233968062, 4235069525, 4236161021, 4237242550, 4238314108, 4239375693, 4240427302, 4241468933, 4242500584, 4243522251, 4244533933, 4245535628, 4246527332, 4247509043, 4248480760, 4249442480, 4250394200, 4251335919, 4252267634, 4253189343, 4254101044, 4255002735, 4255894413, 4256776076, 4257647723, 4258509352, 4259360959, 4260202544, 4261034104, 4261855638, 4262667143, 4263468618, 4264260060, 4265041468, 4265812840, 4266574174, 4267325469, 4268066722, 4268797931, 4269519096, 4270230215, 4270931285, 4271622305, 4272303274, 4272974189, 4273635050, 4274285855, 4274926601, 4275557289, 4276177915, 4276788480, 4277388980, 4277979416, 4278559785, 4279130086, 4279690318, 4280240479, 4280780569, 4281310585, 4281830528, 4282340394, 4282840184, 4283329896, 4283809529, 4284279082, 4284738553, 4285187942, 4285627247, 4286056468, 4286475604, 4286884652, 4287283614, 4287672487, 4288051271, 4288419964, 4288778567, 4289127078, 4289465495, 4289793820, 4290112050, 4290420185, 4290718224, 4291006167, 4291284012, 4291551760, 4291809410, 4292056960, 4292294411, 4292521761, 4292739011, 4292946160, 4293143206, 4293330151, 4293506993, 4293673732, 4293830368, 4293976900, 4294113327, 4294239650, 4294355869, 4294461982, 4294557990, 4294643893, 4294719690, 4294785381, 4294840966, 4294886444, 4294921817, 4294947083, 4294962243};
        static const uint32_t fftpow_1024_tab2[] = {4294947083, 4294886444, 4294785381, 4294643893, 4294461982, 4294239650, 4293976900, 4293673732, 4293330151, 4292946160, 4292521761, 4292056960, 4291551760, 4291006167, 4290420185, 4289793820, 4289127078, 4288419964, 4287672487, 4286884652, 4286056468, 4285187942, 4284279082, 4283329896, 4282340394, 4281310585, 4280240479, 4279130086, 4277979416, 4276788480, 4275557289, 4274285855, 4272974189, 4271622305, 4270230215, 4268797931, 4267325469, 4265812840, 4264260060, 4262667143, 4261034104, 4259360959, 4257647723, 4255894413, 4254101044, 4252267634, 4250394200, 4248480760, 4246527332, 4244533933, 4242500584, 4240427302, 4238314108, 4236161021, 4233968062, 4231735252, 4229462610, 4227150159, 4224797921, 4222405917, 4219974170, 4217502704, 4214991540, 4212440704, 4209850218, 4207220108, 4204550397, 4201841112, 4199092278, 4196303920, 4193476065, 4190608739, 4187701970, 4184755784, 4181770210, 4178745276, 4175681009, 4172577440, 4169434596, 4166252509, 4163031206, 4159770720, 4156471081, 4153132319, 4149754467, 4146337555, 4142881616, 4139386683, 4135852789, 4132279966, 4128668249, 4125017671, 4121328267, 4117600071, 4113833119, 4110027446, 4106183088, 4102300081, 4098378461, 4094418266, 4090419533, 4086382299, 4082306603, 4078192482, 4074039976, 4069849124, 4065619964, 4061352537, 4057046884, 4052703044, 4048321058, 4043900968, 4039442815, 4034946641, 4030412489, 4025840401, 4021230421, 4016582591, 4011896955, 4007173558, 4002412444, 3997613658, 3992777245, 3987903250, 3982991719, 3978042699, 3973056236, 3968032378, 3962971170, 3957872662, 3952736900, 3947563934, 3942353812, 3937106583, 3931822297, 3926501002, 3921142750, 3915747591, 3910315575, 3904846754, 3899341179, 3893798902, 3888219974, 3882604450, 3876952381, 3871263820, 3865538822, 3859777440, 3853979728, 3848145741, 3842275534, 3836369162, 3830426680, 3824448145, 3818433613, 3812383140, 3806296784, 3800174601, 3794016650, 3787822988, 3781593674, 3775328765, 3769028322, 3762692404, 3756321069, 3749914379, 3743472393, 3736995171, 3730482776, 3723935269, 3717352710, 3710735162, 3704082687, 3697395348, 3690673207, 3683916329, 3677124776, 3670298613, 3663437903, 3656542712, 3649613104, 3642649144, 3635650898, 3628618433, 3621551813, 3614451106, 3607316378, 3600147697, 3592945130, 3585708745, 3578438609, 3571134792, 3563797363, 3556426389, 3549021941, 3541584088, 3534112901, 3526608449, 3519070803, 3511500034, 3503896214, 3496259414, 3488589706, 3480887161, 3473151854, 3465383855, 3457583240, 3449750080, 3441884449, 3433986423, 3426056074, 3418093478, 3410098710, 3402071844, 3394012957, 3385922125, 3377799422, 3369644927, 3361458715, 3353240863, 3344991450, 3336710553, 3328398249, 3320054617, 3311679735, 3303273682, 3294836538, 3286368382, 3277869293, 3269339351, 3260778637, 3252187232, 3243565216, 3234912670, 3226229675, 3217516315, 3208772670, 3199998822, 3191194855, 3182360851, 3173496894, 3164603066, 3155679453, 3146726136, 3137743202, 3128730733, 3119688816, 3110617535, 3101516976, 3092387225, 3083228366, 3074040487, 3064823674, 3055578014, 3046303593, 3037000500, 3027668821, 3018308645, 3008920059, 2999503152, 2990058012, 2980584729, 2971083391, 2961554089, 2951996911, 2942411948, 2932799290, 2923159027, 2913491250, 2903796051, 2894073520, 2884323748, 2874546829, 2864742853, 2854911913, 2845054101, 2835169511, 2825258235, 2815320366, 2805355999, 2795365227, 2785348143, 2775304843, 2765235421, 2755139971, 2745018589, 2734871369, 2724698408, 2714499801, 2704275644, 2694026034, 2683751066, 2673450838, 2663125446, 2652774988, 2642399561, 2631999263, 2621574191, 2611124444, 2600650120, 2590151318, 2579628136, 2569080674, 2558509031, 2547913306, 2537293599, 2526650010, 2515982640, 2505291588, 2494576955, 2483838842, 2473077351, 2462292582, 2451484637, 2440653617, 2429799626, 2418922764, 2408023134, 2397100839, 2386155981, 2375188665, 2364198992, 2353187066, 2342152991, 2331096871, 2320018810, 2308918911, 2297797281, 2286654023, 2275489241, 2264303042, 2253095531, 2241866812, 2230616993, 2219346178, 2208054473, 2196741986, 2185408821, 2174055087, 2162680890, 2151286337, 2139871536, 2128436593, 2116981616, 2105506713, 2094011993, 2082497563, 2070963532, 2059410008, 2047837100, 2036244917, 2024633568, 2013003163, 2001353810, 1989685620, 1977998702, 1966293167, 1954569124, 1942826684, 1931065957, 1919287054, 1907490086, 1895675165, 1883842400, 1871991904, 1860123788, 1848238164, 1836335144, 1824414839, 1812477362, 1800522825, 1788551342, 1776563023, 1764557983, 1752536335, 1740498191, 1728443664, 1716372869, 1704285919, 1692182927, 1680064008, 1667929275, 1655778843, 1643612827, 1631431340, 1619234497, 1607022414, 1594795204, 1582552984, 1570295869, 1558023973, 1545737412, 1533436302, 1521120759, 1508790899, 1496446837, 1484088690, 1471716574, 1459330606, 1446930903, 1434517580, 1422090755, 1409650544, 1397197066, 1384730436, 1372250773, 1359758194, 1347252816, 1334734758, 1322204136, 1309661069, 1297105676, 1284538073, 1271958380, 1259366714, 1246763195, 1234147941, 1221521071, 1208882703, 1196232957, 1183571952, 1170899806, 1158216639, 1145522571, 1132817720, 1120102207, 1107376152, 1094639673, 1081892891, 1069135926, 1056368897, 1043591926, 1030805132, 1018008636, 1005202558, 992387019, 979562138, 966728038, 953884839, 941032661, 928171626, 915301854, 902423468, 889536587, 876641334, 863737830, 850826195, 837906553, 824979024, 812043729, 799100792, 786150333, 773192474, 760227338, 747255046, 734275721, 721289485, 708296459, 695296767, 682290530, 669277872, 656258914, 643233779, 630202589, 617165468, 604122538, 591073921, 578019742, 564960121, 551895183, 538825051, 525749847, 512669694, 499584716, 486495035, 473400776, 460302060, 447199012, 434091755, 420980412, 407865107, 394745962, 381623102, 368496651, 355366730, 342233465, 329096979, 315957395, 302814837, 289669429, 276521294, 263370557, 250217341, 237061769, 223903967, 210744057, 197582163, 184418409, 171252920, 158085819, 144917230, 131747276, 118576083, 105403774, 92230472, 79056303, 65881389, 52705856, 39529826, 26353424, 13176774};
        res = fftpow_calc(isig, 1024, fftpow_1024_tab1, fftpow_1024_tab2);
        for(int i=0; i<=1024; i++)
            res[i] = RSHIFT(res[i], 12);
        break;
#endif

    default:
        /* You screwed up and provided an unsupported length! */
        while(1)
            ;
    }
    return res;
}

static void fft_csum_inplace(pwr_t *pwr, int fft_len) {
    /* This won't overflow because of the RSHIFTs in calculate_fft.
       Even if it did, it would be OK as long as the bins don't cover too many samples. */
    pwr_t sum = 0;
    for(int i=0; i<fft_len; i++) {
        sum += pwr[i];
        pwr[i] = sum;
    }
}

static uint32_t roundup_pow2(uint32_t v) {
    /* https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2 */
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}

/* This function converts a floating-point value into a value suitable for
   a fixed-point feature, using a floating-point transform (a partially
   logarithmic transformation).
   This transformation is monotonic for finite values. */
static int32_t float_feat(float f) {
    /* NOTE: Due to the very wide dynamic range of some feature values, a linear transformation
       (e.g. RSHIFT(v, constant)) was deemed insufficiently accurate over the entire range.
       Instead, we encode the value as a floating-point number, and interpret the bits of
       the float as an integer. The resulting transformation is monotonic (preserving order)
       for finite values (as long as negative integers are flipped around). It partially
       approximates the log2 function, preserving high precision across the full range,
       at the expense of being somewhat more complex.
       Decision classifiers such as XGB and GBM should not have
       problems with this transformation, but other classifiers such as SVM or NN might
       behave suboptimally as this is not a fully linear nor fully logarithmic transform.
       In such cases, you may prefer to simply perform a full logf transform followed by a
       linear scaling instead (which is slower, but numerically more sensible).
    */
    int32_t res;
    memcpy(&res, &f, 4);
    /* Negative floats are represented only with a sign bit, which makes them compare
       incorrectly; to resolve this, flip the negative values around. */
    if(res < 0)
        res = (-2147483648) - res;
    return res;
}

/* This function converts a 64-bit power (or sum of power values) into a value suitable for
   a fixed-point feature. */
static int32_t fftpow_feat(pwr_t v) {
    /* NOTE: Because we do not scale the fftpow coefficients by samp_freq, these values
       will similarly be scaled up by samp_freq compared to the floating-point pipeline.
       Take this into account if comparing the pipelines! */
    return float_feat(i64_to_f32(v, 0));
}

static void set_feats_fft_stat_simple(Features *F_ptr, const pwr_t *pwr, int fft_len, float samp_freq) {
    pwr_t mina = pwr[0]; /* minimum */
    pwr_t maxa = pwr[0]; /* maximum */
    pwr_t m1i = 0; /* first moment */

    /* calculate max, min, and moments */
    int imax = 0;
    for (int i = 0; i < fft_len; i++) {
        pwr_t cur = pwr[i];
        if (cur < mina) {
            mina = cur;
        } else if (cur > maxa) {
            maxa = cur;
            imax = i;
        }
        /* does not overflow due to RSHIFT in calculate_fft */
        m1i += cur;
    }
    SET_FEAT(F_ptr, g_feature_prefix, "fftpowsimple_stat", "freq_of_max", float_feat((imax+1)*samp_freq/(2 * fft_len)));

    float m1 = i64_to_f32(m1i, 0);
    SET_FEAT(F_ptr, g_feature_prefix, "fftpowsimple_stat", "sum_sqrt", float_feat(sqrtf(m1)));
    float inv_len = 1.0f / fft_len;
    m1 *= inv_len;
    SET_FEAT(F_ptr, g_feature_prefix, "fftpowsimple_stat", "mean", float_feat(m1));

    pwr_t rng = maxa - mina;
    if (rng == 0) rng = 1;
    SET_FEAT(F_ptr, g_feature_prefix, "fftpowsimple_stat", "range", fftpow_feat(rng));
    SET_FEAT(F_ptr, g_feature_prefix, "fftpowsimple_stat", "mean_by_range", float_feat(m1 / rng));
}

static const int max_num_fft_bins = MIN(1 / FFT_BINS_RATIO, MAX_NUM_POW_BINS);
static void set_feats_fft_stat_advanced(Features *F_ptr, const pwr_t *pwr, int fft_len) {
    int n = fft_len;
    int num_bins = MIN(max_num_fft_bins, n / 2); /* to group at least two fft cofficients */

    float scaler = 1.0f / (2*n);
    int binidx = 0;

    /* If you're looking for a fixed-point flatness feature, you've come to the right place. This
     * feature was removed from the fixed point pipeline because it was very slow and offered little
     * evidence that it boosted accuracy to be worth the latency cost.
     *
     * Our featurization test code explicitly ignores the spectral flatness feature for M0 devices.
     *
     * See EM-1236 and https://github.com/qeexo/embedded_ml/pull/1180 for more details. */

    pwr_t ipwr_sum = 0, icen = 0;
    for(int i=0; i<n; i++) {
        ipwr_sum += pwr[i];
        icen += pwr[i] * (i+1);
    }

    float pwr_sum = i64_to_f32(ipwr_sum, 0);
    float cen = i64_to_f32(icen, 0) * scaler;
    if (pwr_sum == 0) {
        cen = 0;
    } else {
        cen /= pwr_sum;
    }
    SET_FEAT(F_ptr, g_feature_prefix, "fftpowadvanced_stat", "centroid", float_feat(cen));

    float pwr_avg = pwr_sum / (float)n;
    float frq_avg = scaler*(1 + n) / 2.0;

    pwr_t pwr_cumsum = 0; /* cumulative power sum */
    int irol = -1; /* index of rolloff */
    pwr_t pwr_sum_95_perc = (pwr_t)(0.95f * pwr_sum);
    for (int i = 0; i < n; i++) {
        /* rolloff. freq where cumsum exceeds 95% total sum */
        if (irol == -1) {
            pwr_cumsum += pwr[i];
            if (pwr_cumsum > pwr_sum_95_perc) {
                irol = i;
            }
        }
    }

    /* In case signal is 0, irol was not updated above */
    if (irol == -1) {
        irol = 0;
    }
    float roll = scaler*(irol+1);     /* rolloff */
    SET_FEAT(F_ptr, g_feature_prefix, "fftpowadvanced_stat", "rolloff", float_feat(roll));

    /* frq_var = sum[((i+1) - (n + 1)/2)^2], scaled */
    float frq_var = ((n+1) * (n-1)) / ((float)(n * 48));
    /* frq_pwr_cov = sum((i+1) * pwr[i]), scaled */
    float frq_pwr_cov = cen * pwr_sum;
    frq_pwr_cov -= (float)n * frq_avg * pwr_avg;
    float slope = frq_pwr_cov / frq_var; /* slope of FFT. slope in simple linear regression is cov(x,y)/var(x) */
    /* NOTE: should do "frq_var /= (float)n;" and "frq_pwr_cov /= (float)n" but comp faster to not since cancels */
    SET_FEAT(F_ptr, g_feature_prefix, "fftpowadvanced_stat", "slope", float_feat(slope));
}

static void set_feats_fft_binned(Features *F_ptr, const pwr_t *pwr_csum, int fft_len) {
#ifdef PYTHON_FEATS
    char bin_num[10], bin_num2[10]; /* space used for naming bins */
#endif

    int num_bins = MIN(max_num_fft_bins, fft_len / 2); /* to group at least two fft cofficients */

    /* bin += fft_len / num_bins at each step,
       but in order to avoid expensive divisions in the loop, we track
       the integral and fractional parts of the fraction separately */
    int bin_q = 0;
    int bin_r = 0;
    int incr_q = fft_len / num_bins;
    int incr_r = fft_len % num_bins;

#ifdef PYTHON_FEATS
    int num_digits = get_num_digits(num_bins);
#endif
    for (int i = 0; i < num_bins; i++) {
        pwr_t left = pwr_csum[bin_q];
        bin_q += incr_q;
        bin_r += incr_r;
        if(bin_r >= num_bins) {
            bin_q++;
            bin_r -= num_bins;
        }
        pwr_t right = pwr_csum[bin_q];

#ifdef PYTHON_FEATS
        set_feat_num(bin_num, i+1, num_digits);
#endif
        SET_FEAT(F_ptr, g_feature_prefix, "fftpowlinearlybinned_bin", bin_num, fftpow_feat(right - left));
    }
}

static void set_feats_fft_octave(Features *F_ptr, const pwr_t *pwr_csum, int fft_len, float samp_freq) {
    /* compute the number of octave bands */
    int num_bins = min(log2_uint32(fft_len), MAX_NUM_OCTAVE_BANDS);
    /* compute the number of bins for the first octave band (B0) */
    int first_band_num_bins = fft_len >> (num_bins - 1);
    first_band_num_bins = max(first_band_num_bins, MIN_BINS_PER_OCTAVE_BAND);

    /* stack allocation - bounded to 64 bytes (and in practice much less) */
    uint16_t oct_bins[num_bins + 2];  /* +2 to be able to include endpoints if needed */
    int counter = 0;
    int idx = first_band_num_bins;
    /* compute the upper bin for each octave band using Bi = (2 ** i) * B0 */
    for (int i = 0; i < num_bins + 1; i++) {  /* num_bins + 1 to include the first bin with idx 0*/
        if (i == 0){
             oct_bins[counter] = 0;
             counter += 1;
        } else {
            if (idx > fft_len){
                break;  /* avoid index out of bounds */
            }
            if (idx > oct_bins[counter-1]){
                oct_bins[counter] = idx;
                counter += 1;
            }
            idx *= 2;
        }
    }

    /* include last bin */
    if (oct_bins[counter - 1] < fft_len){
        oct_bins[counter] = fft_len;
        counter += 1;
    }

    num_bins = counter - 1;

#ifdef PYTHON_FEATS
    char bin_num[10], bin_num2[10];
    int num_digits = get_num_digits(num_bins);
#endif
    /* stack allocation - bounded by 128 bytes - in practice, much less */
    float log_oct[num_bins];
    for(int i=0; i<num_bins; i++) {
        pwr_t binsum = pwr_csum[oct_bins[i+1]] - pwr_csum[oct_bins[i]];
        log_oct[i] = logf(binsum);
        /* Removing FFT OOCTAVE BIN features as they are subset of Thirds features. More analysis can be found in EM-1049. */


#ifdef PYTHON_FEATS
        set_feat_num(bin_num, i+1, num_digits);
#endif
        /*
        See scaling comment in set_feats_fft_binned
        SET_FEAT(F_ptr, g_feature_prefix, "fftpowoctavebinned_bin", bin_num, fftpow_feat(binsum));
        */
    }

    /* log ratio of octaves */
#ifdef PYTHON_FEATS
    char buf[10+10+6]; /* str buffer space to hold "{bin_num[i]}_over_{bin_num[j]}" */
#endif
    for (int i = 0; i < num_bins; i++) {
        for (int j = 0; j < i; j++) {
#ifdef PYTHON_FEATS
            set_feat_num(bin_num, i+1, num_digits);
            set_feat_num(bin_num2, j+1, num_digits);
            sprintf(buf, "%s_over_%s", bin_num, bin_num2);
#endif
            SET_FEAT(F_ptr, g_feature_prefix, "fftpowoctavebinned_logratio", buf, f32_to_i32(log_oct[i] - log_oct[j], 24));
        }
    }
}

static void set_feats_fft_thirds(Features *F_ptr, const pwr_t *pwr_csum, int fft_len, float samp_freq) {
    /* compute log2(samp_freq / 2) */
    /* compute the number of octave bands */
    int num_oct = min(log2_uint32(fft_len), MAX_NUM_OCTAVE_BANDS);
    /* compute the number of bins for the first octave band (B0) */
    int first_band_num_bins = fft_len >> (num_oct - 1);
    first_band_num_bins = max(first_band_num_bins, MIN_BINS_PER_OCTAVE_BAND);
    int num_bins = 3 * num_oct + 2; /* to include endpoints */

#ifdef PYTHON_FEATS
    char bin_num[10];
    int num_digits = get_num_digits(num_bins);
    int binidx = 0;
#endif
    float idx_unrounded = i32_to_f32(first_band_num_bins, 0);
    int leftidx = 0;
    const float cube_root_two = 1.25992104f; // 2.0**(1.0/3.0)
    /* compute the upper bin for each thirds band using Bi = (2 ** (i/3)) * B0 */
    for(int i=0; i<num_bins; i++) {
        if (i > 0){
            idx_unrounded *= cube_root_two;
        }
        int rightidx = f32_to_i32(idx_unrounded, 0);
        if (rightidx > fft_len){ /* avoid index out of bounds */
            if (leftidx != fft_len){
                rightidx = fft_len; /* include last bin */
            } else {
                break;
            }
        }
        if(rightidx > leftidx) {
            pwr_t binsum = pwr_csum[rightidx] - pwr_csum[leftidx];
#ifdef PYTHON_FEATS
            set_feat_num(bin_num, binidx+1, num_digits);
            binidx++;
#endif
            SET_FEAT(F_ptr, g_feature_prefix, "fftpowthirdsbinned_bin", bin_num, fftpow_feat(binsum));
        }
        leftidx = rightidx;
    }
}



static void calculate_adaptive_binned_features(Features* F_ptr, const pwr_t *pwr_csum, int fft_len, int adaptive_binning_sensor_channel_key){
#ifdef PYTHON_FEATS
    char bin_num[10];
#endif
    int *binning_strategy;
    int adaptive_bins_len;

#ifdef ACCEL_X_ADAPTIVE_BINNING_LEN
    if (adaptive_binning_sensor_channel_key == 1){
        binning_strategy = ACCEL_X_ADAPTIVE_BINNING_STRATEGY;
        adaptive_bins_len = ACCEL_X_ADAPTIVE_BINNING_LEN;
    }
#endif
#ifdef ACCEL_Y_ADAPTIVE_BINNING_LEN
    if (adaptive_binning_sensor_channel_key == 2){
        binning_strategy = ACCEL_Y_ADAPTIVE_BINNING_STRATEGY;
        adaptive_bins_len = ACCEL_Y_ADAPTIVE_BINNING_LEN;
    }
#endif
#ifdef ACCEL_Z_ADAPTIVE_BINNING_LEN
    if (adaptive_binning_sensor_channel_key == 3){
        binning_strategy = ACCEL_Z_ADAPTIVE_BINNING_STRATEGY;
        adaptive_bins_len = ACCEL_Z_ADAPTIVE_BINNING_LEN;
    }
#endif
#ifdef ACCEL_W_ADAPTIVE_BINNING_LEN
    if (adaptive_binning_sensor_channel_key == 4){
        binning_strategy = ACCEL_W_ADAPTIVE_BINNING_STRATEGY;
        adaptive_bins_len = ACCEL_W_ADAPTIVE_BINNING_LEN;
    }
#endif
#ifdef GYRO_X_ADAPTIVE_BINNING_LEN
    if (adaptive_binning_sensor_channel_key == 5){
        binning_strategy = GYRO_X_ADAPTIVE_BINNING_STRATEGY;
        adaptive_bins_len = GYRO_X_ADAPTIVE_BINNING_LEN;
    }
#endif
#ifdef GYRO_Y_ADAPTIVE_BINNING_LEN
    if (adaptive_binning_sensor_channel_key == 6){
        binning_strategy = GYRO_Y_ADAPTIVE_BINNING_STRATEGY;
        adaptive_bins_len = GYRO_Y_ADAPTIVE_BINNING_LEN;
    }
#endif
#ifdef GYRO_Z_ADAPTIVE_BINNING_LEN
    if (adaptive_binning_sensor_channel_key == 7){
        binning_strategy = GYRO_Z_ADAPTIVE_BINNING_STRATEGY;
        adaptive_bins_len = GYRO_Z_ADAPTIVE_BINNING_LEN;
    }
#endif
#ifdef GYRO_W_ADAPTIVE_BINNING_LEN
    if (adaptive_binning_sensor_channel_key == 8){
        binning_strategy = GYRO_W_ADAPTIVE_BINNING_STRATEGY;
        adaptive_bins_len = GYRO_W_ADAPTIVE_BINNING_LEN;
    }
#endif
#ifdef MICROPHONE_ADAPTIVE_BINNING_LEN
    if (adaptive_binning_sensor_channel_key == 9){
        binning_strategy = MICROPHONE_ADAPTIVE_BINNING_STRATEGY;
        adaptive_bins_len = MICROPHONE_ADAPTIVE_BINNING_LEN;
    }
#endif
#ifdef ACCEL_LOWPOWER_X_ADAPTIVE_BINNING_LEN
    if (adaptive_binning_sensor_channel_key == 10){
        binning_strategy = ACCEL_LOWPOWER_X_ADAPTIVE_BINNING_STRATEGY;
        adaptive_bins_len = ACCEL_LOWPOWER_X_ADAPTIVE_BINNING_LEN;
    }
#endif
#ifdef ACCEL_LOWPOWER_Y_ADAPTIVE_BINNING_LEN
    if (adaptive_binning_sensor_channel_key == 11){
        binning_strategy = ACCEL_LOWPOWER_Y_ADAPTIVE_BINNING_STRATEGY;
        adaptive_bins_len = ACCEL_LOWPOWER_Y_ADAPTIVE_BINNING_LEN;
    }
#endif
#ifdef ACCEL_LOWPOWER_Z_ADAPTIVE_BINNING_LEN
    if (adaptive_binning_sensor_channel_key == 12){
        binning_strategy = ACCEL_LOWPOWER_Z_ADAPTIVE_BINNING_STRATEGY;
        adaptive_bins_len = ACCEL_LOWPOWER_Z_ADAPTIVE_BINNING_LEN;
    }
#endif
#ifdef ACCEL_LOWPOWER_W_ADAPTIVE_BINNING_LEN
    if (adaptive_binning_sensor_channel_key == 13){
        binning_strategy = ACCEL_LOWPOWER_W_ADAPTIVE_BINNING_STRATEGY;
        adaptive_bins_len = ACCEL_LOWPOWER_W_ADAPTIVE_BINNING_LEN;
    }
#endif
#ifdef ACCEL_HIGHSENSITIVE_X_ADAPTIVE_BINNING_LEN
    if (adaptive_binning_sensor_channel_key == 14){
        binning_strategy = ACCEL_HIGHSENSITIVE_X_ADAPTIVE_BINNING_STRATEGY;
        adaptive_bins_len = ACCEL_HIGHSENSITIVE_X_ADAPTIVE_BINNING_LEN;
    }
#endif
#ifdef ACCEL_HIGHSENSITIVE_Y_ADAPTIVE_BINNING_LEN
    if (adaptive_binning_sensor_channel_key == 15){
        binning_strategy = ACCEL_HIGHSENSITIVE_Y_ADAPTIVE_BINNING_STRATEGY;
        adaptive_bins_len = ACCEL_HIGHSENSITIVE_Y_ADAPTIVE_BINNING_LEN;
    }
#endif
#ifdef ACCEL_HIGHSENSITIVE_Z_ADAPTIVE_BINNING_LEN
    if (adaptive_binning_sensor_channel_key == 16){
        binning_strategy = ACCEL_HIGHSENSITIVE_Z_ADAPTIVE_BINNING_STRATEGY;
        adaptive_bins_len = ACCEL_HIGHSENSITIVE_Z_ADAPTIVE_BINNING_LEN;
    }
#endif
#ifdef ACCEL_HIGHSENSITIVE_W_ADAPTIVE_BINNING_LEN
    if (adaptive_binning_sensor_channel_key == 17){
        binning_strategy = ACCEL_HIGHSENSITIVE_W_ADAPTIVE_BINNING_STRATEGY;
        adaptive_bins_len = ACCEL_HIGHSENSITIVE_W_ADAPTIVE_BINNING_LEN;
    }
#endif


    /* power spectrum summed into offlien adaptive binning. */
    pwr_t adaptive_pwr[adaptive_bins_len - 2];


/* power spectrum summed using adaptive binning strategy. */
#ifdef PYTHON_FEATS
    int num_digits = get_num_digits(adaptive_bins_len - 1);
#endif
    for (int i = 0; i < adaptive_bins_len - 1; i++) {
        adaptive_pwr[i] = pwr_csum[binning_strategy[i + 1]] - pwr_csum[binning_strategy[i]];
#ifdef PYTHON_FEATS
        set_feat_num(bin_num, i+1, num_digits);
#endif
        SET_FEAT(F_ptr, g_feature_prefix, "fftpowadaptivebinned_bin", bin_num, fftpow_feat(adaptive_pwr[i]));
    }
    float numerator;
    float denominator;
    float ratio;
#ifdef PYTHON_FEATS
    num_digits = get_num_digits(adaptive_bins_len - 2);
#endif

    for (int i = 0; i < adaptive_bins_len - 2; i++) {
#ifdef PYTHON_FEATS
        set_feat_num(bin_num, i+1, num_digits);
#endif
        numerator = adaptive_pwr[i];
        denominator = adaptive_pwr[i + 1];
        ratio =  numerator / (1 + denominator);
        SET_FEAT(F_ptr, g_feature_prefix, "fftpowadaptivebinned_bin_ratio", bin_num, f32_to_i32(ratio, 24));
    }
}




static void set_feats_fft(Features* F_ptr, sigval_t* accel, fval_t sig_mean, int n, float samp_freq, bool is_microphone, fft_feats calc_fft_sub_feats, mfcc_feats calc_mfcc_sub_feats, int adaptive_binning_sensor_channel_key) {
    int fft_sig_len = roundup_pow2(n);
    /* Copy signal to temporary buffer for FFT computation and pad with zero (mean) */

    /* Note the below differences in FFT computation for fixed point case.

        1. The signal is not mean subtracted for the case of fixed point.
        2. For signals with length less than fft_sig_len, the signal is mean padded instead of zero padded.
        3. No window is applied to the signal in fixed point case. We apply Bartlett Window otherwise. Look at the
        other set_feats_fft function for non fixed point case.

    */


    int32_t *isig = TMP_BUF; /* Q1.31 */
    for(int i=0; i<n; i++) {
        isig[i] = ((int32_t)accel[i]) << 16;

    }
    int ssig_mean = sig_mean << 1; /* Q1.31 */
    for(int i=n; i<fft_sig_len; i++) {
        isig[i] = ssig_mean;
    }

    /* Drop DC component (element 0) because we don't need it */
    pwr_t *pwr = calculate_fft(isig, fft_sig_len) + 1;
    /* Length of the power spectrum */
    int fft_len = fft_sig_len / 2;

    if (calc_fft_sub_feats.simple) {
        set_feats_fft_stat_simple(F_ptr, pwr, fft_len, samp_freq);
    }

    if (calc_fft_sub_feats.advanced) {
        set_feats_fft_stat_advanced(F_ptr, pwr, fft_len);
    }

    if (!calc_fft_sub_feats.binned && !calc_fft_sub_feats.octave && !calc_fft_sub_feats.thirds) {
        return;
    }

    /* Take the cumulative sum of the power spectrum for easy computation of bin features.
       This is done in place - therefore, all features from now on can only work with the csum! */
    fft_csum_inplace(pwr, fft_len);
    /* For easier csum calculation, insert a zero in front.
       This makes pwr_csum[i] = sum(pwr[:i]) for i in [0, n], which is easier to work with. */
    pwr_t *pwr_csum = pwr - 1;
    pwr_csum[0] = 0;

    if (calc_fft_sub_feats.binned) {
        set_feats_fft_binned(F_ptr, pwr_csum, fft_len);
    }

    if (calc_fft_sub_feats.octave) {
        set_feats_fft_octave(F_ptr, pwr_csum, fft_len, samp_freq);
    }

    if (calc_fft_sub_feats.thirds) {
        set_feats_fft_thirds(F_ptr, pwr_csum, fft_len, samp_freq);
    }
    if (calc_fft_sub_feats.adaptive) {
        calculate_adaptive_binned_features(F_ptr, pwr_csum, fft_len, adaptive_binning_sensor_channel_key);
    }
}
#endif

/* Set features for one axis. Call three times - once per axis. */
#if defined(ACCEL_FEATS) || defined(MICROPHONE_FEATS) || defined(GYRO_FEATS) || defined(MAGNO_FEATS)
void featurize(Features* F_ptr, sigval_t* accel, int n, float samp_freq, bool is_microphone, bool is_magno,
                    bool calc_stat_features, bool calc_autocorr_features, fft_feats calc_fft_sub_feats, bool calc_peak_features, mfcc_feats calc_mfcc_sub_feats, int adaptive_binning_sensor_channel_key) {

    /* accel is Q0.15 - so it's safe to just sum it up unless n >= 65536 */
    int32_t sig_sum = 0; /* Q16.15 */
    for(int i=0; i<n; i++) {
        sig_sum += accel[i];
    }
    fval_t sig_mean = f32_to_i32(i32_to_f32(sig_sum, -15) / n, 30); /* Q0.30 */

    if (calc_stat_features) {
        if (!(is_magno)){
            SET_FEAT(F_ptr, g_feature_prefix, "rawstat_mean", "", sig_mean); /* Q0.30 */
        }
        set_feats_rawstat_summary(F_ptr, accel, sig_mean, n);
    }
    if (calc_autocorr_features) {
        set_feats_autocorr(F_ptr, accel, sig_mean, n);
    }
#if defined(FFT_FEATURES) || defined(MFCC_FEATURES)
    if (check_fft_true(calc_fft_sub_feats) || check_mfcc_true(calc_mfcc_sub_feats)){
        set_feats_fft(F_ptr, accel, sig_mean, n, samp_freq, is_microphone, calc_fft_sub_feats, calc_mfcc_sub_feats, adaptive_binning_sensor_channel_key);
    }
#endif /* mfcc_feature or fft_features */
    if (calc_peak_features) {
        set_feats_peak(F_ptr, accel, sig_mean, n, samp_freq);
    }
}
#endif /* ACCEL_FEATS or GYRO_FEATS or MICROPHONE_FEATS*/


/* calculate magnetometer features for one axis. also performs mean subtraction on that axis */
#ifdef MAGNO_FEATS
#ifdef PYTHON_FEATS
void calculate_mean_variance_per_axis_py(Features* F_ptr, char* prefix, int n, sigval_t* signal) {
    g_feature_prefix = prefix;
    calculate_mean_variance_per_axis(F_ptr, n, signal);
}
#endif /* PYTHON_FEATS */

void calculate_mean_variance_per_axis(Features* F_ptr, int n, sigval_t* signal) {
    /* TODO */
}

/* ASSUME: magno is mean subtracted along each axis (meaning sum(magno[i]) == 0) */
#ifdef PYTHON_FEATS
void calculate_axis_divergence_py(Features* F_ptr, char* prefix, int n_axes, int n_samp, sigval_t* signal, int* magno_channels) {
    g_feature_prefix = prefix;
#ifdef MAGNO_STAT_FEATURES
    calculate_axis_divergence(F_ptr, n_axes, n_samp, signal, magno_channels);
#endif
}
#endif

void calculate_axis_divergence(Features* F_ptr, int n_axes, int n_samp, sigval_t* signal, int* magno_channels) {
    /* TODO */
}
#endif /* MAGNO_FEATS */


/* Set features for one low frequency signal */
#ifdef LOWFREQ_FEATS
#ifdef PYTHON_FEATS
void calculate_low_freq_features_py(Features* F_ptr, char* prefix, sigval_t* sig, int n) {
    g_feature_prefix = prefix;
    calculate_low_freq_features(F_ptr, sig, n);
}
#endif

void calculate_low_freq_features(Features* F_ptr, sigval_t* sig, int n) {
    /* TODO */
}
#endif /* LOWFREQ_FEATS */

#endif /* USE_FIXEDPOINT */

#endif /* RAW_MODEL */

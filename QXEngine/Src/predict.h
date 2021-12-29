
/* THIS FILE WAS AUTO-GENERATED */

#ifndef PREDICT_H
#define PREDICT_H

#ifdef  __cplusplus
extern "C"
{
#endif

/* Accel sensor sampling rate required by this engine for FFT computations */
#define PRED_ACCEL_SAMPLING_FREQ 417.0
#define PRED_ACCEL_LOWPOWER_SAMPLING_FREQ 100.0
#define PRED_ACCEL_HIGHSENSITIVE_SAMPLING_FREQ 26667.0
/* Gyro sensor sampling rate required by this engine for FFT computations */
#define PRED_GYRO_SAMPLING_FREQ 417.0
/* Micro sensor data rate required by this engine */
#define PRED_MICROPHONE_SAMPLING_FREQ 16000
/* Number of accel samples required to perform a classification if using accel features */
#define PRED_NUM_ACCEL_SAMPLES 810
#define PRED_NUM_ACCEL_LOWPOWER_SAMPLES 195
#define PRED_NUM_ACCEL_HIGHSENSITIVE_SAMPLES 51800
/* Number of gyro samples required to perform a classification if using gyro features */
#define PRED_NUM_GYRO_SAMPLES 810
/* Number of magnetometer samples required to perform classification if using magnetometer features (100 Hz) */
#define PRED_NUM_MAGNO_SAMPLES 195
/* Number of pressure samples required to perform classification if they are used (100 Hz) */
#define PRED_NUM_PRESSURE_SAMPLES 195
/* Number of temperature samples required to perform classification if they are used (100 Hz) */
#define PRED_NUM_TEMPERATURE_SAMPLES 195
/* Number of proximity samples required to perform classification if it is used (25 Hz) */
#define PRED_NUM_PROXIMITY_SAMPLES 195
/* Number of ambient samples required to perform classification if it is used (100 Hz) */
#define PRED_NUM_AMBIENT_SAMPLES 195
/* Number of humidity samples required to perform classification if it is used (100 Hz) */
#define PRED_NUM_HUMIDITY_SAMPLES 49
/* Number of light samples required to perform classification if it is used (10 Hz) */
#define PRED_NUM_LIGHT_SAMPLES 32
/* Number of microphone samples required to perform classification if it is used (16000 Hz) */
#define PRED_NUM_MICROPHONE_SAMPLES 31080
/* Maximum number of samples for a single sensor */
#define PRED_NUM_SAMPLES_MAX 810
/* Number of features used by this engine */
#define PRED_NUM_FEATURES 1206
/* Number of samples used by this engine for each channel*/
#define PRED_NUM_SAMPLES 810
/* Number of samples after padding till next power of 2 */
#define MAX_FFT_SIG_LEN 1024

//#define WEIGHTED_AVERAGE
/* Number of class labels */
#define NUM_CLASSES 3
// number of predictions we store for event classification
#define MAX_NUM_INSTANCE 10

/* Use contamination_threshold instead of mSensitivity if ONE_CLASS_MODEL is defined*/
//#define ONE_CLASS_MODEL
//#define HYBRID_MODEL
/* if ON_DEVICE_ONE_CLASS_MODEL is defined*/
// #define ON_DEVICE_ONE_CLASS_MODEL
/* if ON_DEVICE_MULTI_CLASS_MODEL is defined*/
// #define ON_DEVICE_MULTI_CLASS_MODEL
static float mSensitivity[NUM_CLASSES] = {1,1,1};

// event classification status
static int EVENT_CLASSIFICATION_DISABLED = 0;
static int START_EVENT_CLASSIFICATION = 1;
static int STOP_EVENT_CLASSIFICATION = 2;

#ifndef RAW_MODEL
/* Indicator list for event class */
static int event_classes[] = {1,1,0};
#endif

/* Channel variables represent x, y, z axes or c, r, g, b axes; 1 is turned ON and 0 is OFF */

#if defined(QXAUTOMLCONFIG_SENSOR_ENABLE_ACCEL) || defined(QXAUTOMLCONFIG_SENSOR_ENABLE_ACCEL_LOWPOWER) || defined(QXAUTOMLCONFIG_SENSOR_ENABLE_ACCEL_HIGHSENSITIVE)
static int accel_channels[] = {1,1,1};
#endif

#ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_GYRO
static int gyro_channels[] = {1,1,1};
#endif

#ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_MAG
static int magno_channels[] = {0,0,0};
#endif

#ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_AMBIENT
static int ambient_channels[] = {1,1,1,1};
#endif

/*define initial Contamination_threshold*/
/* change to -2 for now, need to update later*/


#if defined (ONE_CLASS_MODEL) || defined(HYBRID_MODEL)
static float Contamination_threshold = 0.5;
#ifdef ON_DEVICE_ONE_CLASS_MODEL
static int collected_samples_counter = 0; // counter to count how many samples have been collected
static int start_count_down = 60; // initial count down to give user some time before data collection
static int train_lof = 1; // binary value, 0 or 1 to tell the code if the LOF is training or not
static int train_if = 1; // binary value, 0 or 1 to tell the code if the IF is training or not
static int train_wait_count_down = 25; // Extra time count down to show visualtization during training
#endif
#endif

#ifdef ON_DEVICE_MULTI_CLASS_MODEL
static int collected_samples_counter = 0; // counter to count how many samples have been collected
static int start_count_down = 60; // initial count down to give user some time before data collection
static int train_model = 1; // binary value, 0 or 1 to tell the code if the LOF is training or not
static int train_wait_count_down = 25; // Extra time count down to show visualtization during training
static int collected_samples_per_class = 0;
static int class_being_collected = 0;
static int count_down_in_between_classes = 25;
static int label_counter = 0;
#endif

/* ON-device training parameters */
#ifdef ON_DEVICE_ONE_CLASS_MODEL
typedef struct {
    int samples_collected;  // How many samples have been collected
    int collect_data; // binary value, 0 or 1 to tell the code if it is time to collect data
    int start_time_left; // How much time is left to start the data collection
    int train_wait_left; // How much time is left during training visualization
} on_device_one_class_train_state_t;
#endif

/* ON-device training parameters */
#ifdef ON_DEVICE_MULTI_CLASS_MODEL
typedef struct {
    int samples_collected;  // How many samples have been collected
    int collect_data; // binary value, 0 or 1 to tell the code if it is time to collect data
    int start_time_left; // How much time is left to start the data collection
    int train_wait_left; // How much time is left during training visualization
    int class_being_collected;
    int in_between_class_time_left;
} on_device_multi_class_train_state_t;
#endif

/* Prediction Class */

typedef struct {
    float probs[NUM_CLASSES];
    int current_num_instances;
    int event_classification_status;
#ifdef ON_DEVICE_ONE_CLASS_MODEL
    on_device_one_class_train_state_t training_state;
#endif
#ifdef ON_DEVICE_MULTI_CLASS_MODEL
    on_device_multi_class_train_state_t training_state;
#endif
#ifdef PYTHON_PRED
    char* class_names[NUM_CLASSES];
#endif
} Prediction;

/* Feature Struct: defined in same way as production version of features.h */

#ifdef PYTHON_PRED
/* This is copied from qxo-features.h - TODO, deduplicate this! */
#include <stdint.h>
#ifdef USE_FIXEDPOINT
typedef int32_t fval_t;
typedef int16_t sigval_t;
#else
typedef float fval_t;
typedef float sigval_t;
#endif

typedef struct { /* Feature contains only a value if called from C */
    fval_t val;
} Feature;

/* data type for sigval_t and fval_t in that order */
enum DType {
	DTYPE_F32_F32 = 0,
	DTYPE_I16_I32 = 1,
};

/* This function is not static because it needs to be exported,
   but it's weak because multiple copies might exist. (predict
   and featurize both need to export this symbol, but they might
   be compiled together) */
enum DType __attribute__((weak)) get_dtype(void) {
#ifdef USE_FIXEDPOINT
    return DTYPE_I16_I32;
#else
    return DTYPE_F32_F32;
#endif
}
#else
#include "qxo_features.h"
#endif

/* PUBLIC */

Prediction* init_predict();
void predict(Feature* feats, Prediction* pred);
void free_predict(Prediction* pred);

#ifdef  __cplusplus
}
#endif

#endif /* PREDICT_H */

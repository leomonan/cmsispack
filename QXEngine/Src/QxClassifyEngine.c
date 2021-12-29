/*
 * QxClassifyEngine.c
 *
 * Created for APIs calling by embedded device
 */

#ifndef RAW_MODEL
#include "qxo_features.h"
#endif
#include "predict.h"

#define QXO_GLOBALS
#include "QxClassifyEngine.h"


SensorData sensorSets[SENSOR_TYPE_MAX-1]={0};

PredictionFrame predFrame = {
    .mEnabledSensorCount =0,
    .mSensorData = &sensorSets[0],
};

static int16_t mSensorTypeIndex[SENSOR_TYPE_MAX];
#define COMBINE_AXIS_KEY 4
#define ACCEL 1
#define ACCEL_LOWPOWER 10
#define ACCEL_HIGHSENSITIVE 14


#ifndef RAW_MODEL
  #ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_MAG
    /* XXX MAG is the only sensor that requires all axes simultaneously */
    #define MAX(a,b) (((a)>(b))?(a):(b))
    static sigval_t sensor_data[MAX(PRED_NUM_SAMPLES, 3 * PRED_NUM_MAGNO_SAMPLES)];
  #else
    static sigval_t sensor_data[PRED_NUM_SAMPLES];
  #endif
#endif // ifndef RAW_MODEL

#ifndef RAW_MODEL
static Features features = {
    .num_feats_total = PRED_NUM_FEATURES,
    .num_feats_current = 0,
    .feats = (Feature[PRED_NUM_FEATURES]) {},
};
#else
static sigval_t raw_data_array[RAW_DATA_SIZE];
#endif


#define N_WEIGHTS 12

typedef struct SensorTileState {
    float weights[N_WEIGHTS];
    float prob_buf[N_WEIGHTS][NUM_CLASSES];
    uint8_t prob_buf_i;
    uint8_t lapped; /* boolean */
} SensorTileState;

static Prediction pred = { 0 };

static uint8_t SensorTileState_update(const float probs[])
{
    int i, j, k;
    static struct SensorTileState sts = {
        /* Gaussian weights
         * equivelent to the following in Python:
         *     bandwidth = 2.0
         *     maxlen = int(bandwidth * 4)
         *     weights = norm.pdf(np.linspace(0, maxlen, num = maxlen), 0, bandwidth)
         */
        .weights = {0.1329807601338109, 0.12447299125132628, 0.10207845121270008, 0.07334417845666358, 0.046171061041316026, 0.025465176595022256, 0.012305412027642874, 0.005209766492285203, 0.0019324710068276772, 0.000628029936717939, 0.00017882174794729794, 4.461007525496179e-05},

        /* circular buffer of probabilities for past classifications */
        .prob_buf = {},
        .prob_buf_i = 0,
        .lapped = 0 /* have not yet completed first lap */
    };

    for (i = 0; i < NUM_CLASSES; i++) {
        sts.prob_buf[sts.prob_buf_i][i] = probs[i];
    }

    /* prediction using probabilities from previous times and gaussian weights */
    float class_probs[NUM_CLASSES] = {0};
    j = 0;
    for (i = sts.prob_buf_i; j < N_WEIGHTS; i--) {

        /* first lap of buffer not complete yet */
        if ((i < 0) && (sts.lapped == 0)) break;

        /* wrap from left to right */
        if (i < 0) i = N_WEIGHTS-1;

        for (k = 0; k < NUM_CLASSES; k++) {
            class_probs[k] += sts.prob_buf[i][k] * sts.weights[j];
        }
        j++;
    }

    /* iterate circular buffer */
    sts.prob_buf_i++;
    if (sts.prob_buf_i >= N_WEIGHTS) {
        sts.prob_buf_i = 0;
        sts.lapped = 1;
    }

    /* class with highest probability */
    j = 0;
    for (i = 1; i < NUM_CLASSES; i++) {
        if (class_probs[i] > class_probs[j])
            j = i;
    }
    return j;
}


static uint8_t State_update(float probs[], int* event_classes_l){
    // weighted prediction
    int cls_weighted = SensorTileState_update(probs);

    // non-weighted prediction
    int cls = 0;
    for (int c = 1; c < NUM_CLASSES; c++) {
        if (probs[c] > probs[cls]) cls = c;
    }

    if (event_classes_l[cls] != 1){
        cls = cls_weighted;
    }

    return cls;
}


static void apply_sensitivity(Prediction* pred_l, float *pSensitivity){
    float sum_ = 0;
    for (int c = 0; c < NUM_CLASSES; c++) {
        pred_l->probs[c] = pred_l->probs[c] * pSensitivity[c];
        sum_ += pred_l->probs[c];
    }
    // renormalize the the probability
    for (int c = 0; c < NUM_CLASSES; c++) {
        pred_l->probs[c] = pred_l->probs[c]/sum_;
    }
}


#if defined(ONE_CLASS_MODEL) || defined (HYBRID_MODEL)
static void apply_Contamination_threshold(Prediction* pred, int class_i){
    #ifdef ONE_CLASS_MODEL
        if (pred -> probs[0] < Contamination_threshold){
            pred -> probs[0] = 1;
            pred -> probs[1] = 0;
        }
        else{
            pred -> probs[0] = 0;
            pred -> probs[1] = 1;
        }
    #endif

    #ifdef HYBRID_MODEL

        // For Hybrid model Score is the probability of the Outlier. So we use 1 - pred -> probs[0]

        if (1 - pred -> probs[0] < Contamination_threshold){
            pred -> probs[0] = 1;
            pred -> probs[1] = 0;
        }
        else{
            pred -> probs[0] = 0;
            pred -> probs[1] = 1;
        }
    #endif

}
#endif

pPredictionFrame QXO_MLEngine_Init(void) {

// TODO Auto-generated constructor stub
    for(uint16_t i = 0; i < SENSOR_TYPE_MAX; i++ ) {
        mSensorTypeIndex[i] = -1;
    }
    for(uint16_t i = 0; i < enabled_sensors_count; i++)
    {
        predFrame.mEnabledSensorCount++;

        sensorSets[i].sensor_type = enabled_sensors[i].type;
        mSensorTypeIndex[sensorSets[i].sensor_type] = i;
        sensorSets[i].buff_end = 0;
        sensorSets[i].buff_max = enabled_sensors[i].buf_count*enabled_sensors[i].sample_bytes;
        sensorSets[i].odr = enabled_sensors[i].odr;
        sensorSets[i].fsr = enabled_sensors[i].fsr;
        sensorSets[i].bytes_per_sample = enabled_sensors[i].sample_bytes;
        sensorSets[i].buff_ptr = enabled_sensors[i].buf;
    }
    predFrame.mFrameMutex = QxOS_CreateMutex("rw_mutex");

#ifndef RAW_MODEL
    startup_feats();
#endif
    return &predFrame;
}

static void ExtractPredictionFrame(pPredictionFrame frame){

#ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_PROXIMITY
    uint8_t* ptr_uint8 = NULL;
#endif

#if defined(QXAUTOMLCONFIG_SENSOR_ENABLE_ACCEL) || defined(QXAUTOMLCONFIG_SENSOR_ENABLE_ACCEL_LOWPOWER) || defined(QXAUTOMLCONFIG_SENSOR_ENABLE_ACCEL_HIGHSENSITIVE) || defined(QXAUTOMLCONFIG_SENSOR_ENABLE_GYRO) || defined(QXAUTOMLCONFIG_SENSOR_ENABLE_MAG) || defined(QXAUTOMLCONFIG_SENSOR_ENABLE_TEMPERATURE_EXT1) || defined(QXAUTOMLCONFIG_SENSOR_ENABLE_TEMPERATURE) ||defined(QXAUTOMLCONFIG_SENSOR_ENABLE_HUMIDITY) || defined(QXAUTOMLCONFIG_SENSOR_ENABLE_MICROPHONE) || defined(QXAUTOMLCONFIG_SENSOR_ENABLE_AMBIENT) || defined(QXAUTOMLCONFIG_SENSOR_ENABLE_LIGHT)
    int16_t* ptr_int16 = NULL;
#endif

#if defined(QXAUTOMLCONFIG_SENSOR_ENABLE_PRESSURE)|| defined(QXAUTOMLCONFIG_SENSOR_ENABLE_ECO2)|| defined(QXAUTOMLCONFIG_SENSOR_ENABLE_TVOC) || defined(QXAUTOMLCONFIG_SENSOR_ENABLE_ETOH) || defined(QXAUTOMLCONFIG_SENSOR_ENABLE_RCDA) || defined(QXAUTOMLCONFIG_SENSOR_ENABLE_IAQ)
    int32_t* ptr_int32 = NULL;
#endif

#if defined(QXAUTOMLCONFIG_SENSOR_ENABLE_RMOX)
    uint32_t* ptr_uint32 = NULL;
#endif

    PredictionFrame* frame_ptr = frame;

#ifndef RAW_MODEL
#ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_ACCEL
    QxOS_LockMutex(frame_ptr->mFrameMutex);
    ptr_int16 = (int16_t*)frame_ptr->mSensorData[mSensorTypeIndex[SENSOR_TYPE_ACCEL]].buff_ptr;

    #ifndef COMBINE_ACCEL_AXES
        for(int axis = 0; axis < 3; axis++) {
            if (accel_channels[axis] == 1) {
                for (int i = 0; i < PRED_NUM_ACCEL_SAMPLES; i++) {
                    sensor_data[i] = ptr_int16[i * 3 + axis];
                }
                calc_accel_features(&features, sensor_data, PRED_NUM_ACCEL_SAMPLES, PRED_ACCEL_SAMPLING_FREQ, ACCEL, axis);
            }
        }

        QxOS_UnLockMutex(frame_ptr->mFrameMutex);

    #else

        for (int i = 0; i < PRED_NUM_ACCEL_SAMPLES; i++) {
            sensor_data[i] = sqrtf(ptr_int16[3*i]*ptr_int16[3*i] + ptr_int16[3*i+1]*ptr_int16[3*i+1] + ptr_int16[3*i + 2]*ptr_int16[3*i + 2]);
        }

        calc_accel_features(&features, sensor_data, PRED_NUM_ACCEL_SAMPLES, PRED_ACCEL_SAMPLING_FREQ, ACCEL, COMBINE_AXIS_KEY);
        QxOS_UnLockMutex(frame_ptr->mFrameMutex);

    #endif
#endif

#ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_ACCEL_LOWPOWER
    QxOS_LockMutex(frame_ptr->mFrameMutex);
    ptr_int16 = (int16_t*)frame_ptr->mSensorData[mSensorTypeIndex[SENSOR_TYPE_ACCEL_LOWPOWER]].buff_ptr;

    #ifndef COMBINE_ACCEL_LOWPOWER_AXES
        for(int axis = 0; axis < 3; axis++) {
            if (accel_channels[axis] == 1) {
                for (int i = 0; i < PRED_NUM_ACCEL_LOWPOWER_SAMPLES; i++) {
                    sensor_data[i] = ptr_int16[i * 3 + axis];
                }
                calc_accel_features(&features, sensor_data, PRED_NUM_ACCEL_LOWPOWER_SAMPLES, PRED_ACCEL_LOWPOWER_SAMPLING_FREQ, ACCEL_LOWPOWER, axis);
            }
        }

        QxOS_UnLockMutex(frame_ptr->mFrameMutex);

    #else

        for (int i = 0; i < PRED_NUM_ACCEL_LOWPOWER_SAMPLES; i++) {
            sensor_data[i] = sqrtf(ptr_int16[3*i]*ptr_int16[3*i] + ptr_int16[3*i+1]*ptr_int16[3*i+1] + ptr_int16[3*i + 2]*ptr_int16[3*i + 2]);
        }

        calc_accel_features(&features, sensor_data, PRED_NUM_ACCEL_LOWPOWER_SAMPLES, PRED_ACCEL_LOWPOWER_SAMPLING_FREQ, ACCEL_LOWPOWER, COMBINE_AXIS_KEY);
        QxOS_UnLockMutex(frame_ptr->mFrameMutex);

    #endif

#endif

#ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_ACCEL_HIGHSENSITIVE
    QxOS_LockMutex(frame_ptr->mFrameMutex);
    ptr_int16 = (int16_t*)frame_ptr->mSensorData[mSensorTypeIndex[SENSOR_TYPE_ACCEL_HIGHSENSITIVE]].buff_ptr;

    #ifndef COMBINE_ACCEL_HIGHSENSITIVE_AXES

        for(int axis = 0; axis < 3; axis++) {
            if (accel_channels[axis] == 1) {
                for (int i = 0; i < PRED_NUM_ACCEL_HIGHSENSITIVE_SAMPLES; i++) {
                    sensor_data[i] = ptr_int16[i * 3 + axis];
                }
                calc_accel_features(&features, sensor_data, PRED_NUM_ACCEL_HIGHSENSITIVE_SAMPLES, PRED_ACCEL_HIGHSENSITIVE_SAMPLING_FREQ, ACCEL_HIGHSENSITIVE, axis);
            }
        }

        QxOS_UnLockMutex(frame_ptr->mFrameMutex);

    #else

        for (int i = 0; i < PRED_NUM_ACCEL_HIGHSENSITIVE_SAMPLES; i++) {
            sensor_data[i] = sqrtf(ptr_int16[3*i]*ptr_int16[3*i] + ptr_int16[3*i+1]*ptr_int16[3*i+1] + ptr_int16[3*i + 2]*ptr_int16[3*i + 2]);
        }
        calc_accel_features(&features, sensor_data, PRED_NUM_ACCEL_HIGHSENSITIVE_SAMPLES, PRED_ACCEL_HIGHSENSITIVE_SAMPLING_FREQ, ACCEL_HIGHSENSITIVE, COMBINE_AXIS_KEY);
        QxOS_UnLockMutex(frame_ptr->mFrameMutex);

    #endif

#endif

#ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_GYRO
    QxOS_LockMutex(frame_ptr->mFrameMutex);
    ptr_int16 = (int16_t*)frame_ptr->mSensorData[mSensorTypeIndex[SENSOR_TYPE_GYRO]].buff_ptr;

    #ifndef COMBINE_GYRO_AXES
        for(int axis = 0; axis < 3; axis++) {
            if (gyro_channels[axis] == 1) {
                for (int i = 0; i < PRED_NUM_GYRO_SAMPLES; i++) {
                    sensor_data[i] = ptr_int16[i * 3 + axis];
                }
                calc_gyro_features(&features, sensor_data, PRED_NUM_GYRO_SAMPLES, PRED_GYRO_SAMPLING_FREQ, axis);
            }
        }

        QxOS_UnLockMutex(frame_ptr->mFrameMutex);

    #else

        for (int i = 0; i < PRED_NUM_GYRO_SAMPLES; i++) {
            sensor_data[i] = sqrtf(ptr_int16[3*i]*ptr_int16[3*i] + ptr_int16[3*i+1]*ptr_int16[3*i+1] + ptr_int16[3*i + 2]*ptr_int16[3*i + 2]);
        }

        calc_gyro_features(&features, sensor_data, PRED_NUM_GYRO_SAMPLES, PRED_GYRO_SAMPLING_FREQ, COMBINE_AXIS_KEY);
        QxOS_UnLockMutex(frame_ptr->mFrameMutex);

    #endif

#endif

#ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_MAG
    QxOS_LockMutex(frame_ptr->mFrameMutex);
    ptr_int16 = (int16_t*)frame_ptr->mSensorData[mSensorTypeIndex[SENSOR_TYPE_MAG]].buff_ptr;
    for (int axis = 0; axis < 3; axis++) {
        for (int i = 0; i < PRED_NUM_MAGNO_SAMPLES; i++) {
            sensor_data[axis*PRED_NUM_MAGNO_SAMPLES + i] = ptr_int16[i * 3 + axis]; /* magno buf is filled x0,y0,z0, x1,y1,z1, ... */
        }
    }
    QxOS_UnLockMutex(frame_ptr->mFrameMutex);

    for (int axis = 0; axis < 3; axis++) {
        if (magno_channels[axis] == 1) {
            QxOS_LockMutex(frame_ptr->mFrameMutex);
            calc_magno_features(&features, &sensor_data[axis*PRED_NUM_MAGNO_SAMPLES], PRED_NUM_MAGNO_SAMPLES, PRED_MAGNO_SAMPLING_FREQ);
            QxOS_UnLockMutex(frame_ptr->mFrameMutex);
        }
    }
#ifdef MAGNO_STAT_FEATURES
    calculate_axis_divergence(&features, 3, PRED_NUM_MAGNO_SAMPLES, sensor_data, magno_channels);
#endif
#endif

#ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_TEMPERATURE_EXT1

    QxOS_LockMutex(frame_ptr->mFrameMutex);
    ptr_int16 = (int16_t*)frame_ptr->mSensorData[mSensorTypeIndex[SENSOR_TYPE_TEMPERATURE_EXT1]].buff_ptr;
    for (int i = 0; i < PRED_NUM_TEMPERATURE_SAMPLES; i++) {
        sensor_data[i] = ptr_int16[i];
    }
    QxOS_UnLockMutex(frame_ptr->mFrameMutex);
    calculate_low_freq_features(&features, sensor_data, PRED_NUM_TEMPERATURE_SAMPLES);

#endif

#ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_TEMPERATURE
    QxOS_LockMutex(frame_ptr->mFrameMutex);
    ptr_int16 = (int16_t*)frame_ptr->mSensorData[mSensorTypeIndex[SENSOR_TYPE_TEMPERATURE]].buff_ptr;;
    for (int i = 0; i < PRED_NUM_TEMPERATURE_SAMPLES; i++) {
        sensor_data[i] = ptr_int16[i];
    }
    QxOS_UnLockMutex(frame_ptr->mFrameMutex);
    calculate_low_freq_features(&features, sensor_data, PRED_NUM_TEMPERATURE_SAMPLES);
#endif

#ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_PRESSURE
    QxOS_LockMutex(frame_ptr->mFrameMutex);
    ptr_int32 = (int32_t*)frame_ptr->mSensorData[mSensorTypeIndex[SENSOR_TYPE_PRESSURE]].buff_ptr;
    for (int i = 0; i < PRED_NUM_PRESSURE_SAMPLES; i++) {
        sensor_data[i] = ptr_int32[i];
    }
    QxOS_UnLockMutex(frame_ptr->mFrameMutex);
    calculate_low_freq_features(&features, sensor_data, PRED_NUM_PRESSURE_SAMPLES);
#endif

#ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_HUMIDITY
    QxOS_LockMutex(frame_ptr->mFrameMutex);
    ptr_int16 = (int16_t*)frame_ptr->mSensorData[mSensorTypeIndex[SENSOR_TYPE_HUMIDITY]].buff_ptr;
    for (int i = 0; i < PRED_NUM_HUMIDITY_SAMPLES; i++) {
        sensor_data[i] = ptr_int16[i];
    }
    QxOS_UnLockMutex(frame_ptr->mFrameMutex);
    calculate_low_freq_features(&features, sensor_data, PRED_NUM_HUMIDITY_SAMPLES);

#endif

#ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_MICROPHONE
    QxOS_LockMutex(frame_ptr->mFrameMutex);
    ptr_int16 = (int16_t*)frame_ptr->mSensorData[mSensorTypeIndex[SENSOR_TYPE_MICROPHONE]].buff_ptr;
    for (int i = 0; i < PRED_NUM_MICROPHONE_SAMPLES; i++) {
        sensor_data[i] = ptr_int16[i];
    }
    QxOS_UnLockMutex(frame_ptr->mFrameMutex);
    calc_microphone_features(&features, sensor_data, PRED_NUM_MICROPHONE_SAMPLES, PRED_MICROPHONE_SAMPLING_FREQ);

#endif

#ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_PROXIMITY
    QxOS_LockMutex(frame_ptr->mFrameMutex);
    ptr_uint8 = (uint8_t*)frame_ptr->mSensorData[mSensorTypeIndex[SENSOR_TYPE_PROXIMITY]].buff_ptr;
    for (int i = 0; i < PRED_NUM_PROXIMITY_SAMPLES; i++) {
        sensor_data[i] = ptr_uint8[i];
    }
    QxOS_UnLockMutex(frame_ptr->mFrameMutex);
    calculate_low_freq_features(&features, sensor_data, PRED_NUM_PROXIMITY_SAMPLES);
#endif

#ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_AMBIENT
    QxOS_LockMutex(frame_ptr->mFrameMutex);
    ptr_int16 = (int16_t*)frame_ptr->mSensorData[mSensorTypeIndex[SENSOR_TYPE_AMBIENT]].buff_ptr;
    for(int axis = 0; axis < 4; axis++) {
        if (ambient_channels[axis] == 1) {
            for (int i = 0; i < PRED_NUM_AMBIENT_SAMPLES; i++) {
                sensor_data[i] = ptr_int16[i * 4 + axis];
            }
            calculate_low_freq_features(&features, sensor_data, PRED_NUM_AMBIENT_SAMPLES);
        }
    }
    QxOS_UnLockMutex(frame_ptr->mFrameMutex);
#endif

#ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_LIGHT
    QxOS_LockMutex(frame_ptr->mFrameMutex);
    ptr_int16 = (int16_t*)frame_ptr->mSensorData[mSensorTypeIndex[SENSOR_TYPE_LIGHT]].buff_ptr;
    for (int i = 0; i < PRED_NUM_LIGHT_SAMPLES; i++) {
        sensor_data[i] = ptr_int16[i];
    }
    QxOS_UnLockMutex(frame_ptr->mFrameMutex);
    calculate_low_freq_features(&features, sensor_data, PRED_NUM_LIGHT_SAMPLES);
#endif

#ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_TVOC
    QxOS_LockMutex(frame_ptr->mFrameMutex);
    ptr_int32 = (int32_t *)frame_ptr->mSensorData[mSensorTypeIndex[SENSOR_TYPE_TVOC]].buff_ptr;
    for (int i = 0; i < PRED_NUM_TVOC_SAMPLES; i++) {
        sensor_data[i] = ptr_int32[i];
    }
    QxOS_UnLockMutex(frame_ptr->mFrameMutex);
    calculate_low_freq_features(&features, sensor_data, PRED_NUM_TVOC_SAMPLES);
#endif

#ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_ECO2
    QxOS_LockMutex(frame_ptr->mFrameMutex);
    ptr_int32 = (int32_t *)frame_ptr->mSensorData[mSensorTypeIndex[SENSOR_TYPE_ECO2]].buff_ptr;
    for (int i = 0; i < PRED_NUM_ECO2_SAMPLES; i++) {
        sensor_data[i] = ptr_int32[i];
    }
    QxOS_UnLockMutex(frame_ptr->mFrameMutex);
    calculate_low_freq_features(&features, sensor_data, PRED_NUM_ECO2_SAMPLES);
#endif

#ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_ETOH
    QxOS_LockMutex(frame_ptr->mFrameMutex);
    ptr_int32 = (int32_t *)frame_ptr->mSensorData[mSensorTypeIndex[SENSOR_TYPE_ETOH]].buff_ptr;
    for (int i = 0; i < PRED_NUM_ETOH_SAMPLES; i++) {
        sensor_data[i] = ptr_int32[i];
    }
    QxOS_UnLockMutex(frame_ptr->mFrameMutex);
    calculate_low_freq_features(&features, sensor_data, PRED_NUM_ETOH_SAMPLES);
#endif

#ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_RCDA
    QxOS_LockMutex(frame_ptr->mFrameMutex);
    ptr_int32 = (int32_t *)frame_ptr->mSensorData[mSensorTypeIndex[SENSOR_TYPE_RCDA]].buff_ptr;
    for (int i = 0; i < PRED_NUM_RCDA_SAMPLES; i++) {
        sensor_data[i] = ptr_int32[i];
    }
    QxOS_UnLockMutex(frame_ptr->mFrameMutex);
    calculate_low_freq_features(&features, sensor_data, PRED_NUM_RCDA_SAMPLES);
#endif

#ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_IAQ
    QxOS_LockMutex(frame_ptr->mFrameMutex);
    ptr_int32 = (int32_t *)frame_ptr->mSensorData[mSensorTypeIndex[SENSOR_TYPE_IAQ]].buff_ptr;
    for (int i = 0; i < PRED_NUM_IAQ_SAMPLES; i++) {
        sensor_data[i] = ptr_int32[i];
    }
    QxOS_UnLockMutex(frame_ptr->mFrameMutex);
    calculate_low_freq_features(&features, sensor_data, PRED_NUM_IAQ_SAMPLES);
#endif

#ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_RMOX
    QxOS_LockMutex(frame_ptr->mFrameMutex);
    ptr_uint32 = (uint32_t *)frame_ptr->mSensorData[mSensorTypeIndex[SENSOR_TYPE_RMOX]].buff_ptr;

    for (int axis = 0; axis < 13; axis++) {
        for (int i = 0; i < PRED_NUM_RMOX_SAMPLES; i++) {
            sensor_data[i] = ptr_uint32[i*13 + axis]; /* magno buf is filled x0,y0,z0, x1,y1,z1, ... */
        }
        calculate_low_freq_features(&features, sensor_data, PRED_NUM_RMOX_SAMPLES);
    }
    QxOS_UnLockMutex(frame_ptr->mFrameMutex);
#endif

#elif defined MODEL_KIND_CNN
// The data format is channels-first to match onnx2c's native data format.
int selected_channels = 0;
#ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_ACCEL
    QxOS_LockMutex(frame_ptr->mFrameMutex);
    ptr_int16 = (int16_t*)frame_ptr->mSensorData[mSensorTypeIndex[SENSOR_TYPE_ACCEL]].buff_ptr;
    selected_channels = 0;
    for(int axis = 0; axis < 3; axis++) {
        if (accel_channels[axis] == 1) {
            for (int i = 0; i < PRED_NUM_ACCEL_SAMPLES; i++) {
                raw_data_array[ACCEL_RAW_DATA_OFFSET + (selected_channels + ACCEL_CHANNEL_OFFSET) * PRED_NUM_ACCEL_SAMPLES + i] = ptr_int16[i * 3 + axis];
            }
            selected_channels += 1;
        }
    }
    QxOS_UnLockMutex(frame_ptr->mFrameMutex);
#endif

#ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_ACCEL_HIGHSENSITIVE
    QxOS_LockMutex(frame_ptr->mFrameMutex);
    ptr_int16 = (int16_t*)frame_ptr->mSensorData[mSensorTypeIndex[SENSOR_TYPE_ACCEL_HIGHSENSITIVE]].buff_ptr;
    selected_channels = 0;
    for(int axis = 0; axis < 3; axis++) {
        if (accel_channels[axis] == 1) {
            for (int i = 0; i < PRED_NUM_ACCEL_HIGHSENSITIVE_SAMPLES; i++) {
                raw_data_array[ACCEL_HIGHSENSITIVE_RAW_DATA_OFFSET + (selected_channels + ACCEL_HIGHSENSITIVE_CHANNEL_OFFSET) * PRED_NUM_ACCEL_HIGHSENSITIVE_SAMPLES + i] = ptr_int16[i * 3 + axis];
            }
            selected_channels += 1;
        }
    }
    QxOS_UnLockMutex(frame_ptr->mFrameMutex);
#endif

#ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_ACCEL_LOWPOWER
    QxOS_LockMutex(frame_ptr->mFrameMutex);
    ptr_int16 = (int16_t*)frame_ptr->mSensorData[mSensorTypeIndex[SENSOR_TYPE_ACCEL_LOWPOWER]].buff_ptr;
    selected_channels = 0;
    for(int axis = 0; axis < 3; axis++) {
        if (accel_channels[axis] == 1) {
            for (int i = 0; i < PRED_NUM_ACCEL_LOWPOWER_SAMPLES; i++) {
                raw_data_array[ACCEL_LOWPOWER_RAW_DATA_OFFSET + (selected_channels + ACCEL_LOWPOWER_CHANNEL_OFFSET) * PRED_NUM_ACCEL_LOWPOWER_SAMPLES + i] = ptr_int16[i * 3 + axis];
            }
            selected_channels += 1;
        }
    }
    QxOS_UnLockMutex(frame_ptr->mFrameMutex);
#endif

#ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_GYRO
    QxOS_LockMutex(frame_ptr->mFrameMutex);
    ptr_int16 = (int16_t*)frame_ptr->mSensorData[mSensorTypeIndex[SENSOR_TYPE_GYRO]].buff_ptr;
    selected_channels = 0;
    for(int axis = 0; axis < 3; axis++) {
        if (gyro_channels[axis] == 1) {
            for (int i = 0; i < PRED_NUM_GYRO_SAMPLES; i++) {
                raw_data_array[GYRO_RAW_DATA_OFFSET + (selected_channels + GYRO_CHANNEL_OFFSET) * PRED_NUM_GYRO_SAMPLES + i] = ptr_int16[i * 3 + axis];
            }
            selected_channels += 1;
        }
    }
    QxOS_UnLockMutex(frame_ptr->mFrameMutex);
#endif

#ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_MAG
    QxOS_LockMutex(frame_ptr->mFrameMutex);
    ptr_int16 = (int16_t*)frame_ptr->mSensorData[mSensorTypeIndex[SENSOR_TYPE_MAG]].buff_ptr;
    selected_channels = 0;
    for(int axis = 0; axis < 3; axis++) {
        if (magno_channels[axis] == 1) {
            for (int i = 0; i < PRED_NUM_MAGNO_SAMPLES; i++) {
                raw_data_array[MAG_RAW_DATA_OFFSET + (selected_channels + MAG_CHANNEL_OFFSET) * PRED_NUM_MAGNO_SAMPLES + i] = ptr_int16[i * 3 + axis];
            }
            selected_channels += 1;
        }
    }
    QxOS_UnLockMutex(frame_ptr->mFrameMutex);
#endif

#ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_TEMPERATURE_EXT1
    QxOS_LockMutex(frame_ptr->mFrameMutex);
    ptr_int16 = (int16_t*)frame_ptr->mSensorData[mSensorTypeIndex[SENSOR_TYPE_TEMPERATURE_EXT1]].buff_ptr;
    for (int i = 0; i < PRED_NUM_TEMPERATURE_SAMPLES; i++) {
        raw_data_array[TEMPERATURE_RAW_DATA_OFFSET + TEMPERATURE_CHANNEL_OFFSET * PRED_NUM_TEMPERATURE_SAMPLES + i] = ptr_int16[i];
    }
    QxOS_UnLockMutex(frame_ptr->mFrameMutex);
#endif

#ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_TEMPERATURE
    QxOS_LockMutex(frame_ptr->mFrameMutex);
    ptr_int16 = (int16_t*)frame_ptr->mSensorData[mSensorTypeIndex[SENSOR_TYPE_TEMPERATURE]].buff_ptr;
    for (int i = 0; i < PRED_NUM_TEMPERATURE_SAMPLES; i++) {
        raw_data_array[TEMPERATURE_RAW_DATA_OFFSET + TEMPERATURE_CHANNEL_OFFSET * PRED_NUM_TEMPERATURE_SAMPLES + i] = ptr_int16[i];
    }
    QxOS_UnLockMutex(frame_ptr->mFrameMutex);
#endif

#ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_PRESSURE
    QxOS_LockMutex(frame_ptr->mFrameMutex);
    ptr_int32 = (int32_t*)frame_ptr->mSensorData[mSensorTypeIndex[SENSOR_TYPE_PRESSURE]].buff_ptr;;
    for (int i = 0; i < PRED_NUM_PRESSURE_SAMPLES; i++) {
        raw_data_array[PRESSURE_RAW_DATA_OFFSET + PRESSURE_CHANNEL_OFFSET * PRED_NUM_PRESSURE_SAMPLES + i] = ptr_int32[i];
    }
    QxOS_UnLockMutex(frame_ptr->mFrameMutex);
#endif

#ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_HUMIDITY
    QxOS_LockMutex(frame_ptr->mFrameMutex);
    ptr_int16 = (int16_t*)frame_ptr->mSensorData[mSensorTypeIndex[SENSOR_TYPE_HUMIDITY]].buff_ptr;
    for (int i = 0; i < PRED_NUM_HUMIDITY_SAMPLES; i++) {
        raw_data_array[HUMIDITY_RAW_DATA_OFFSET + HUMIDITY_CHANNEL_OFFSET * PRED_NUM_HUMIDITY_SAMPLES + i] = ptr_int16[i];
    }
    QxOS_UnLockMutex(frame_ptr->mFrameMutex);
#endif

#ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_MICROPHONE
    QxOS_LockMutex(frame_ptr->mFrameMutex);
    ptr_int16 = (int16_t*)frame_ptr->mSensorData[mSensorTypeIndex[SENSOR_TYPE_MICROPHONE]].buff_ptr;
    for (int i = 0; i < PRED_NUM_MICROPHONE_SAMPLES; i++) {
        raw_data_array[MICROPHONE_RAW_DATA_OFFSET + MICROPHONE_CHANNEL_OFFSET * PRED_NUM_MICROPHONE_SAMPLES + i] = ptr_int16[i];
    }
    QxOS_UnLockMutex(frame_ptr->mFrameMutex);
#endif

#ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_PROXIMITY
    QxOS_LockMutex(frame_ptr->mFrameMutex);
    ptr_uint8 = (uint8_t*)frame_ptr->mSensorData[mSensorTypeIndex[SENSOR_TYPE_PROXIMITY]].buff_ptr;
    for (int i = 0; i < PRED_NUM_PROXIMITY_SAMPLES; i++) {
        raw_data_array[PROXIMITY_RAW_DATA_OFFSET + PROXIMITY_CHANNEL_OFFSET * PRED_NUM_PROXIMITY_SAMPLES + i] = ptr_uint8[i];
    }
    QxOS_UnLockMutex(frame_ptr->mFrameMutex);
#endif

#ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_AMBIENT
    QxOS_LockMutex(frame_ptr->mFrameMutex);
    ptr_int16 = (int16_t*)frame_ptr->mSensorData[mSensorTypeIndex[SENSOR_TYPE_AMBIENT]].buff_ptr;
    selected_channels = 0;
    for(int axis = 0; axis < 4; axis++) {
        if (ambient_channels[axis] == 1) {
            for (int i = 0; i < PRED_NUM_AMBIENT_SAMPLES; i++) {
                raw_data_array[AMBIENT_RAW_DATA_OFFSET + (selected_channels + AMBIENT_CHANNEL_OFFSET) * PRED_NUM_AMBIENT_SAMPLES + i] = ptr_int16[i * 4 + axis];
            }
            selected_channels += 1;
        }
    }
    QxOS_UnLockMutex(frame_ptr->mFrameMutex);
#endif

#ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_LIGHT
    QxOS_LockMutex(frame_ptr->mFrameMutex);
    ptr_int16 = (int16_t*)frame_ptr->mSensorData[mSensorTypeIndex[SENSOR_TYPE_LIGHT]].buff_ptr;
    for (int i = 0; i < PRED_NUM_LIGHT_SAMPLES; i++) {
        raw_data_array[LIGHT_RAW_DATA_OFFSET + LIGHT_CHANNEL_OFFSET * PRED_NUM_LIGHT_SAMPLES + i] = ptr_int16[i];
    }
    QxOS_UnLockMutex(frame_ptr->mFrameMutex);
#endif

#ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_TVOC
    QxOS_LockMutex(frame_ptr->mFrameMutex);

    ptr_int32 = (int32_t *)frame_ptr->mSensorData[mSensorTypeIndex[SENSOR_TYPE_TVOC]].buff_ptr;
    for (int i = 0; i < PRED_NUM_TVOC_SAMPLES; i++) {
        raw_data_array[TVOC_RAW_DATA_OFFSET + TVOC_CHANNEL_OFFSET * PRED_NUM_TVOC_SAMPLES + i] = ptr_int32[i];
    }
    QxOS_UnLockMutex(frame_ptr->mFrameMutex);
#endif

#ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_ECO2
    QxOS_LockMutex(frame_ptr->mFrameMutex);

    ptr_int32 = (int32_t *)frame_ptr->mSensorData[mSensorTypeIndex[SENSOR_TYPE_ECO2]].buff_ptr;
    for (int i = 0; i < PRED_NUM_ECO2_SAMPLES; i++) {
        raw_data_array[ECO2_RAW_DATA_OFFSET + ECO2_CHANNEL_OFFSET * PRED_NUM_ECO2_SAMPLES + i] = ptr_int32[i];
    }
    QxOS_UnLockMutex(frame_ptr->mFrameMutex);
#endif

#ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_ETOH
    QxOS_LockMutex(frame_ptr->mFrameMutex);

    ptr_int32 = (int32_t *)frame_ptr->mSensorData[mSensorTypeIndex[SENSOR_TYPE_ETOH]].buff_ptr;
    for (int i = 0; i < PRED_NUM_ETOH_SAMPLES; i++) {
        raw_data_array[ETOH_RAW_DATA_OFFSET + ETOH_CHANNEL_OFFSET * PRED_NUM_ETOH_SAMPLES + i] = ptr_int32[i];
    }
    QxOS_UnLockMutex(frame_ptr->mFrameMutex);
#endif

#ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_RCDA
    QxOS_LockMutex(frame_ptr->mFrameMutex);

    ptr_int32 = (int32_t *)frame_ptr->mSensorData[mSensorTypeIndex[SENSOR_TYPE_RCDA]].buff_ptr;
    for (int i = 0; i < PRED_NUM_RCDA_SAMPLES; i++) {
        raw_data_array[RCDA_RAW_DATA_OFFSET + RCDA_CHANNEL_OFFSET * PRED_NUM_RCDA_SAMPLES + i] = ptr_int32[i];
    }
    QxOS_UnLockMutex(frame_ptr->mFrameMutex);
#endif

#ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_IAQ
    QxOS_LockMutex(frame_ptr->mFrameMutex);

    ptr_int32 = (int32_t *)frame_ptr->mSensorData[mSensorTypeIndex[SENSOR_TYPE_IAQ]].buff_ptr;
    for (int i = 0; i < PRED_NUM_IAQ_SAMPLES; i++) {
        raw_data_array[IAQ_RAW_DATA_OFFSET + IAQ_CHANNEL_OFFSET * PRED_NUM_IAQ_SAMPLES + i] = ptr_int32[i];
    }
    QxOS_UnLockMutex(frame_ptr->mFrameMutex);
#endif

#ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_RMOX
    QxOS_LockMutex(frame_ptr->mFrameMutex);

    ptr_uint32 = (uint32_t *)frame_ptr->mSensorData[mSensorTypeIndex[SENSOR_TYPE_RMOX]].buff_ptr;
    selected_channels = 0;
    for(int axis = 0; axis < 13; axis++) {
        if (rmox_channels[axis] == 1) {
            for (int i = 0; i < PRED_NUM_RMOX_SAMPLES; i++) {
                raw_data_array[RMOX_RAW_DATA_OFFSET + (selected_channels + RMOX_CHANNEL_OFFSET) * PRED_NUM_RMOX_SAMPLES + i] = ptr_uint32[i * 13 + axis];
            }
            selected_channels += 1;
        }
    }
    QxOS_UnLockMutex(frame_ptr->mFrameMutex);
#endif

#elif defined MODEL_KIND_CRNN

// Fill the raw data array for CRNN channels-first by time step (see PR #1642 for details).
//   ptr: sensor data pointer
//   pns: predict num samples
//   gnc: group num channels
#define FILL_CHANNELS_FIRST_BY_TIME_STEP(ptr, pns, gnc, channel_offset, num_axes, channels, raw_data_offset) \
    do {                                                                                                     \
        QxOS_LockMutex(frame_ptr->mFrameMutex);                                                              \
        int offset = 0;                                                                                      \
        int num_step_samples = (pns) / RNN_NUM_TIME_STEPS;                                                   \
        for (int t = 0; t <= RNN_NUM_TIME_STEPS; ++t) {                                                      \
            int idx = t * num_step_samples * (gnc);                                                          \
            if (t == RNN_NUM_TIME_STEPS)                                                                     \
                num_step_samples = (pns) % RNN_NUM_TIME_STEPS;                                               \
            idx += num_step_samples * (channel_offset);                                                      \
            for (int axis = 0; axis < (num_axes); ++axis) {                                                  \
                if (!(channels)[axis]) continue;                                                             \
                for (int s = 0; s < num_step_samples; ++s) {                                                 \
                    raw_data_array[(raw_data_offset) + idx++] = (ptr)[(s + offset) * (num_axes) + axis];     \
                }                                                                                            \
            }                                                                                                \
            offset += num_step_samples;                                                                      \
        }                                                                                                    \
        QxOS_UnLockMutex(frame_ptr->mFrameMutex);                                                            \
    } while (0)

#ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_ACCEL
    FILL_CHANNELS_FIRST_BY_TIME_STEP(
        (int16_t*)frame_ptr->mSensorData[mSensorTypeIndex[SENSOR_TYPE_ACCEL]].buff_ptr,
        PRED_NUM_ACCEL_SAMPLES,
        ACCEL_GROUP_NUM_CHANNELS,
        ACCEL_CHANNEL_OFFSET,
        3,
        accel_channels,
        ACCEL_RAW_DATA_OFFSET
    );
#endif

#ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_ACCEL_HIGHSENSITIVE
    FILL_CHANNELS_FIRST_BY_TIME_STEP(
        (int16_t*)frame_ptr->mSensorData[mSensorTypeIndex[SENSOR_TYPE_ACCEL_HIGHSENSITIVE]].buff_ptr,
        PRED_NUM_ACCEL_HIGHSENSITIVE_SAMPLES,
        ACCEL_HIGHSENSITIVE_GROUP_NUM_CHANNELS,
        ACCEL_HIGHSENSITIVE_CHANNEL_OFFSET,
        3,
        accel_channels,
        ACCEL_HIGHSENSITIVE_RAW_DATA_OFFSET
    );
#endif

#ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_ACCEL_LOWPOWER
    FILL_CHANNELS_FIRST_BY_TIME_STEP(
        (int16_t*)frame_ptr->mSensorData[mSensorTypeIndex[SENSOR_TYPE_ACCEL_LOWPOWER]].buff_ptr,
        PRED_NUM_ACCEL_LOWPOWER_SAMPLES,
        ACCEL_LOWPOWER_GROUP_NUM_CHANNELS,
        ACCEL_LOWPOWER_CHANNEL_OFFSET,
        3,
        accel_channels,
        ACCEL_LOWPOWER_RAW_DATA_OFFSET
    );
#endif

#ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_GYRO
    FILL_CHANNELS_FIRST_BY_TIME_STEP(
        (int16_t*)frame_ptr->mSensorData[mSensorTypeIndex[SENSOR_TYPE_GYRO]].buff_ptr,
        PRED_NUM_GYRO_SAMPLES,
        GYRO_GROUP_NUM_CHANNELS,
        GYRO_CHANNEL_OFFSET,
        3,
        gyro_channels,
        GYRO_RAW_DATA_OFFSET
    );
#endif

#ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_MAG
    FILL_CHANNELS_FIRST_BY_TIME_STEP(
        (int16_t*)frame_ptr->mSensorData[mSensorTypeIndex[SENSOR_TYPE_MAG]].buff_ptr,
        PRED_NUM_MAGNO_SAMPLES,
        MAG_GROUP_NUM_CHANNELS,
        MAG_CHANNEL_OFFSET,
        3,
        magno_channels,
        MAG_RAW_DATA_OFFSET
    );
#endif

#ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_TEMPERATURE_EXT1
    FILL_CHANNELS_FIRST_BY_TIME_STEP(
        (int16_t*)frame_ptr->mSensorData[mSensorTypeIndex[SENSOR_TYPE_TEMPERATURE_EXT1]].buff_ptr,
        PRED_NUM_TEMPERATURE_SAMPLES,
        TEMPERATURE_GROUP_NUM_CHANNELS,
        TEMPERATURE_CHANNEL_OFFSET,
        1,
        (int[]){1},
        TEMPERATURE_RAW_DATA_OFFSET
    );
#endif

#ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_TEMPERATURE
    FILL_CHANNELS_FIRST_BY_TIME_STEP(
        (int16_t*)frame_ptr->mSensorData[mSensorTypeIndex[SENSOR_TYPE_TEMPERATURE]].buff_ptr,
        PRED_NUM_TEMPERATURE_SAMPLES,
        TEMPERATURE_GROUP_NUM_CHANNELS,
        TEMPERATURE_CHANNEL_OFFSET,
        1,
        (int[]){1},
        TEMPERATURE_RAW_DATA_OFFSET
    );
#endif

#ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_PRESSURE
    FILL_CHANNELS_FIRST_BY_TIME_STEP(
        (int32_t*)frame_ptr->mSensorData[mSensorTypeIndex[SENSOR_TYPE_PRESSURE]].buff_ptr,
        PRED_NUM_PRESSURE_SAMPLES,
        PRESSURE_GROUP_NUM_CHANNELS,
        PRESSURE_CHANNEL_OFFSET,
        1,
        (int[]){1},
        PRESSURE_RAW_DATA_OFFSET
    );
#endif

#ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_HUMIDITY
    FILL_CHANNELS_FIRST_BY_TIME_STEP(
        (int16_t*)frame_ptr->mSensorData[mSensorTypeIndex[SENSOR_TYPE_HUMIDITY]].buff_ptr,
        PRED_NUM_HUMIDITY_SAMPLES,
        HUMIDITY_GROUP_NUM_CHANNELS,
        HUMIDITY_CHANNEL_OFFSET,
        1,
        (int[]){1},
        HUMIDITY_RAW_DATA_OFFSET
    );
#endif

#ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_MICROPHONE
    FILL_CHANNELS_FIRST_BY_TIME_STEP(
        (int16_t*)frame_ptr->mSensorData[mSensorTypeIndex[SENSOR_TYPE_MICROPHONE]].buff_ptr,
        PRED_NUM_MICROPHONE_SAMPLES,
        MICROPHONE_GROUP_NUM_CHANNELS,
        MICROPHONE_CHANNEL_OFFSET,
        1,
        (int[]){1},
        MICROPHONE_RAW_DATA_OFFSET
    );
#endif

#ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_PROXIMITY
    FILL_CHANNELS_FIRST_BY_TIME_STEP(
        (uint8_t*)frame_ptr->mSensorData[mSensorTypeIndex[SENSOR_TYPE_PROXIMITY]].buff_ptr,
        PRED_NUM_PROXIMITY_SAMPLES,
        PROXIMITY_GROUP_NUM_CHANNELS,
        PROXIMITY_CHANNEL_OFFSET,
        1,
        (int[]){1},
        PROXIMITY_RAW_DATA_OFFSET
    );
#endif

#ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_AMBIENT
    FILL_CHANNELS_FIRST_BY_TIME_STEP(
        (int16_t*)frame_ptr->mSensorData[mSensorTypeIndex[SENSOR_TYPE_AMBIENT]].buff_ptr,
        PRED_NUM_AMBIENT_SAMPLES,
        AMBIENT_GROUP_NUM_CHANNELS,
        AMBIENT_CHANNEL_OFFSET,
        4,
        ambient_channels,
        AMBIENT_RAW_DATA_OFFSET
    );
    QxOS_UnLockMutex(frame_ptr->mFrameMutex);
#endif

#ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_LIGHT
    FILL_CHANNELS_FIRST_BY_TIME_STEP(
        (int16_t*)frame_ptr->mSensorData[mSensorTypeIndex[SENSOR_TYPE_LIGHT]].buff_ptr,
        PRED_NUM_LIGHT_SAMPLES,
        LIGHT_GROUP_NUM_CHANNELS,
        LIGHT_CHANNEL_OFFSET,
        1,
        (int[]){1},
        LIGHT_RAW_DATA_OFFSET
    );
#endif

#ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_TVOC
    FILL_CHANNELS_FIRST_BY_TIME_STEP(
        (int32_t *)frame_ptr->mSensorData[mSensorTypeIndex[SENSOR_TYPE_TVOC]].buff_ptr,
        PRED_NUM_TVOC_SAMPLES,
        TVOC_GROUP_NUM_CHANNELS,
        TVOC_CHANNEL_OFFSET,
        1,
        (int[]){1},
        TVOC_RAW_DATA_OFFSET
    );
#endif

#ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_ECO2
    FILL_CHANNELS_FIRST_BY_TIME_STEP(
        (int32_t *)frame_ptr->mSensorData[mSensorTypeIndex[SENSOR_TYPE_ECO2]].buff_ptr,
        PRED_NUM_ECO2_SAMPLES,
        ECO2_GROUP_NUM_CHANNELS,
        ECO2_CHANNEL_OFFSET,
        1,
        (int[]){1},
        ECO2_RAW_DATA_OFFSET
    );
#endif

#ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_ETOH
    FILL_CHANNELS_FIRST_BY_TIME_STEP(
        (int32_t *)frame_ptr->mSensorData[mSensorTypeIndex[SENSOR_TYPE_ETOH]].buff_ptr,
        PRED_NUM_ETOH_SAMPLES,
        ETOH_GROUP_NUM_CHANNELS,
        ETOH_CHANNEL_OFFSET,
        1,
        (int[]){1},
        ETOH_RAW_DATA_OFFSET
    );
#endif

#ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_RCDA
    FILL_CHANNELS_FIRST_BY_TIME_STEP(
        (int32_t *)frame_ptr->mSensorData[mSensorTypeIndex[SENSOR_TYPE_RCDA]].buff_ptr,
        PRED_NUM_RCDA_SAMPLES,
        RCDA_GROUP_NUM_CHANNELS,
        RCDA_CHANNEL_OFFSET,
        1,
        (int[]){1},
        RCDA_RAW_DATA_OFFSET
    );
#endif

#ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_IAQ
    FILL_CHANNELS_FIRST_BY_TIME_STEP(
        (int32_t *)frame_ptr->mSensorData[mSensorTypeIndex[SENSOR_TYPE_IAQ]].buff_ptr,
        PRED_NUM_IAQ_SAMPLES,
        IAQ_GROUP_NUM_CHANNELS,
        IAQ_CHANNEL_OFFSET,
        1,
        (int[]){1},
        IAQ_RAW_DATA_OFFSET
    );
#endif

#ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_RMOX
    FILL_CHANNELS_FIRST_BY_TIME_STEP(
        (uint32_t *)frame_ptr->mSensorData[mSensorTypeIndex[SENSOR_TYPE_RMOX]].buff_ptr,
        PRED_NUM_RMOX_SAMPLES,
        RMOX_GROUP_NUM_CHANNELS,
        RMOX_CHANNEL_OFFSET,
        13,
        rmox_channels,
        RMOX_RAW_DATA_OFFSET
    );
#endif

#elif defined MODEL_KIND_RNN
    // The data format is channels-last to match the expected input format for pytorch RNN models.
int selected_channels = 0;
#ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_ACCEL
    QxOS_LockMutex(frame_ptr->mFrameMutex);
    ptr_int16 = (int16_t*)frame_ptr->mSensorData[mSensorTypeIndex[SENSOR_TYPE_ACCEL]].buff_ptr;
    selected_channels = 0;
    for(int axis = 0; axis < 3; axis++) {
        if (accel_channels[axis] == 1) {
            for (int i = 0; i < PRED_NUM_ACCEL_SAMPLES; i++) {
                raw_data_array[ACCEL_RAW_DATA_OFFSET + i * ACCEL_GROUP_NUM_CHANNELS + selected_channels + ACCEL_CHANNEL_OFFSET] = ptr_int16[i * 3 + axis];
            }
            selected_channels += 1;
        }
    }
    QxOS_UnLockMutex(frame_ptr->mFrameMutex);
#endif

#ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_ACCEL_HIGHSENSITIVE
    QxOS_LockMutex(frame_ptr->mFrameMutex);
    ptr_int16 = (int16_t*)frame_ptr->mSensorData[mSensorTypeIndex[SENSOR_TYPE_ACCEL_HIGHSENSITIVE]].buff_ptr;
    selected_channels = 0;
    for(int axis = 0; axis < 3; axis++) {
        if (accel_channels[axis] == 1) {
            for (int i = 0; i < PRED_NUM_ACCEL_HIGHSENSITIVE_SAMPLES; i++) {
                raw_data_array[ACCEL_HIGHSENSITIVE_RAW_DATA_OFFSET + i * ACCEL_HIGHSENSITIVE_GROUP_NUM_CHANNELS + selected_channels + ACCEL_HIGHSENSITIVE_CHANNEL_OFFSET] = ptr_int16[i * 3 + axis];
            }
            selected_channels += 1;
        }
    }
    QxOS_UnLockMutex(frame_ptr->mFrameMutex);
#endif

#ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_ACCEL_LOWPOWER
    QxOS_LockMutex(frame_ptr->mFrameMutex);
    ptr_int16 = (int16_t*)frame_ptr->mSensorData[mSensorTypeIndex[SENSOR_TYPE_ACCEL_LOWPOWER]].buff_ptr;
    selected_channels = 0;
    for(int axis = 0; axis < 3; axis++) {
        if (accel_channels[axis] == 1) {
            for (int i = 0; i < PRED_NUM_ACCEL_LOWPOWER_SAMPLES; i++) {
                raw_data_array[ACCEL_LOWPOWER_RAW_DATA_OFFSET + i * ACCEL_LOWPOWER_GROUP_NUM_CHANNELS + selected_channels + ACCEL_LOWPOWER_CHANNEL_OFFSET] = ptr_int16[i * 3 + axis];
            }
            selected_channels += 1;
        }
    }
    QxOS_UnLockMutex(frame_ptr->mFrameMutex);
#endif

#ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_GYRO
    QxOS_LockMutex(frame_ptr->mFrameMutex);
    ptr_int16 = (int16_t*)frame_ptr->mSensorData[mSensorTypeIndex[SENSOR_TYPE_GYRO]].buff_ptr;
    selected_channels = 0;
    for(int axis = 0; axis < 3; axis++) {
        if (gyro_channels[axis] == 1) {
            for (int i = 0; i < PRED_NUM_GYRO_SAMPLES; i++) {
                raw_data_array[GYRO_RAW_DATA_OFFSET + i * GYRO_GROUP_NUM_CHANNELS + selected_channels + GYRO_CHANNEL_OFFSET] = ptr_int16[i * 3 + axis];
            }
            selected_channels += 1;
        }
    }
    QxOS_UnLockMutex(frame_ptr->mFrameMutex);
#endif

#ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_MAG
    QxOS_LockMutex(frame_ptr->mFrameMutex);
    ptr_int16 = (int16_t*)frame_ptr->mSensorData[mSensorTypeIndex[SENSOR_TYPE_MAG]].buff_ptr;
    selected_channels = 0;
    for(int axis = 0; axis < 3; axis++) {
        if (magno_channels[axis] == 1) {
            for (int i = 0; i < PRED_NUM_MAGNO_SAMPLES; i++) {
                raw_data_array[MAG_RAW_DATA_OFFSET + i * MAG_GROUP_NUM_CHANNELS + selected_channels + MAG_CHANNEL_OFFSET] = ptr_int16[i * 3 + axis];
            }
            selected_channels += 1;
        }
    }
    QxOS_UnLockMutex(frame_ptr->mFrameMutex);
#endif

#ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_TEMPERATURE_EXT1
    QxOS_LockMutex(frame_ptr->mFrameMutex);
    ptr_int16 = (int16_t*)frame_ptr->mSensorData[mSensorTypeIndex[SENSOR_TYPE_TEMPERATURE_EXT1]].buff_ptr;
    for (int i = 0; i < PRED_NUM_TEMPERATURE_SAMPLES; i++) {
        raw_data_array[TEMPERATURE_RAW_DATA_OFFSET + i * TEMPERATURE_GROUP_NUM_CHANNELS + TEMPERATURE_CHANNEL_OFFSET] = ptr_int16[i];
    }
    QxOS_UnLockMutex(frame_ptr->mFrameMutex);
#endif

#ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_TEMPERATURE
    QxOS_LockMutex(frame_ptr->mFrameMutex);
    ptr_int16 = (int16_t*)frame_ptr->mSensorData[mSensorTypeIndex[SENSOR_TYPE_TEMPERATURE]].buff_ptr;
    for (int i = 0; i < PRED_NUM_TEMPERATURE_SAMPLES; i++) {
        raw_data_array[TEMPERATURE_RAW_DATA_OFFSET + i * TEMPERATURE_GROUP_NUM_CHANNELS + TEMPERATURE_CHANNEL_OFFSET] = ptr_int16[i];
    }
    QxOS_UnLockMutex(frame_ptr->mFrameMutex);
#endif

#ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_PRESSURE
    QxOS_LockMutex(frame_ptr->mFrameMutex);
    ptr_int32 = (int32_t*)frame_ptr->mSensorData[mSensorTypeIndex[SENSOR_TYPE_PRESSURE]].buff_ptr;;
    for (int i = 0; i < PRED_NUM_PRESSURE_SAMPLES; i++) {
        raw_data_array[PRESSURE_RAW_DATA_OFFSET + i * PRESSURE_GROUP_NUM_CHANNELS + PRESSURE_CHANNEL_OFFSET] = ptr_int32[i];
    }
    QxOS_UnLockMutex(frame_ptr->mFrameMutex);
#endif

#ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_HUMIDITY
    QxOS_LockMutex(frame_ptr->mFrameMutex);
    ptr_int16 = (int16_t*)frame_ptr->mSensorData[mSensorTypeIndex[SENSOR_TYPE_HUMIDITY]].buff_ptr;
    for (int i = 0; i < PRED_NUM_HUMIDITY_SAMPLES; i++) {
        raw_data_array[HUMIDITY_RAW_DATA_OFFSET + i * HUMIDITY_GROUP_NUM_CHANNELS + HUMIDITY_CHANNEL_OFFSET] = ptr_int16[i];
    }
    QxOS_UnLockMutex(frame_ptr->mFrameMutex);
#endif

#ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_MICROPHONE
    QxOS_LockMutex(frame_ptr->mFrameMutex);
    ptr_int16 = (int16_t*)frame_ptr->mSensorData[mSensorTypeIndex[SENSOR_TYPE_MICROPHONE]].buff_ptr;
    for (int i = 0; i < PRED_NUM_MICROPHONE_SAMPLES; i++) {
        raw_data_array[MICROPHONE_RAW_DATA_OFFSET + i * MICROPHONE_GROUP_NUM_CHANNELS + MICROPHONE_CHANNEL_OFFSET] = ptr_int16[i];
    }
    QxOS_UnLockMutex(frame_ptr->mFrameMutex);
#endif

#ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_PROXIMITY
    QxOS_LockMutex(frame_ptr->mFrameMutex);
    ptr_uint8 = (uint8_t*)frame_ptr->mSensorData[mSensorTypeIndex[SENSOR_TYPE_PROXIMITY]].buff_ptr;
    for (int i = 0; i < PRED_NUM_PROXIMITY_SAMPLES; i++) {
        raw_data_array[PROXIMITY_RAW_DATA_OFFSET + i * PROXIMITY_GROUP_NUM_CHANNELS + PROXIMITY_CHANNEL_OFFSET] = ptr_uint8[i];
    }
    QxOS_UnLockMutex(frame_ptr->mFrameMutex);
#endif

#ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_AMBIENT
    QxOS_LockMutex(frame_ptr->mFrameMutex);
    ptr_int16 = (int16_t*)frame_ptr->mSensorData[mSensorTypeIndex[SENSOR_TYPE_AMBIENT]].buff_ptr;
    selected_channels = 0;
    for(int axis = 0; axis < 4; axis++) {
        if (ambient_channels[axis] == 1) {
            for (int i = 0; i < PRED_NUM_AMBIENT_SAMPLES; i++) {
                raw_data_array[AMBIENT_RAW_DATA_OFFSET + i * AMBIENT_GROUP_NUM_CHANNELS + selected_channels + AMBIENT_CHANNEL_OFFSET] = ptr_int16[i * 4 + axis];
            }
            selected_channels += 1;
        }
    }
    QxOS_UnLockMutex(frame_ptr->mFrameMutex);
#endif

#ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_LIGHT
    QxOS_LockMutex(frame_ptr->mFrameMutex);
    ptr_int16 = (int16_t*)frame_ptr->mSensorData[mSensorTypeIndex[SENSOR_TYPE_LIGHT]].buff_ptr;
    for (int i = 0; i < PRED_NUM_LIGHT_SAMPLES; i++) {
        raw_data_array[LIGHT_RAW_DATA_OFFSET + i * LIGHT_GROUP_NUM_CHANNELS + LIGHT_CHANNEL_OFFSET] = ptr_int16[i];
    }
    QxOS_UnLockMutex(frame_ptr->mFrameMutex);
#endif

#ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_TVOC
    QxOS_LockMutex(frame_ptr->mFrameMutex);

    ptr_int32 = (int32_t *)frame_ptr->mSensorData[mSensorTypeIndex[SENSOR_TYPE_TVOC]].buff_ptr;
    for (int i = 0; i < PRED_NUM_TVOC_SAMPLES; i++) {
        raw_data_array[TVOC_RAW_DATA_OFFSET + i * TVOC_GROUP_NUM_CHANNELS + TVOC_CHANNEL_OFFSET] = ptr_int32[i];
    }
    QxOS_UnLockMutex(frame_ptr->mFrameMutex);
#endif

#ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_ECO2
    QxOS_LockMutex(frame_ptr->mFrameMutex);

    ptr_int32 = (int32_t *)frame_ptr->mSensorData[mSensorTypeIndex[SENSOR_TYPE_ECO2]].buff_ptr;
    for (int i = 0; i < PRED_NUM_ECO2_SAMPLES; i++) {
        raw_data_array[ECO2_RAW_DATA_OFFSET + i * ECO2_GROUP_NUM_CHANNELS + ECO2_CHANNEL_OFFSET] = ptr_int32[i];
    }
    QxOS_UnLockMutex(frame_ptr->mFrameMutex);
#endif

#ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_ETOH
    QxOS_LockMutex(frame_ptr->mFrameMutex);

    ptr_int32 = (int32_t *)frame_ptr->mSensorData[mSensorTypeIndex[SENSOR_TYPE_ETOH]].buff_ptr;
    for (int i = 0; i < PRED_NUM_ETOH_SAMPLES; i++) {
        raw_data_array[ETOH_RAW_DATA_OFFSET + i * ETOH_GROUP_NUM_CHANNELS + ETOH_CHANNEL_OFFSET] = ptr_int32[i];
    }
    QxOS_UnLockMutex(frame_ptr->mFrameMutex);
#endif

#ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_RCDA
    QxOS_LockMutex(frame_ptr->mFrameMutex);

    ptr_int32 = (int32_t *)frame_ptr->mSensorData[mSensorTypeIndex[SENSOR_TYPE_RCDA]].buff_ptr;
    for (int i = 0; i < PRED_NUM_RCDA_SAMPLES; i++) {
        raw_data_array[RCDA_RAW_DATA_OFFSET + i * RCDA_GROUP_NUM_CHANNELS + RCDA_CHANNEL_OFFSET] = ptr_int32[i];
    }
    QxOS_UnLockMutex(frame_ptr->mFrameMutex);
#endif

#ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_IAQ
    QxOS_LockMutex(frame_ptr->mFrameMutex);

    ptr_int32 = (int32_t *)frame_ptr->mSensorData[mSensorTypeIndex[SENSOR_TYPE_IAQ]].buff_ptr;
    for (int i = 0; i < PRED_NUM_IAQ_SAMPLES; i++) {
        raw_data_array[IAQ_RAW_DATA_OFFSET + i * IAQ_GROUP_NUM_CHANNELS + IAQ_CHANNEL_OFFSET] = ptr_int32[i];
    }
    QxOS_UnLockMutex(frame_ptr->mFrameMutex);
#endif

#ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_RMOX
    QxOS_LockMutex(frame_ptr->mFrameMutex);

    ptr_uint32 = (uint32_t *)frame_ptr->mSensorData[mSensorTypeIndex[SENSOR_TYPE_RMOX]].buff_ptr;
    selected_channels = 0;
    for(int axis = 0; axis < 13; axis++) {
        if (rmox_channels[axis] == 1) {
            for (int i = 0; i < PRED_NUM_RMOX_SAMPLES; i++) {
                raw_data_array[RMOX_RAW_DATA_OFFSET + i * RMOX_GROUP_NUM_CHANNELS + selected_channels + RMOX_CHANNEL_OFFSET] = ptr_uint32[i * 13 + axis];
            }
            selected_channels += 1;
        }
    }
    QxOS_UnLockMutex(frame_ptr->mFrameMutex);
#endif

#else
#error Unsupported model kind
#endif

}

bool QXO_MLEngine_SetSensitivity(float * pSensitivity) {
    //Update sensitivity parameters
    if(pSensitivity != NULL)
    {
        float * p = (float *)pSensitivity;
        for(int i = 0; i < NUM_CLASSES; i++) {
            mSensitivity[i] = *p++;
        }
        return TRUE;
    }
    return FALSE;
}

void QXO_MLEngine_GetSensitivity(
                float * pSensitivity,
                int size_of_pSensitivity,
                int * pNumOfClasses) {

    if(pSensitivity != NULL && pNumOfClasses != NULL)
    {
        *pNumOfClasses = NUM_CLASSES;

        if ( size_of_pSensitivity >= sizeof(mSensitivity)) {
            memcpy(pSensitivity, &mSensitivity[0], sizeof(mSensitivity) );
        } else {
        // TODO: report size error.
        }
    }
}

int QXO_MLEngine_GetPredictionInterval(void) {

    return PRED_CLASSIFICATION_INTERVAL_IN_MSECS;
}

#if defined (ONE_CLASS_MODEL) || defined(HYBRID_MODEL)
void QXO_MLEngine_SetCtmThreshold(float ctmThreshold) {
//update comtamination threshold

    Contamination_threshold = ctmThreshold;
}

void QXO_MLEngine_GetCtmThreshold(float * pCtmThreshold){
    if(pCtmThreshold != NULL)
    {
        *pCtmThreshold = Contamination_threshold;
    }
}
#endif

int QXO_MLEngine_Work(pPredictionFrame frame, int mEvClsStatus) {

//Invalid predicton frame
    if(frame == NULL)
    {
        return -1;
    }
    int cls = 0;
#ifndef RAW_MODEL
    init_feats(&features);
#endif
//Extract prediction framle to seperate sensor buffers
    ExtractPredictionFrame(frame);
    memset(frame->mProbs, 0, sizeof(frame->mProbs));

    // pred.probs stores the accumulative probas during event classification
    // it needs to set to zeros when changing from non_start status to start
    // non_start status NONE(continuous classification), stop(stop of last event classification),
    // wake and sleep(transition)

    if (pred.event_classification_status != 1 && mEvClsStatus == 1){
        for (int c = 0; c < NUM_CLASSES; c++) {
            pred.probs[c] = 0;
        }
    }
    pred.event_classification_status = mEvClsStatus;

#ifndef RAW_MODEL
    predict(features.feats, &pred);
#else
    predict(&pred, raw_data_array);
#endif

#if !defined (ONE_CLASS_MODEL) && !defined(HYBRID_MODEL)
#ifdef ON_DEVICE_MULTI_CLASS_MODEL
    if (pred.training_state.start_time_left > 0){
        QxOS_ClassifyPrint("Starting Collection in %d\n", pred.training_state.start_time_left);
    }
    else {
        if (pred.training_state.collect_data == 1){
            QxOS_ClassifyPrint("Collected Inst: %d ", pred.training_state.samples_collected);
            QxOS_ClassifyPrint("of Class: %d\n", pred.training_state.class_being_collected);
        }
        else {
            if (pred.training_state.in_between_class_time_left > 1){
                QxOS_ClassifyPrint("Starting Collection in %d\n", pred.training_state.in_between_class_time_left - 1);
            }
            if (pred.training_state.train_wait_left > 0){
                QxOS_ClassifyPrint("Training ... \n");
            }
        }
    }
#endif
#ifdef MCU_MODEL
    apply_sensitivity(&pred, mSensitivity);
#else
    // prod-> probas store the accumulative probas when mEvClsStatus = start(1)
    // only apply to continuous classification(0) or at the stop(2)
    if (mEvClsStatus != 1){
        apply_sensitivity(&pred, mSensitivity);
    }
#endif

#ifdef MCU_MODEL
    for (int c = 1; c < NUM_CLASSES - 1; c++) {
            if (pred.probs[c] > pred.probs[cls]) {
                cls = c;
        }
    }
    predict_mcu_oc(features.feats, &pred, cls);
#endif

#else

#ifdef ON_DEVICE_ONE_CLASS_MODEL
    if (pred.training_state.start_time_left > 0){
        QxOS_DebugPrint("Starting Collection in %d\n", pred.training_state.start_time_left);
        frame->mOnDeviceModelStatus = ONDEVICE_INIT;
    }
    else {
        if (pred.training_state.collect_data == 1){
            QxOS_DebugPrint("Collected Inst: %d\n", pred.training_state.samples_collected);
            frame->mOnDeviceModelStatus = ONDEVICE_DC;
        }
        else {
            if (pred.training_state.train_wait_left == 0){
                QxOS_DebugPrint("SCORE: %.2f\n", pred.probs[0]);
            }
            else{
                QxOS_DebugPrint("Training ... \n");
            }
            frame->mOnDeviceModelStatus = ONDEVICE_TRAINING;
        }
    }
#else
    #ifdef ONE_CLASS_MODEL
        QxOS_DebugPrint("SCORE: %.2f\n", pred.probs[0]);
    #endif

    #ifdef HYBRID_MODEL
        QxOS_DebugPrint("SCORE: %.2f\n", 1 - pred.probs[0]);
    #endif
#endif
    // prod-> probas store the accumulative probas when mEvClsStatus = start(1)
    // only apply to continuous classification(0) or at the stop(2)
    if (mEvClsStatus != 1){
        apply_Contamination_threshold(&pred, 0);
    }
#endif

#ifndef RAW_MODEL
    // don't use gaussian weight for event classification
    /* apply state update using gaussian weights */
    if (mEvClsStatus == 0){
        cls = State_update(pred.probs, event_classes);
    }
    else{
        for (int c = 0; c < NUM_CLASSES; c++) {
            if (pred.probs[c] > pred.probs[cls]) {
                cls = c;
            }
        }
    }
#else
    /* find highest probability class */
    for (int c = 0; c < NUM_CLASSES; c++) {
        if (pred.probs[c] > pred.probs[cls]) {
            cls = c;
        }
    }
#endif
    memcpy(frame->mProbs, pred.probs, sizeof(pred.probs));
    frame->mEventClassificationStatus = pred.event_classification_status;

#if defined(ON_DEVICE_ONE_CLASS_MODEL) || defined(ON_DEVICE_MULTI_CLASS_MODEL)
    if (pred.training_state.start_inference == 1)
        frame->mOnDeviceModelStatus = ONDEVICE_CLASSIFY;
#endif
    return cls;
}


MLEngineStatus_t QXO_MLEngine_DeInit(pPredictionFrame frame) {
    if(frame != NULL) {
        frame = NULL;
    }
    return MLENGINE_OK;
}

#undef QXO_GLOBALS

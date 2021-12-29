#pragma once

#ifndef MIDDLEWARES_QEEXO_INCLUDE_APPS_QXAUTOMLCONFIG_H_
#define MIDDLEWARES_QEEXO_INCLUDE_APPS_QXAUTOMLCONFIG_H_

#include "QxAutoMLUser.h" 
/* Configuration for Qeexo middleware components. */

/*** general configuration ***/
/* Define to enable RTOS integration */
#define QXO_ENABLE_RTOS
/* Define to enable debug logging */
#define QXO_ENABLE_LOGGING

/* Configure logging buffer size */
#define QXO_LOGGING_SIZE 5120

/*** features configuration ***/
/* Define to enable logging of feature computation times, at the cost of 4 bytes of RAM per feature */
// #define QXO_FEATURES_ENABLE_PROFILING

/*** fftpack configuration ***/
/* Define to enable support for non-power-of-two FFTs, at the cost of significantly more code (~10KB). */
// #define QXO_FFT_ENABLE_NONPOW2

/* Define to use doubles for FFT computation. It is **strongly** recommended that you do not use this option,
   as it will massively increase computation time on Cortex-M4F. */
// #define QXO_FFT_USE_DOUBLE

/* Define to enable logging to the USB Serial console */
#define QXO_DATALOG_ENABLE_USB

#define QXO_DATALOG_DEBUGLOG_BUFSIZE          2048

/*** qxo_gpio configuration ***/
#define QXO_GPIO_PWM_FREQDIVIDER              640  /* PWM frequency = SYSCLK / FREQDIVIDER / 255; default = ~490Hz */

/************************************************
 * RAM MODEL DEFINITIONS
 ************************************************/
// #define RAW_MODEL


/*** define features based on sensor configurations ***/
/* Define if acce/gyro, magnetometer and low frequency features were added during featurization */
#if defined(QXAUTOMLCONFIG_SENSOR_ENABLE_ACCEL) || defined(QXAUTOMLCONFIG_SENSOR_ENABLE_ACCEL_HIGHSENSITIVE) || defined(QXAUTOMLCONFIG_SENSOR_ENABLE_ACCEL_LOWPOWER)
#define ACCEL_FEATS
#endif
#if defined(QXAUTOMLCONFIG_SENSOR_ENABLE_GYRO)
#define GYRO_FEATS
#endif
#ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_MICROPHONE
#define MICROPHONE_FEATS
#endif
#ifdef QXAUTOMLCONFIG_SENSOR_ENABLE_MAG
#define MAGNO_FEATS
#endif
#if defined(QXAUTOMLCONFIG_SENSOR_ENABLE_PRESSURE) || defined(QXAUTOMLCONFIG_SENSOR_ENABLE_TEMPERATURE) || defined(QXAUTOMLCONFIG_SENSOR_ENABLE_TEMPERATURE_EXT1) || defined(QXAUTOMLCONFIG_SENSOR_ENABLE_HUMIDITY) || defined(QXAUTOMLCONFIG_SENSOR_ENABLE_PROXIMITY) || defined(QXAUTOMLCONFIG_SENSOR_ENABLE_AMBIENT) || defined(QXAUTOMLCONFIG_SENSOR_ENABLE_LIGHT)
#define LOWFREQ_FEATS
#endif


#ifndef RAW_MODEL
/* Define which feature(s) to use */
// Make thse configurable
 #define STAT_FEATURES
// #define AUTOCORR_FEATURES
// #define FFT_FEATURES
// #define PEAK_FEATURES


#ifdef ACCEL_FEATS
 #define ACCEL_STAT_FEATURES
// #define ACCEL_AUTOCORR_FEATURES
// #define ACCEL_FFT_FEATURES_SIMPLE
// #define ACCEL_FFT_FEATURES_ADVANCED
// #define ACCEL_FFT_FEATURES_LINEARLY_BINNED
// #define ACCEL_FFT_FEATURES_OCTAVE_BINNED
// #define ACCEL_FFT_FEATURES_THIRDS_BINNED
// #define ACCEL_PEAK_FEATURES

// #define ACCEL_LOWPOWER_STAT_FEATURES
// #define ACCEL_LOWPOWER_AUTOCORR_FEATURES
// #define ACCEL_LOWPOWER_FFT_FEATURES_SIMPLE
// #define ACCEL_LOWPOWER_FFT_FEATURES_ADVANCED
// #define ACCEL_LOWPOWER_FFT_FEATURES_LINEARLY_BINNED
// #define ACCEL_LOWPOWER_FFT_FEATURES_OCTAVE_BINNED
// #define ACCEL_LOWPOWER_FFT_FEATURES_THIRDS_BINNED
// #define ACCEL_LOWPOWER_PEAK_FEATURES

// #define ACCEL_HIGHSENSITIVE_STAT_FEATURES
// #define ACCEL_HIGHSENSITIVE_AUTOCORR_FEATURES
// #define ACCEL_HIGHSENSITIVE_FFT_FEATURES_SIMPLE
// #define ACCEL_HIGHSENSITIVE_FFT_FEATURES_ADVANCED
// #define ACCEL_HIGHSENSITIVE_FFT_FEATURES_LINEARLY_BINNED
// #define ACCEL_HIGHSENSITIVE_FFT_FEATURES_OCTAVE_BINNED
// #define ACCEL_HIGHSENSITIVE_FFT_FEATURES_THIRDS_BINNED
// #define ACCEL_HIGHSENSITIVE_PEAK_FEATURES
#endif

#ifdef GYRO_FEATS
 #define GYRO_STAT_FEATURES
// #define GYRO_AUTOCORR_FEATURES
// #define GYRO_FFT_FEATURES_SIMPLE
// #define GYRO_FFT_FEATURES_ADVANCED
// #define GYRO_FFT_FEATURES_LINEARLY_BINNED
// #define GYRO_FFT_FEATURES_OCTAVE_BINNED
// #define GYRO_FFT_FEATURES_THIRDS_BINNED
// #define GYRO_PEAK_FEATURES
#endif

#ifdef MICROPHONE_FEATS
// #define MICROPHONE_STAT_FEATURES
// #define MICROPHONE_AUTOCORR_FEATURES
// #define MICROPHONE_FFT_FEATURES_SIMPLE
// #define MICROPHONE_FFT_FEATURES_ADVANCED
// #define MICROPHONE_FFT_FEATURES_LINEARLY_BINNED
// #define MICROPHONE_FFT_FEATURES_OCTAVE_BINNED
// #define MICROPHONE_FFT_FEATURES_THIRDS_BINNED
// #define MICROPHONE_PEAK_FEATURES
#define MICROPHONE_MFCC_FEATURES_FILTER
//#define MICROPHONE_MFCC_FEATURES_DELTA
//#define MICROPHONE_MFCC_FEATURES_DELTADELTA
// #define MFCC_FEATURES
#endif

#ifdef MAGNO_FEATS
 #define MAGNO_STAT_FEATURES
 #define MAGNO_AUTOCORR_FEATURES
 #define MAGNO_PEAK_FEATURES
#endif


#endif // RAW_MODEL Indicator


#define PRED_CLASSIFICATION_INTERVAL_IN_MSECS 100

#endif /* MIDDLEWARES_QEEXO_INCLUDE_APPS_QXAUTOMLCONFIG_H_ */

#pragma once

#ifndef QEEXO_AUTOML_SENSOR_CONFIGS_H_
#define QEEXO_AUTOML_SENSOR_CONFIGS_H_

#include "QxAutoMLUser.h"

 char accel_buffer[20*6];
  char gyro_buffer[20*6];
// char magno_buffer[30*6];
// char pressure_buffer[30*4];
// char temp_buffer[30*2];
// char hum_buffer[30*2];
// char mic_buffer[800*2];
// char accel_lowpower_buffer[30*6];
// char accel_highsen_buffer[7200*6];
// char temp_ext1_buffer[30*2];
// char proximity_buffer[30*1];
// char ambient_buffer[30*8];
// char light_buffer[30*2];
// char tvoc_buffer[10*4];
// char eco2_buffer[10*4];
// char etoh_buffer[10*4];
// char rcda_buffer[10*4];
// char iaq_buffer[10*4];
// char rmox_buffer[10*4];

tQxAutoMLSensor enabled_sensors[] = {
    {
        SENSOR_TYPE_ACCEL,
        16.0f,
        200.0f,
        20,
        6,
 	   accel_buffer,
    },

    {
        SENSOR_TYPE_GYRO,
        2000.0f,
        200.0f,
        20,
        6,
          gyro_buffer,
    },

//    {
//        SENSOR_TYPE_MAG,
//        50.0f,
//        100.0f,
//        30,
//        6,
//		   magno_buffer,
//    },

//    {
//        SENSOR_TYPE_PRESSURE,
//        50.0f,
//        100.0f,
//        30,
//        4,
//		  pressure_buffer,
//    },

//    {
//        SENSOR_TYPE_TEMPERATURE,
//        0.0f,
//        0.0,
//        30,
//        2,
//        temp_buffer,
//    },

//    {
//        SENSOR_TYPE_HUMIDITY,
//        0.0f,
//        0.0,
//        30,
//        2,
//        hum_buffer,
//    },

//    {
//        SENSOR_TYPE_MICROPHONE,
//        0.0f,
//        16000.0f ,
//        800,
//        2,
//        mic_buffer,
//    },

//    {
//        SENSOR_TYPE_ACCEL_LOWPOWER,
//        0.0f,
//        0.0f,
//        30,
//        6,
//        accel_lowpower_buffer,
//    },

//    {
//        SENSOR_TYPE_ACCEL_HIGHSENSITIVE,
//        0.0f,
//        26667.0f,
//        7200,
//        6,
//        accel_highsen_buffer,
//    },

//    {
//        SENSOR_TYPE_TEMPERATURE_EXT1,
//        0.0f,
//        100.0f ,
//        30,
//        2,
//          temp_ext1_buffer,
//    },

//    {
//        SENSOR_TYPE_PROXIMITY,
//        0.0f,
//        100.0f,
//        30,
//        1,
//        proximity_buffer,
//    },

//    {
//        SENSOR_TYPE_AMBIENT,
//        0.0f,
//        100.0f,
//        30,
//        8,
//        ambient_buffer,
//    },

//    {
//        SENSOR_TYPE_LIGHT,
//        0.0f,
//        10.0f ,
//        30,
//        2,
//        light_buffer,
//    },

//    {
//        SENSOR_TYPE_TVOC,
//        0.0f,
//        0.5f,
//        10,
//        4,
//        tvoc_buffer,
//    },

//    {
//        SENSOR_TYPE_ECO2,
//        0.0f,
//        0.5f,
//        10,
//        4,
//        eco2_buffer,
//    },

//    {
//        SENSOR_TYPE_ETOH,
//        0.0f,
//        0.5f,
//        10,
//        4,
//        etoh_buffer,
//    },

//    {
//        SENSOR_TYPE_RCDA,
//        0.0f,
//        0.5f,
//        10,
//        4,
//        rcda_buffer,
//    },

//    {
//        SENSOR_TYPE_IAQ,
//         0.0f,
//        0.5f,
//        10,
//        4,
//        iaq_buffer,
//    },

//    {
//        SENSOR_TYPE_RMOX,
//        0.0f,
//        0.5f,
//        10,
//        4,
//        rmox_buffer,
//    },
};

int enabled_sensors_count = sizeof(enabled_sensors)/sizeof(enabled_sensors[0]);
#endif
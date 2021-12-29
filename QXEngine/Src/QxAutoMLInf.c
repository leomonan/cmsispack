/**
  ******************************************************************************
  * @file    QxAutoMLInf.cpp
  * @author  Qeexo Kernel Development team
  * @version V1.0.0
  * @date    30-Sep-2020
  * @brief   Auto ML module for Inference 
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2020 Qeexo Co.
  * All rights reserved
  *
  *
  * ALL INFORMATION CONTAINED HEREIN IS AND REMAINS THE PROPERTY OF QEEXO, CO.
  * THE INTELLECTUAL AND TECHNICAL CONCEPTS CONTAINED HEREIN ARE PROPRIETARY TO
  * QEEXO, CO. AND MAY BE COVERED BY U.S. AND FOREIGN PATENTS, PATENTS IN PROCESS,
  * AND ARE PROTECTED BY TRADE SECRET OR COPYRIGHT LAW. DISSEMINATION OF
  * THIS INFORMATION OR REPRODUCTION OF THIS MATERIAL IS STRICTLY FORBIDDEN UNLESS
  * PRIOR WRITTEN PERMISSION IS OBTAINED OR IS MADE PURSUANT TO A LICENSE AGREEMENT
  * WITH QEEXO, CO. ALLOWING SUCH DISSEMINATION OR REPRODUCTION.
  *
  ******************************************************************************
 */

#include "QxClassifyEngine.h"
#include "QxAutoMLConfig.h"

extern void NativeOSPrint(char* str);

//#define DEBUG_ON
#ifndef DEBUG_ON
#define NativeOSPrint
#endif

static bool mEngineInited = false;
static pPredictionFrame mPred;
static int mNumOfClasses;
static float mEngineSensitivity[50];
static int mPredictionInterval;

/*
    This funtion initialize all requeirements for classification
*/
static void QxCheckAndInitEngineOnce()
{
    if (mEngineInited) {
        return;
    }
        
    //Serial.println("Ready to initialize MLEngine!!");
    /* Initialize Engine. Allocate memory for buffering sensor data for prediction purpose */
    mPred = QXO_MLEngine_Init();
    mPredictionInterval = QXO_MLEngine_GetPredictionInterval();

    memset(mEngineSensitivity, 0.0, sizeof(mEngineSensitivity));
    QXO_MLEngine_GetSensitivity(&mEngineSensitivity[0], sizeof(mEngineSensitivity),  &mNumOfClasses);

    /* In the following section, we assign the prediction sensor data buffer pointers to
        each private pointer variables, then we can feed the sensor data separately.  */
    if(mPred != NULL) {
        for(int i = 0; i < mPred->mEnabledSensorCount; i++) {
            if(mPred->mSensorData[i].sensor_type == SENSOR_TYPE_ACCEL) {
                NativeOSPrint("use Accel Data.");    
            }else if(mPred->mSensorData[i].sensor_type == SENSOR_TYPE_GYRO) {
                NativeOSPrint("use Gyro Data.");
            }else if(mPred->mSensorData[i].sensor_type == SENSOR_TYPE_MAG) {
                NativeOSPrint("use mMagData");
            }else if(mPred->mSensorData[i].sensor_type == SENSOR_TYPE_PRESSURE) {
                NativeOSPrint("use mPressData");
            }else if(mPred->mSensorData[i].sensor_type == SENSOR_TYPE_TEMPERATURE) {
                NativeOSPrint("use mTempData");
            }else if(mPred->mSensorData[i].sensor_type == SENSOR_TYPE_HUMIDITY) {
                NativeOSPrint("use mHumidityData");
            }else if(mPred->mSensorData[i].sensor_type == SENSOR_TYPE_PROXIMITY) {
                NativeOSPrint("use mProximityData");
            }else if(mPred->mSensorData[i].sensor_type == SENSOR_TYPE_AMBIENT) {
                NativeOSPrint("use mLightData");
            }else if(mPred->mSensorData[i].sensor_type == SENSOR_TYPE_MICROPHONE) {
                NativeOSPrint("use mPCMData");
            }
        }

        mEngineInited = true;
        
        NativeOSPrint("MLEngine is ready! Classify intervcal:");
        char buff[16];
        sprintf(buff, "%d", mPredictionInterval);
        NativeOSPrint(buff);
    } else {
        NativeOSPrint("MLEngine init error!!");
    }
}

static void CopyDataToFrameBuffer(void* data, SensorData *sensor, int data_len)
{
    if (data_len > sensor->buff_max) {
        data_len = sensor->buff_max;
    }

    /* Use weak attribute to ignore muxtex in case of single thread classify mode */
    QxOS_LockMutex(mPred->mFrameMutex);

    int delta = sensor->buff_max - (data_len + sensor->buff_end);
    if(delta >= 0) {
        memcpy((sensor->buff_ptr+sensor->buff_end), (uint8_t*)data, data_len );
        sensor->buff_end += data_len;
    } else {
        //remove oldest data to make a room for new data
        memmove(sensor->buff_ptr, sensor->buff_ptr+abs(delta), sensor->buff_max + delta );
        memcpy((sensor->buff_ptr+sensor->buff_end + delta), (uint8_t*)data, data_len );
        sensor->buff_end += (data_len + delta);
    }

    QxOS_UnLockMutex(mPred->mFrameMutex);
}

void QxFillSensorData(QXOSensorType type, void* data, int data_len)
{
    SensorData * sensor = NULL;
    
    QxCheckAndInitEngineOnce();
        
    for(int i = 0; i < mPred->mEnabledSensorCount; i++) {
        if(mPred->mSensorData[i].sensor_type == type) {
            sensor = &mPred->mSensorData[i];
            if (sensor) {
                CopyDataToFrameBuffer(data, sensor, data_len);
            }
        }
    }
}

int QxClassify()
{
    /* Call classification prediction, the input sensor data in 'mPred' is feeding
        in another thread that created by QxAutoMLInf::CheckAndInitEngine() */
    int cls = 0;

    QxCheckAndInitEngineOnce();
    
    cls = QXO_MLEngine_Work(mPred, 0);

#ifdef DEBUG_ON    
    const uint8_t buffsize = 125;
    char classifylogBuffer[buffsize];
    int offset = 0;

    offset = sprintf(classifylogBuffer, "PRED: %d", cls);

    for (int i = 0; i < mNumOfClasses; i++) {
        offset += sprintf(classifylogBuffer + offset, ", %.2f", mPred->mProbs[i]);
    }

    NativeOSPrint(classifylogBuffer);
#endif

    return cls;
}


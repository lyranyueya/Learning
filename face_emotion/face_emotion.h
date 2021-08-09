/****************************************************************************
*   amlogic nn api util header file
*
*   Neural Network appliction network definition some util header file
*
*   Date: 2019.8
***************************************************************************/
#ifndef _AMLOGIC_NN_SDK_POSTPROCESS_H
#define _AMLOGIC_NN_SDK_POSTPROCESS_H
#include "nn_sdk.h"
#include "nn_util.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct detection_{
    box bbox;
    float *prob;
    float objectness;
    int classes;
    int sort_class;
}detection;
typedef struct __nn_face_emotion
{
	int emotion;
    float prob;
}face_emotion_out_t;

void* postprocess_emotion(nn_output *pout);

unsigned char *get_jpeg_rawData(const char *name,unsigned int width,unsigned int height);
float Float16ToFloat32(const signed short* src , float* dst ,int lenth);
float *dtype_To_F32(nn_output * outdata ,int sz);
static int _jpeg_to_bmp
    (
    FILE * inputFile,
    unsigned char* bmpData,
    unsigned int bmpWidth,
    unsigned int bmpHeight,
    unsigned int channel
    );


#ifdef __cplusplus
} //extern "C"
#endif
#endif
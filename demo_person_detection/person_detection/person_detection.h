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

typedef struct __nn_person_detect
{
    unsigned int  detNum;
    detBox pBox[MAX_DETECT_NUM];
}person_detect_out_t;

void* post_process_person_detection(nn_output *pOut);
int person_do_post_process(person_detect_out_t* pperson_detect_result);
void *post_process_person_detect(nn_output *pOut);
void* postprocess_person_detect(nn_output *pout);

float retina_overlap(float x1, float w1, float x2, float w2);
float retina_box_intersection(box a, box b);
float retina_box_union(box a, box b);
float retina_box_iou(box a, box b);
int retina_nms_comparator(const void *pa, const void *pb);
void do_global_sort(box *boxe1,box *boxe2, float prob1[][1],float prob2[][1], int len1,int len2,float thresh);
int person_nms_comparator(const void *pa, const void *pb);
void person_do_nms_sort(box *boxes, float probs[][1], int total, int classes, float thresh);
void person_set_detections(int num, float thresh, box *boxes, float probs[][1],person_detect_out_t* pperson_detect_result);


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
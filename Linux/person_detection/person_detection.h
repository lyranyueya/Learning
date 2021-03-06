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

int person_do_post_process(person_detect_out_t* pperson_detect_result);
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

#ifdef __cplusplus
} //extern "C"
#endif
#endif
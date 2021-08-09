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
#include "sdk_log.h"
#include "nn_util.h"
#ifdef __cplusplus
extern "C" {
#endif
extern float bbox_32[12][20][8]; //384/32,640/32
extern float bbox_16[24][40][8];
extern float bbox_8[48][80][8];
//prob score,input
extern float prob_32[240][2][2];
extern float prob_16[960][2][2];
extern float prob_8[3840][2][2];
//land mark
extern float land_32[12][20][20]; //384/32,640/32
extern float land_16[24][40][20];
extern float land_8[48][80][20];

extern float p_bbox_32[8][12][20]; //384/32,640/32
extern float p_bbox_16[8][24][40];
extern float p_bbox_8[8][48][80];
//prob score,input
extern float p_prob_32[480][2];
extern float p_prob_16[1920][2];
extern float p_prob_8[7680][2];

extern float bbox[5875][4];
extern float pprob[5875][2];
extern float llandmark[5875][10];

typedef struct detection_{
    box bbox;
    float *prob;
    float objectness;
    int classes;
    int sort_class;
}detection;
typedef struct __nn_bodypos
{
	int valid;    //whether this point is valid
	point_t pos;
}bodypos_t;
typedef struct __nn_body_pose
{
	unsigned int  detNum;
	bodypos_t bpos[18];
}body_pose_out_t;

float overlap(float x1, float w1, float x2, float w2);
float box_intersection(box a, box b);
float box_union(box a, box b);
float box_iou(box a, box b);
int nms_comparator(const void *pa, const void *pb);
void do_nms_sort(box *boxes, float **probs, int total, int classes, float thresh);
void flatten(float *x, int size, int layers, int batch, int forward);
void softmax(float *input, int n, float temp, float *output);
float sigmod(float x);
float logistic_activate(float x);
unsigned char *transpose(const unsigned char * src,int width,int height);
int entry_index(int lw, int lh, int lclasses, int loutputs, int batch, int location, int entry);
void activate_array(float *start, int num);

float retina_overlap(float x1, float w1, float x2, float w2);
float retina_box_intersection(box a, box b);
float retina_box_union(box a, box b);
float retina_box_iou(box a, box b);
int retina_nms_comparator(const void *pa, const void *pb);
void retina_do_nms_sort(box *boxes, float probs[][1], int total,  float thresh);
void retina_result(int num, float thresh, box *boxes, float probs[][1],landmark *pland,face_detect_out_t* pface_det_result);
void retina_point5_result(int num, float thresh, box *boxes, float probs[][1],landmark *pland,face_landmark5_out_t* pface_landmark5_result);

void* postprocess_bodypose(nn_output *pout);
void* post_posenet(float* pbody,unsigned int size);
int person_do_post_process(person_detect_out_t* pperson_detect_result);

/*************      pose_postprocess.c      *******/
void* post_posenet(float* pbody,unsigned int size);

#ifdef __cplusplus
} //extern "C"
#endif
#endif
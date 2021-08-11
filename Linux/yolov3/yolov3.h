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
typedef struct __nn_obj_detect
{
	unsigned int  detNum;
	detBox *pBox;
}obj_detect_out_t;

void* yolov3_postprocess(float **predictions, int width, int height, int modelWidth, int modelHeight, int input_num);
void* postprocess_yolov3(nn_output *pout);
int yolo_v3_post_process_onescale(float *predictions, int input_size[3] , float *biases, box *boxes, float **pprobs, float threshold_in);
void* yolov2_result(int num, float thresh, box *boxes, float **probs, int classes);

float overlap(float x1, float w1, float x2, float w2);
float box_intersection(box a, box b);
float box_union(box a, box b);
float box_iou(box a, box b);
int nms_comparator(const void *pa, const void *pb);
void do_nms_sort(box *boxes, float **probs, int total, int classes, float thresh);
void flatten(float *x, int size, int layers, int batch, int forward);
void softmax(float *input, int n, float temp, float *output);
float logistic_activate(float x);
box get_region_box(float *x, float *biases, int n, int index, int i, int j, int w, int h);
int max_index(float *a, int n);
float get_color(int c, int x, int max);


#ifdef __cplusplus
} //extern "C"
#endif
#endif
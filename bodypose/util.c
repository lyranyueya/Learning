/****************************************************************************
*
*    Copyright (c) 2019  by amlogic Corp.  All rights reserved.
*
*    The material in this file is confidential and contains trade secrets
*    of amlogic Corporation. No part of this work may be disclosed,
*    reproduced, copied, transmitted, or used in any way for any purpose,
*    without the express written permission of amlogic Corporation.
*
***************************************************************************/

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "body_pose.h"
#include <time.h>
#include <math.h>
#include "nn_sdk.h"
#include "nn_util.h"
#include "jpeglib.h"
#include <unistd.h>
#include<sys/types.h>
#include<fcntl.h>

#define _BASETSD_H
#define NN_TENSOR_MAX_DIMENSION_NUMBER 4
#define X_MAX 639.0 / 640.0
#define Y_MAX 383.0 / 384.0
#define FLT_MAX 3.402823466e+38F
/** Status enum */
typedef enum
{
    UTIL_FAILURE = -1,
    UTIL_SUCCESS = 0,
}nn_status_e;

/*-------------------------------------------
				jpeg_util
-------------------------------------------*/
static int _jpeg_to_bmp
    (
    FILE * inputFile,
    unsigned char* bmpData,
    unsigned int bmpWidth,
    unsigned int bmpHeight,
    unsigned int channel
    )
{
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;
    JSAMPARRAY buffer;
    unsigned char *point = NULL;
    unsigned long width, height;
    unsigned short depth = 0;

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo,inputFile);
    jpeg_read_header(&cinfo,TRUE);

    cinfo.dct_method = JDCT_IFAST;

    if (bmpData == NULL)
    {
        return -1;
    }
    else
    {
        jpeg_start_decompress(&cinfo);

        width  = cinfo.output_width;
        height = cinfo.output_height;
        depth  = cinfo.output_components;
        if (width * height * depth != bmpWidth * bmpHeight * channel)
        {
           printf("wrong jpg file , the jpg file size should be %u %u %u\n",
               bmpWidth, bmpHeight, channel);
           return -1;
        }

        buffer = (*cinfo.mem->alloc_sarray)
            ((j_common_ptr)&cinfo, JPOOL_IMAGE, width*depth, 1);

        point = bmpData;

        while (cinfo.output_scanline < height)
        {
            jpeg_read_scanlines(&cinfo, buffer, 1);
            memcpy(point, *buffer, width * depth);
            point += width * depth;
        }

        jpeg_finish_decompress(&cinfo);
    }

    jpeg_destroy_decompress(&cinfo);

    return 0;
}

unsigned char *get_jpeg_rawData(const char *name,unsigned int width,unsigned int height)
{
    FILE *bmpFile;
    unsigned char *bmpData;
    unsigned int sz,w,h,c;
    int status;

    bmpFile = NULL;
    bmpData = NULL;
    w = width;
    h = height;
    c = 3;
    sz = w*h*3;

    bmpFile = fopen( name, "rb" );
	if (bmpFile == NULL)
	{
		printf("null returned\n");
		goto final;
	}

    bmpData = (unsigned char *)malloc(sz * sizeof(char));
	if (bmpData == NULL)
	{
		printf("null returned\n");
		goto final;
	}
    memset(bmpData, 0, sz * sizeof(char));

    status = _jpeg_to_bmp( bmpFile, bmpData, w, h, c);
    if (status == -1)
    {
        free(bmpData);
        fclose(bmpFile);
        return NULL;
    }

final:
    if (bmpFile)fclose(bmpFile);
    return bmpData;
}

float Float16ToFloat32(const signed short* src , float* dst ,int lenth)
{
	signed int t1;
	signed int t2;
	signed int t3;
	float out;
	int i;
	for (i = 0 ;i < lenth ;i++)
	{
		t1 = src[i] & 0x7fff;                       // Non-sign bits
		t2 = src[i] & 0x8000;                       // Sign bit
		t3 = src[i] & 0x7c00;                       // Exponent

		t1 <<= 13;                              // Align mantissa on MSB
		t2 <<= 16;                              // Shift sign bit into position

		t1 += 0x38000000;                       // Adjust bias

		t1 = (t3 == 0 ? 0 : t1);                // Denormals-as-zero

		t1 |= t2;
		*((unsigned int*)&out) = t1;                 // Re-insert sign bit
		dst[i] = out;

	}
	return out;
}

float *dtype_To_F32(nn_output * outdata ,int sz)
{
	int stride, fl, i, zeropoint;
	float scale;
	unsigned char *buffer_u8 = NULL;
	signed char *buffer_int8 = NULL;
	signed short *buffer_int16 = NULL;
	float *buffer_f32 = NULL;

	buffer_f32 = (float *)malloc(sizeof(float) * sz );

	if (outdata->out[0].param->data_format == NN_BUFFER_FORMAT_UINT8)
	{
		stride = (outdata->out[0].size)/sz;
		scale = outdata->out[0].param->quant_data.affine.scale;
		zeropoint =  outdata->out[0].param->quant_data.affine.zeroPoint;

		buffer_u8 = (unsigned char*)outdata->out[0].buf;
		for (i = 0; i < sz; i++)
		{
			buffer_f32[i] = (float)(buffer_u8[stride * i] - zeropoint) * scale;
		}
	}

	else if (outdata->out[0].param->data_format == NN_BUFFER_FORMAT_INT8)
	{
		buffer_int8 = (signed char*)outdata->out[0].buf;
		if (outdata->out[0].param->quant_data.dfp.fixed_point_pos >= 0)
		{
			fl = 1 << (outdata->out[0].param->quant_data.dfp.fixed_point_pos);
			for (i = 0; i < sz; i++)
			{
				buffer_f32[i] = (float)buffer_int8[i] * (1.0/(float)fl);
			}
		}
		else
		{
			fl = 1 << (-outdata->out[0].param->quant_data.dfp.fixed_point_pos);
			for (i = 0; i < sz; i++)
				buffer_f32[i] = (float)buffer_int8[i] * ((float)fl);
		}
	}

	else if (outdata->out[0].param->data_format == NN_BUFFER_FORMAT_INT16)
	{
		buffer_int16 =	(signed short*)outdata->out[0].buf;
		if (outdata->out[0].param->quant_data.dfp.fixed_point_pos >= 0)
		{
			fl = 1 << (outdata->out[0].param->quant_data.dfp.fixed_point_pos);
			for (i = 0; i < sz; i++)
			{
				buffer_f32[i] = (float)((buffer_int16[i]) * (1.0/(float)fl));
			}
		}
		else
		{
			fl = 1 << (-outdata->out[0].param->quant_data.dfp.fixed_point_pos);
			for (i = 0; i < sz; i++)
				buffer_f32[i] = (float)((buffer_int16[i]) * ((float)fl));
		}
	}
	else if (outdata->out[0].param->data_format == NN_BUFFER_FORMAT_FP16 )
	{
		buffer_int16 = (signed short*)outdata->out[0].buf;

		Float16ToFloat32(buffer_int16 ,buffer_f32 ,sz);
	}

	else if (outdata->out[0].param->data_format == NN_BUFFER_FORMAT_FP32)
	{
		memcpy(buffer_f32, outdata->out[0].buf, sz);
	}
	else
	{
		printf("Error: currently not support type, type = %d\n", outdata->out[0].param->data_format);
	}
	return buffer_f32;
}

/*-------------------------------------------
			cv_postprocess_util
-------------------------------------------*/
float overlap(float x1, float w1, float x2, float w2)
{
    float l1 = x1 - w1/2;
    float l2 = x2 - w2/2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1/2;
    float r2 = x2 + w2/2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}

float box_intersection(box a, box b)
{
    float area = 0;
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    if (w < 0 || h < 0)
        return 0;
    area = w*h;
    return area;
}

float box_union(box a, box b)
{
    float i = box_intersection(a, b);
    float u = a.w*a.h + b.w*b.h - i;
    return u;
}

float box_iou(box a, box b)
{
    return box_intersection(a, b)/box_union(a, b);
}

int nms_comparator(const void *pa, const void *pb)
{
    sortable_bbox a = *(sortable_bbox *)pa;
    sortable_bbox b = *(sortable_bbox *)pb;
    float diff = a.probs[a.index][b.classId] - b.probs[b.index][b.classId];
    if (diff < 0) return 1;
    else if(diff > 0) return -1;
    return 0;
}

void do_nms_sort(box *boxes, float **probs, int total, int classes, float thresh)
{
    int i, j, k;
    sortable_bbox *s = (sortable_bbox *)calloc(total, sizeof(sortable_bbox));

    for (i = 0; i < total; ++i)
    {
        s[i].index = i;
        s[i].classId = 0;
        s[i].probs = probs;
    }
    for (k = 0; k < classes; ++k)
    {
        for (i = 0; i < total; ++i)
        {
            s[i].classId = k;
        }
        qsort(s, total, sizeof(sortable_bbox), nms_comparator);
        for (i = 0; i < total; ++i)
        {
            if (probs[s[i].index][k] == 0)
                continue;
            for (j = i+1; j < total; ++j)
            {
                box b = boxes[s[j].index];
                if (probs[s[j].index][k]>0)
                {
                    if (box_iou(boxes[s[i].index], b) > thresh)
                    {
                        probs[s[j].index][k] = 0;
                    }
                }
            }
        }
    }
    free(s);
}

void flatten(float *x, int size, int layers, int batch, int forward)
{
    float *swap = (float*)calloc(size*layers*batch, sizeof(float));
    int i,c,b;
    for (b = 0; b < batch; ++b)
    {
        for (c = 0; c < layers; ++c)
        {
            for (i = 0; i < size; ++i)
            {
                int i1 = b*layers*size + c*size + i;
                int i2 = b*layers*size + i*layers + c;
                if (forward) swap[i2] = x[i1];
                else swap[i1] = x[i2];
            }
        }
    }
    memcpy(x, swap, size*layers*batch*sizeof(float));
    free(swap);
}

void softmax(float *input, int n, float temp, float *output)
{
    int i;
    float sum = 0;
    float largest = -FLT_MAX;
    for (i = 0; i < n; ++i)
    {
        if (input[i] > largest) largest = input[i];
    }
    for (i = 0; i < n; ++i)
    {
        float e = exp(input[i]/temp - largest/temp);
        sum += e;
        output[i] = e;
    }
    for (i = 0; i < n; ++i)
    {
        output[i] /= sum;
    }
}

float sigmod(float x)
{
    return 1.0/(1+exp(-x));
}

float logistic_activate(float x)
{
    return 1./(1. + exp(-x));
}

unsigned char *transpose(const unsigned char * src,int width,int height)
{
    unsigned char* dst;
    int i,j,m;
    int channel = 3;

    dst = (unsigned char*)malloc(width*height*channel);
    memset(dst,0,width*height*channel);

    /*hwc -> whc*/
    for (i = 0;i < width; i++)
    {
        for (j = 0; j < height; j++)
        {
            for (m = 0;m < channel;m++)
                *(dst + i * height * channel + j * channel + m) = *(src + j * width * channel + i * channel + m);
        }
    }
    return dst;
}

int entry_index(int lw, int lh, int lclasses, int loutputs, int batch, int location, int entry)
{
    int n = location / (lw*lh);
    int loc = location % (lw*lh);
    return batch * loutputs + n * lw*lh*(4 + lclasses + 1) + entry * lw*lh + loc;
}

void activate_array(float *start, int num)
{
    for (int i = 0; i < num; i ++){
        start[i] = logistic_activate(start[i]);
    }
}

/*-------------------------------------------
				retina_model
-------------------------------------------*/
typedef unsigned char   uint8_t;
typedef struct
{
    int index;
    int classId;
    float probs;
} sortable_bbox_retina;

float bbox_32[12][20][8]; //384/32,640/32
float bbox_16[24][40][8];
float bbox_8[48][80][8];
//prob score,input
float prob_32[240][2][2];
float prob_16[960][2][2];
float prob_8[3840][2][2];
//land mark
float land_32[12][20][20]; //384/32,640/32
float land_16[24][40][20];
float land_8[48][80][20];

static float prob32[480][1];
static float prob16[1920][1];
static float prob8[7680][1];
//output box
static box box32[12][20][2];
static box *pbox32;
static box box16[24][40][2];
static box *pbox16;
static box box8[48][80][2];
static box *pbox8;
//landmark
static landmark land32[12][20][2][5];
static landmark *pland32;
static landmark land16[24][40][2][5];
static landmark *pland16;
static landmark land8[48][80][2][5];
static landmark *pland8;

int g_detect_number = 230;
float retina_overlap(float x1, float w1, float x2, float w2)
{
    float l1 = x1;
    float l2 = x2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1;
    float r2 = x2 + w2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}
float retina_box_intersection(box a, box b)
{
    float area = 0;
    float w = retina_overlap(a.x, a.w, b.x, b.w);
    float h = retina_overlap(a.y, a.h, b.y, b.h);
    if (w < 0 || h < 0)
        return 0;
    area = w*h;
    return area;
}

float retina_box_union(box a, box b)
{
    float i = retina_box_intersection(a, b);
    float u = a.w*a.h + b.w*b.h - i;
    return u;
}

float retina_box_iou(box a, box b)
{
    return retina_box_intersection(a, b)/retina_box_union(a, b);
}

int retina_nms_comparator(const void *pa, const void *pb)
{
    sortable_bbox_retina a = *(sortable_bbox_retina *)pa;
    sortable_bbox_retina b = *(sortable_bbox_retina *)pb;
    float diff = a.probs - b.probs;
    if (diff < 0) return 1;
    else if (diff > 0) return -1;
    return 0;
}

void do_global_sort(box *boxe1,box *boxe2, float prob1[][1],float prob2[][1], int len1,int len2,float thresh)
{
    int i,j;
    for (i = 0; i < len1; ++i)
    {
        if (prob1[i][0] > thresh)
        {
            for (j = 0;j < len2;j++)
            {
                if (prob2[j][0] > thresh)
                {
                    if (retina_box_iou(boxe1[i], boxe2[j]) > 0.1)
                    {
                        if (prob2[j][0] > prob1[i][0])
                        {
                            prob1[i][0] = 0;
                        }
                        else
                        {
                            prob2[j][0] = 0;
                        }
                    }
                }
            }
        }
    }
}

void retina_do_nms_sort(box *boxes, float probs[][1], int total,  float thresh)
{
    int i, j;
    sortable_bbox_retina *s = (sortable_bbox_retina *)calloc(total, sizeof(sortable_bbox_retina));
    for (i = 0; i < total; ++i)
    {
        s[i].index = i;
        s[i].classId = 0;
        s[i].probs = probs[i][0];
        //printf("%f\n",s[i].probs);
    }

    for (i = 0; i < total; ++i)
    {
        s[i].classId = 0;
    }
    qsort(s, total, sizeof(sortable_bbox_retina), retina_nms_comparator);

    for (i = 0; i < total; ++i)
    {
        if (probs[s[i].index][0] >= 0.01)  //zxw
        {
            for (j = i+1; j < total; j++)
            {
                if (probs[s[j].index][0] >= 0.01)
                {
                    box b = boxes[s[j].index];
                    if (retina_box_iou(boxes[s[i].index], b) > thresh)
                    {
                        probs[s[j].index][0] = 0;
                    }
                }
            }
        }
    }
    free(s);
}


void retina_result(int num, float thresh, box *boxes, float probs[][1],landmark *pland,face_detect_out_t* pface_det_result)
{

    int i;
    int detect_num = pface_det_result->detNum;
    for (i = 0; i < num; i++)
    {
        float prob = probs[i][0];
        if (detect_num < MAX_DETECT_NUM)
        {
            if (prob > thresh)
            {
                if (detect_num >= g_detect_number)
                {
                    break;
                }
                int left = 0;
                int right = 0;
                int top = 0;
                int bot = 0;
                left  = boxes[i].x / 640.0;
                right = (boxes[i].x + boxes[i].w) / 640.0;
                top   = boxes[i].y / 384.0;
                bot   = (boxes[i].y + boxes[i].h) / 384.0;

                if (left < 0) left = 0;
                if (right > 1) right = 1.0;
                if (top < 0) top = 0;
                if (bot > 1) bot = 1.0;
                pface_det_result->pBox[detect_num].x = boxes[i].x / 640.0;
                pface_det_result->pBox[detect_num].y = boxes[i].y / 384.0;
                pface_det_result->pBox[detect_num].w = boxes[i].w / 640.0;
                pface_det_result->pBox[detect_num].h = boxes[i].h / 384.0;
                if (pface_det_result->pBox[detect_num].x <= 0 ) pface_det_result->pBox[detect_num].x =0.000001;
                if (pface_det_result->pBox[detect_num].y <= 0 ) pface_det_result->pBox[detect_num].y =0.000001;
                if (pface_det_result->pBox[detect_num].w <= 0 ) pface_det_result->pBox[detect_num].w =0.000001;
                if (pface_det_result->pBox[detect_num].h <= 0 ) pface_det_result->pBox[detect_num].h =0.000001;
                if (pface_det_result->pBox[detect_num].x >= 1 ) pface_det_result->pBox[detect_num].x =0.999999;
                if (pface_det_result->pBox[detect_num].y >= 1 ) pface_det_result->pBox[detect_num].y =0.999999;
                if (pface_det_result->pBox[detect_num].w >= 1 ) pface_det_result->pBox[detect_num].w =0.999999;
                if (pface_det_result->pBox[detect_num].h >= 1 ) pface_det_result->pBox[detect_num].h =0.999999;
                detect_num++;
            }
        }
    }
    //printf("detect number =%d \n", detect_num);
    pface_det_result->detNum = detect_num;
}


void retina_point5_result(int num, float thresh, box *boxes, float probs[][1],landmark *pland,face_landmark5_out_t* pface_landmark5_result)
{

    int i,j;
    int detect_num = pface_landmark5_result->detNum;
    for (i = 0; i < num; i++)
    {
        float prob = probs[i][0];
        if (detect_num < MAX_DETECT_NUM)
        {
            if (prob > thresh)
            {
                if (detect_num >= g_detect_number)
                {
                    break;
                }
                int left = 0;
                int right = 0;
                int top = 0;
                int bot = 0;

                left  = boxes[i].x / 640.0;
                right = (boxes[i].x + boxes[i].w) / 640.0;
                top   = boxes[i].y / 384.0;
                bot   = (boxes[i].y + boxes[i].h) / 384.0;

                if (left < 0) left = 0;
                if (right > 1) right = 1.0;
                if (top < 0) top = 0;
                if (bot > 1) bot = 1.0;
                pface_landmark5_result->facebox[detect_num].x = boxes[i].x / 640.0;
                pface_landmark5_result->facebox[detect_num].y = boxes[i].y / 384.0;
                pface_landmark5_result->facebox[detect_num].w = boxes[i].w / 640.0;
                pface_landmark5_result->facebox[detect_num].h = boxes[i].h / 384.0;
                if (pface_landmark5_result->facebox[detect_num].x <= 0 ) pface_landmark5_result->facebox[detect_num].x =0.000001;
                if (pface_landmark5_result->facebox[detect_num].y <= 0 ) pface_landmark5_result->facebox[detect_num].y =0.000001;
                if (pface_landmark5_result->facebox[detect_num].w <= 0 ) pface_landmark5_result->facebox[detect_num].w =0.000001;
                if (pface_landmark5_result->facebox[detect_num].h <= 0 ) pface_landmark5_result->facebox[detect_num].h =0.000001;
                if (pface_landmark5_result->facebox[detect_num].x >= 1 ) pface_landmark5_result->facebox[detect_num].x =0.999999;
                if (pface_landmark5_result->facebox[detect_num].y >= 1 ) pface_landmark5_result->facebox[detect_num].y =0.999999;
                if (pface_landmark5_result->facebox[detect_num].w >= 1 ) pface_landmark5_result->facebox[detect_num].w =0.999999;
                if (pface_landmark5_result->facebox[detect_num].h >= 1 ) pface_landmark5_result->facebox[detect_num].h =0.999999;
                for (j=0 ;j <5 ; j++)
                {
                    pface_landmark5_result->pos[detect_num][j].x = pland[i * 5 + j].x / 640.0;
                    pface_landmark5_result->pos[detect_num][j].y = pland[i * 5 + j].y / 384.0;
                    if (pface_landmark5_result->pos[detect_num][j].x <= 0) pface_landmark5_result->pos[detect_num][j].x=0.001;
                    if (pface_landmark5_result->pos[detect_num][j].x >= X_MAX) pface_landmark5_result->pos[detect_num][j].x=0.997;
                    if (pface_landmark5_result->pos[detect_num][j].y <= 0) pface_landmark5_result->pos[detect_num][j].y=0.001;
                    if (pface_landmark5_result->pos[detect_num][j].y >= Y_MAX) pface_landmark5_result->pos[detect_num][j].y=0.997;
                    //printf("point number =%d,rawData-X:%.5f,Y:%.5f\n" , j, pface_landmark5_result->pos[detect_num][j].x, pface_landmark5_result->pos[detect_num][j].y);
                }
                detect_num++;
            }
        }
    }
    //printf("detect number =%d \n", detect_num);
    pface_landmark5_result->detNum = detect_num;
}



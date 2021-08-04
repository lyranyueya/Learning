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
#include "person_detection.h"
#include <time.h>
#include <math.h>
#include "nn_sdk.h"
#include "nn_util.h"
#include "jpeglib.h"

float p_bbox_32[8][12][20]; //384/32,640/32
float p_bbox_16[8][24][40];
float p_bbox_8[8][48][80];
//prob score,input
float p_prob_32[480][2];
float p_prob_16[1920][2];
float p_prob_8[7680][2];


static float p_prob32[480][1];
static float p_prob16[1920][1];
static float p_prob8[7680][1];
//output box
static box p_box32[12][20][2];
static box *p_pbox32;
static box p_box16[24][40][2];
static box *p_pbox16;
static box p_box8[48][80][2];
static box *p_pbox8;
int g_detect_number = 230;

typedef struct{
    int index;
    int classId;
    float probs;  //**probs to probs
} sortable_bbox_person;
typedef struct
{
    int index;
    int classId;
    float probs;
} sortable_bbox_retina;
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

/*******************     person detect postprocess util     ********************/
/****************************************************************************/

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

int person_nms_comparator(const void *pa, const void *pb)
{
    sortable_bbox_person a = *(sortable_bbox_person *)pa;
    sortable_bbox_person b = *(sortable_bbox_person *)pb;
    float diff = a.probs - b.probs;
    if (diff < 0) return 1;
    else if (diff > 0) return -1;
    return 0;
}

void person_do_nms_sort(box *boxes, float probs[][1], int total, int classes, float thresh)
{
    int i, j, k;
    sortable_bbox_person *s = (sortable_bbox_person *)calloc(total, sizeof(sortable_bbox_person));
	if (s == NULL)
	{
		printf("terrible calloc fail\n");
		return;
	}
    for (i = 0; i < total; ++i)
    {
        s[i].index = i;
        s[i].classId = 0;
        s[i].probs = probs[i][0];
    }

    for (k = 0; k < classes; ++k)
    {
        for (i = 0; i < total; ++i)
        {
            s[i].classId = k;
        }
        qsort(s, total, sizeof(sortable_bbox_person), person_nms_comparator);
        for (i = 0; i < total; ++i)
		{
            if (probs[s[i].index][k] <= 0.02)  //zxw
			{
				probs[s[i].index][k] = 0; //zxw;
				continue;
			}
            for (j = i+1; j < total; ++j)
            {
                box b = boxes[s[j].index];
                if (retina_box_iou(boxes[s[i].index], b) > thresh)
                {
                    probs[s[j].index][k] = 0;
                }
            }
        }
    }
    free(s);
}

void person_set_detections(int num, float thresh, box *boxes, float probs[][1],person_detect_out_t* pperson_detect_result)
{
	int i;
	int detect_num = pperson_detect_result->detNum;

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
                float left = 0;
                float right = 0;
                float top = 0;
                float bot = 0;

                left  = boxes[i].x / 640.0;
                right = (boxes[i].x + boxes[i].w) / 640.0;
                top   = boxes[i].y / 384.0;
                bot   = (boxes[i].y + boxes[i].h) / 384.0;
                if (left < 0) left = 0;
                if (right > 1) right = 1.0;
                if (top < 0) top = 0;
                if (bot > 1) bot = 1.0;
                pperson_detect_result->pBox[detect_num].x = boxes[i].x / 640.0;
                pperson_detect_result->pBox[detect_num].y = boxes[i].y / 384.0;
                pperson_detect_result->pBox[detect_num].w = boxes[i].w / 640.0;
                pperson_detect_result->pBox[detect_num].h = boxes[i].h / 384.0;

                if (pperson_detect_result->pBox[detect_num].x <= 0 ) pperson_detect_result->pBox[detect_num].x =0.000001;
                if (pperson_detect_result->pBox[detect_num].y <= 0 ) pperson_detect_result->pBox[detect_num].y =0.000001;
                if (pperson_detect_result->pBox[detect_num].w <= 0 ) pperson_detect_result->pBox[detect_num].w =0.000001;
                if (pperson_detect_result->pBox[detect_num].h <= 0 ) pperson_detect_result->pBox[detect_num].h =0.000001;
                if (pperson_detect_result->pBox[detect_num].x >= 1 ) pperson_detect_result->pBox[detect_num].x =0.999999;
                if (pperson_detect_result->pBox[detect_num].y >= 1 ) pperson_detect_result->pBox[detect_num].y =0.999999;
                if (pperson_detect_result->pBox[detect_num].w >= 1 ) pperson_detect_result->pBox[detect_num].w =0.999999;
                if (pperson_detect_result->pBox[detect_num].h >= 1 ) pperson_detect_result->pBox[detect_num].h =0.999999;

                detect_num++;
            }
        }
    }
    pperson_detect_result->detNum = detect_num;
}

/*******************     person detect postprocess util    ********************/
/****************************************************************************/


/****************************************************************************/
/*******************     person detect postprocess       ********************/

int person_do_post_process(person_detect_out_t* pperson_detect_result)
{
	int i = 0,x,y;
    int idx;
    int h32;
	float prior_w,prior_h;
	float prior_cw,prior_ch;
	int pred_ctrx,pred_ctry,predw,predh;
	int valid_8 = 0,valid_16 = 0,valid_32 = 0;

	p_pbox32 = (box *)p_box32;
	p_pbox16 = (box *)p_box16;
	p_pbox8 = (box *)p_box8;

	for (i=0;i<480;i++)
	{
		p_prob32[i][0] = p_prob_32[i][1];
		if (p_prob32[i][0] > 0.8)
			valid_32 = 1;
		else
			p_prob32[i][0]=0;
	}

	for (i=0;i<1920;i++)
	{
		p_prob16[i][0] = p_prob_16[i][1];
		if (p_prob16[i][0] > 0.8)
			valid_16 = 1;
		else
			p_prob16[i][0]=0;
	}

	for (i=0;i<7680;i++)
	{
		p_prob8[i][0] = p_prob_8[i][1];
		if (p_prob8[i][0] > 0.8)
			valid_8 = 1;
		else
			p_prob8[i][0]=0;
	}

	if (valid_32)
	{
		for (y=0;y<12;y++)
		{
			for (x=0;x<20;x++)
			{
				for (idx=0;idx<2;idx++)
				{
					if (idx == 0)
						h32=256;
					else
						h32=512;

					prior_w = (float)h32;
					prior_h = h32*2.44;
					prior_cw = (x+0.5)*32;
					prior_ch = (y+0.5)*32;

					pred_ctrx = p_bbox_32[4*idx][y][x]*0.1*prior_w+prior_cw;
					pred_ctry = p_bbox_32[4*idx+1][y][x]*0.1*prior_h+prior_ch;
					predw = exp(p_bbox_32[4*idx+2][y][x]*0.2)*prior_w;
					predh = exp(p_bbox_32[4*idx+3][y][x]*0.2)*prior_h;
					p_box32[y][x][idx].x = pred_ctrx-0.5*predw;
					p_box32[y][x][idx].y = pred_ctry-0.5*predh;
					p_box32[y][x][idx].w = predw;
					p_box32[y][x][idx].h = predh;
				}
			}
		}
	}

	if (valid_16)
	{
		for (y=0;y<24;y++)
		{
			for (x=0;x<40;x++)
			{
				for (idx=0;idx<2;idx++)
				{
					if (idx == 0)
						h32=64;
					else
						h32=128;

					prior_w = (float)h32;
					prior_h = h32*2.44;
					prior_cw = (x+0.5)*16;
					prior_ch = (y+0.5)*16;

					pred_ctrx = p_bbox_16[4*idx][y][x]*0.1*prior_w+prior_cw;
					pred_ctry = p_bbox_16[4*idx+1][y][x]*0.1*prior_h+prior_ch;
					predw = exp(p_bbox_16[4*idx+2][y][x]*0.2)*prior_w;
					predh = exp(p_bbox_16[4*idx+3][y][x]*0.2)*prior_h;
					p_box16[y][x][idx].x = pred_ctrx-0.5*predw;
					p_box16[y][x][idx].y = pred_ctry-0.5*predh;
					p_box16[y][x][idx].w = predw;
					p_box16[y][x][idx].h = predh;
				}
			}
		}
	}

	if (valid_8)
	{
		for (y=0;y<48;y++)
		{
			for (x=0;x<80;x++)
			{
				for (idx=0;idx<2;idx++)
				{
					if (idx == 0)
						h32=16;
					else
						h32=32;

					prior_w = (float)h32;
					prior_h = h32*2.44;
					prior_cw = (x+0.5)*8;
					prior_ch = (y+0.5)*8;

					pred_ctrx = p_bbox_8[4*idx][y][x]*prior_w*0.1+prior_cw;
					pred_ctry = p_bbox_8[4*idx+1][y][x]*prior_h*0.1+prior_ch;
					predw = exp(p_bbox_8[4*idx+2][y][x]*0.2)*prior_w;
					predh = exp(p_bbox_8[4*idx+3][y][x]*0.2)*prior_h;
					p_box8[y][x][idx].x = (pred_ctrx-0.5*predw);
					p_box8[y][x][idx].y = (pred_ctry-0.5*predh);
					p_box8[y][x][idx].w = predw;
					p_box8[y][x][idx].h = predh;
				}
			}
		}
	}

	if (valid_32 == 1)
	{
		person_do_nms_sort(p_pbox32, p_prob32, 480, 1, 0.4);
		if (valid_16 == 1)
		{
			person_do_nms_sort(p_pbox16, p_prob16, 1920, 1, 0.4);
			do_global_sort(p_pbox32,p_pbox16,p_prob32,p_prob16,480,1920,0.7);
			if (valid_8 == 1)
			{
				person_do_nms_sort(p_pbox8, p_prob8, 7680, 1, 0.2);
				do_global_sort(p_pbox16, p_pbox8, p_prob16, p_prob8, 1920,7680,0.7);
				person_set_detections(480, 0.6, p_pbox32, p_prob32, pperson_detect_result);
				person_set_detections(1920, 0.6, p_pbox16, p_prob16, pperson_detect_result);
				person_set_detections(7680, 0.6, p_pbox8, p_prob8, pperson_detect_result);
			}
			else
			{
				person_set_detections(480, 0.6, p_pbox32, p_prob32, pperson_detect_result);
				person_set_detections(1920, 0.6, p_pbox16, p_prob16, pperson_detect_result);
			}
		}
		else person_set_detections(480, 0.6, p_pbox32, p_prob32, pperson_detect_result);
	}
	if (valid_16 == 1 && valid_32 == 0 )
	{
		person_do_nms_sort(p_pbox16, p_prob16, 1920, 1, 0.4);
		if (valid_8 == 1)
		{
			person_do_nms_sort(p_pbox8, p_prob8, 7680, 1, 0.4);
			do_global_sort(p_pbox16, p_pbox8, p_prob16, p_prob8, 1920,7680,0.7);
			person_set_detections(1920, 0.6, p_pbox16, p_prob16, pperson_detect_result);
			person_set_detections(7680, 0.6, p_pbox8, p_prob8, pperson_detect_result);
		}
		else
			person_set_detections(1920, 0.6, p_pbox16, p_prob16, pperson_detect_result);
	}

	if (valid_8 == 1 && valid_16 == 0 && valid_32 == 0 )
	{
		person_do_nms_sort(p_pbox8, p_prob8, 7680, 1, 0.2);
		person_set_detections(7680, 0.6, p_pbox8, p_prob8, pperson_detect_result);
	}

	return 0;
}

void* postprocess_person_detect(nn_output *pout)
{
    unsigned int j;
    unsigned int sz;
    float *buffer = NULL;

    static person_detect_out_t person_detect_result;
    memset(&person_detect_result,0,sizeof(point_t));

    for (j = 0; j < pout->num; j++)
    {
        buffer = (float *)pout->out[j].buf;
        sz= pout->out[j].size;

        switch (j)
        {
            case 0:
                memcpy(p_bbox_8,buffer,sz);
                break;
            case 1:
                memcpy(p_bbox_16,buffer,sz);
                break;
            case 2:
                memcpy(p_bbox_32,buffer,sz);
                break;
            case 3:
                memcpy(p_prob_8,buffer,sz);
                break;
            case 4:
                memcpy(p_prob_16,buffer,sz);
                break;
            case 5:
                memcpy(p_prob_32,buffer,sz);
                break;
            default:
                break;
        }
    }
    person_do_post_process(&person_detect_result);
    return (void*)(&person_detect_result);
}

void *post_process_person_detection(nn_output *pOut)
{
	void *data = NULL;
	data = postprocess_person_detect(pOut);
	return data;
}

/****************************************************************************/
/*******************     person detect postprocess      ********************/



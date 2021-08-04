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
#include "head_detection.h"
#include <time.h>
#include <math.h>
#include "nn_sdk.h"
#include "nn_util.h"
#include "jpeglib.h"

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

/*******************     face detect postprocess      ********************/
/****************************************************************************/

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

float sigmod(float x)
{
    return 1.0/(1+exp(-x));
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

float box_iou(box a, box b)
{
    return box_intersection(a, b)/box_union(a, b);
}

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


/*******************     head detect postprocess      ********************/
/****************************************************************************/
void* postprocess_headdet(nn_output *pout)
{
    float *buffer;
    int i,j,m,index;
    unsigned int sz;
    float result_buffer[13][13][5][6];
    float buffer4th[13][13][5];//buffer4th.shape is (13,13,5)
    float buffer5th[13][13][5];//buffer5th.shape is (13,13,5,1)
    float max,min,classes,confidence;
    float obj_threshold = 0.3;
    float nms_threshold = 0.3;
    float x,y,w,h;
    float anchors[10] = {0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828};
    box *pboxes;
    int box_num = 0,det_num = 0;
    float ** probs = NULL;
    head_det_out_t *headdet_result = NULL;
    detBox *det_boxes = NULL;

    headdet_result = (head_det_out_t*)malloc(sizeof(head_det_out_t));

    buffer = (float*)pout->out[0].buf;
    sz= pout->out[0].size;
    memcpy(result_buffer,buffer,sz);

    /*get the 4th dimension of result_buffer,and sigmod the array*/
    for (i = 0;i < 13;i++)
    {
        for (j = 0;j < 13;j++)
        {
            for (m = 0;m < 5;m++)
            {
                buffer4th[i][j][m] = sigmod(result_buffer[i][j][m][4]);
            }
        }
    }
    /*get the 5th dimension of result_buffer,get its max and min element*/
    for (i = 0;i < 13;i++)
    {
        for (j = 0;j < 13;j++)
        {
            for (m = 0;m < 5;m++)
            {
                buffer5th[i][j][m] = result_buffer[i][j][m][5];
            }
        }
    }

    max = buffer5th[0][0][0];
    min = buffer5th[0][0][0];
    for (i = 0;i < 13;i++)
    {
        for (j = 0;j < 13;j++)
        {
            for (m = 0;m < 5;m++)
            {
                if (buffer5th[i][j][m] > max)
                {
                    max = buffer5th[i][j][m];
                }
                if (buffer5th[i][j][m] < min)
                {
                    min = buffer5th[i][j][m];
                }
            }
        }
    }
    /*softmax(buffer5th)*/
    for (i = 0;i < 13;i++)
    {
        for (j = 0;j < 13;j++)
        {
            for (m = 0;m < 5;m++)
            {
                buffer5th[i][j][m] -= max;
            }
        }
    }

    if (min < (-100.0))
    {
        for (i = 0;i < 13;i++)
        {
            for (j = 0;j < 13;j++)
            {
                for (m = 0;m < 5;m++)
                {
                    buffer5th[i][j][m] = buffer5th[i][j][m]/(min*(-100.0));
                }
            }
        }
    }
    /*
    1. e_x = np.exp(x)
    2. e_x / e_x.sum(axis, keepdims=True) ;[...,1.0,...]
       As e_x.sum(axis, keepdims=True) = e_x  {because e_x.shape is (13,13,5,1)};
       so  e_x / e_x.sum(axis, keepdims=True) must be [...,1.0,...]
    */
    for (i = 0;i < 13;i++)
    {
        for (j = 0;j < 13;j++)
        {
            for (m = 0;m < 5;m++)
            {
                buffer5th[i][j][m] = exp(buffer5th[i][j][m]);
                buffer5th[i][j][m] = 1.0;
            }
        }
    }
    /*
    netout[..., 5:] = netout[..., 4][..., np.newaxis] * _softmax(netout[..., 5:])
    so buffer5th = buffer4th;
    */
    for (i = 0;i < 13;i++)
    {
        for (j = 0;j < 13;j++)
        {
            for (m = 0;m < 5;m++)
            {
                buffer5th[i][j][m] = buffer5th[i][j][m] * buffer4th[i][j][m];
            }
        }
    }
    /* netout[..., 5:] *= netout[..., 5:] > obj_threshold */
    for (i = 0;i < 13;i++)
    {
        for (j = 0;j < 13;j++)
        {
            for (m = 0;m < 5;m++)
            {
                if (buffer5th[i][j][m] < obj_threshold)
                    buffer5th[i][j][m] = 0.0;
                else
                    box_num += 1;
            }
        }
    }

    pboxes = (box*)malloc(sizeof(box)*box_num);
    probs = (float**)calloc(box_num,sizeof(float*)*2);

    index = 0;
    for (i = 0;i < 13;i++)
    {
        for (j = 0;j < 13;j++)
        {
            for (m = 0;m < 5;m++)
            {
                classes = buffer5th[i][j][m];
                if (classes > 0.0)
                {
                    /* reseult_buffer fist 4 elemnt are x,y,w,h */
                    x = (float)result_buffer[i][j][m][0];
                    y = (float)result_buffer[i][j][m][1];
                    w = (float)result_buffer[i][j][m][2];
                    h = (float)result_buffer[i][j][m][3];

                    x = (j + sigmod(x)) / 13;
                    y = (i + sigmod(y)) / 13;
                    w = anchors[m*2 +0] * exp(w) / 13;
                    h = anchors[m*2 +1] * exp(h) / 13;

                    confidence = buffer4th[i][j][m];

                    if (index <= box_num)
                    {
                        pboxes[index].x = x ;
                        pboxes[index].y = y;
                        pboxes[index].w = w;
                        pboxes[index].h = h;

                        probs[index] = (float*)calloc(2,sizeof(float));
                        probs[index][0] = classes;
                        probs[index][1] = confidence;
                    }
                    else
                    {
                        printf("[post_process] The number of boxes exceeds\n");
                    }
                    index += 1;
                }
            }
        }
    }

    do_nms_sort(pboxes,probs,box_num,1,nms_threshold);
    for (i = 0;i < box_num;i++)
        if (probs[i][0] > 0)det_num += 1;

    det_boxes = (detBox *)malloc(det_num * sizeof(detBox));
    index = 0;
    for (i = 0;i < box_num;i++)
    {
        if (probs[i][0] > 0)
        {
            det_boxes[index].x = pboxes[i].x;
            det_boxes[index].y = pboxes[i].y;
            det_boxes[index].w = pboxes[i].w;
            det_boxes[index].h = pboxes[i].h;
            det_boxes[index].score = probs[i][0];
            det_boxes[index].objectClass = probs[i][1];
            index += 1;
        }
    }
    headdet_result->headOut.detNum = det_num;
    headdet_result->headOut.pBox = det_boxes;
    for (i = 0;i < box_num;i++)
        if (probs[i])free(probs[i]);
    if (probs)free(probs);
    if (pboxes)free(pboxes);
    return (void*)headdet_result;
}


/****************************************************************************/
/*******************     head detect postprocess       ********************/
void *post_process_head_detcetion(nn_output *pOut)
{
	void *data = NULL;
	data = postprocess_headdet(pOut);
	return data;
}

/****************************************************************************/
/*******************     face detect postprocess      ********************/



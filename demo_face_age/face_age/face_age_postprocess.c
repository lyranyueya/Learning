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
#include "face_age.h"
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

void* postprocess_age(nn_output *pout)
{
    float* buffer;
    unsigned int sz ;
    static face_age_out_t age_result;
    float age;
    float pred_age_s1[3];
    float pred_age_s2[3];
    float pred_age_s3[3];

    float local_s1[3];
    float local_s2[3];
    float local_s3[3];

    float delta_s1;
    float delta_s2;
    float delta_s3;

    unsigned int i,S1,S2,S3,lambda_local,lambda_d;
    float a,b,c;

    for (i=0;i<pout->num;i++)
    {
        buffer = (float *)pout->out[i].buf;
        sz= pout->out[i].size;
        switch (i)
        {
            case 0:
                memcpy(pred_age_s1,buffer,sz);
                break;
            case 1:
                memcpy(pred_age_s2,buffer,sz);
                break;
            case 2:
                memcpy(pred_age_s3,buffer,sz);
                break;
            case 3:
                memcpy(local_s1,buffer,sz);
                break;
            case 4:
                memcpy(local_s2,buffer,sz);
                break;
            case 5:
                memcpy(local_s3,buffer,sz);
                break;
            case 6:
                memcpy(&delta_s1,buffer,sz);
                break;
            case 7:
                memcpy(&delta_s2,buffer,sz);
                break;
            case 8:
                memcpy(&delta_s3,buffer,sz);
                break;
            default:
                break;
        }
    }
    S1 = 3;
    S2 = 3;
    S3 = 3;
    lambda_local = 1;
    lambda_d = 1;
    a = 0;
    b = 0;
    c = 0;

    for (i = 0;i < 3;i++)
    {
        a = a + (i + lambda_local * local_s1[i]) * pred_age_s1[i];
    }
    a = a /(S1*(1 + lambda_d * delta_s1));
    for (i = 0;i < S2;i++)
    {
        b = b + (i + lambda_local * local_s2[i]) * pred_age_s2[i];
    }
    b = b /(S1*(1 + lambda_d * delta_s1)) / (S2*(1+lambda_d*delta_s2)) ;

    for (i = 0;i < S3;i++)
    {
        c = c + (i + lambda_local * local_s3[i]) * pred_age_s3[i];
    }
    c = c /(S1*(1 + lambda_d * delta_s1)) / (S2*(1+lambda_d*delta_s2)) / (S3*(1+lambda_d*delta_s3));

    age = (a+b+c)*101;
    age_result.age = (int)age;

    return (void*)&age_result;
}
void *post_process_face_age(nn_output *pOut)
{
	void *data = NULL;
	data = postprocess_age(pOut);
	return data;
}

/****************************************************************************/
/*******************     face detect postprocess      ********************/



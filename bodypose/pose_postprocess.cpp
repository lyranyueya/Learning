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
#include <iostream>
#include <vector>
#include <unistd.h>
#include<sys/types.h>
#include<fcntl.h>



#include <algorithm>
#include <numeric>

#define USE_OPENCV
#ifdef USE_OPENCV
#include<opencv2/dnn.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/highgui.hpp>
using namespace cv;
using namespace cv::dnn;
#endif

#define _BASETSD_H
#define NN_TENSOR_MAX_DIMENSION_NUMBER 4
#define X_MAX 639.0 / 640.0
#define Y_MAX 383.0 / 384.0
/*
#include <opencv2/opencv.hpp>

const float BOX_THRESHOLD         = 0.6;
const float BOX_NMS               = 0.45;
const int   NN_BUDGET             = 50; //10;
const float MAX_COSINE_DISTANCE   = 0.4; //0.2;
const int   MAX_AGE               = 100; //50;
const int   N_INIT                = 10; //3;
const float MAX_IOU_DISTANCE      = 0.7;
const int   MAX_GAP_NUM                 = 1;     //计算欧式距离的人脸间隔数
const int   MAX_IOU_AGE                 = 10;
const float FACE_BULR_THRESHOLD         = 0.5;   //人脸模糊阈值
const float FACE_POSE_SCORE_THRESHOLD   = 0.3;   //人脸评分阈值
const float FACE_MAX_COSINE_DISTANCE    = 0.5;   //人脸特征最大余弦距离
const int   AGE_ASSIGN_UID              = 20;    //分配uid的帧数
const int   AGE_RECALL_UID              = 10;    //丢失的帧数，在这个范围以外，可以参与被找回，这个参数要小于MAX_AGE
const int   MAX_MAP_SIZE                = 5000;
// const bool  PERSON_FLAG              = true;  //判断是行人跟踪还是人脸跟踪，默认行人跟踪
enum TRACK_TYPE { PERSON= 1, FACE};              //判断跟踪的类型


static const float HEAT_MAP_POINTS_THRES = .1;
static const float PAF_MAP_THRES = 0.1;
static const int   RESIZE_OFFSET_X = 8;
static const int   RESIZE_OFFSET_Y = 8;//(RESIZE_OFFSET_X * POSE_WIDTH)/POSE_HEIGHT;

static const int limbSeq[13][2] = { {2, 3},  {2, 6},  {3, 4}, {4, 5}, {6, 7},
                                    {7, 8},  {2, 9}, {9, 10}, {10, 11}, {2, 12},
                                    {12, 13}, {13, 14}, {2, 1}};

// the middle joints heatmap correpondence
static const int mapIdx[13][2] = { {31, 32}, {37, 39}, {33, 34}, {35, 36}, {39,40},
                                   {41, 42}, {19, 20}, {21, 22}, {23, 24}, {25, 26},
                                   {27, 28}, {29, 30}, {43, 44} };

typedef std::tuple<cv::Point, float, int, int> Peak;
typedef std::vector<Peak> Peaks;
typedef std::array<double, 5> Connection_member;
typedef std::vector<Connection_member> Connection;
typedef std::tuple<int, int, double, double> Connection_Candidate;
*/
using namespace std;

/*-------------------------------------------
                  Functions
-------------------------------------------*/
const int YOLOV3_PERSON_DETECT_WIDTH  = 512;
const int YOLOV3_PERSON_DETECT_HEIGHT = 288;
const int YOLOV3_PERSON_DETECT_CHANNEL = 3;

static const int src_w = YOLOV3_PERSON_DETECT_WIDTH;
static const int src_h = YOLOV3_PERSON_DETECT_HEIGHT;
static const int network_input_w = YOLOV3_PERSON_DETECT_WIDTH;
static const int network_input_h = YOLOV3_PERSON_DETECT_HEIGHT;
static const int layerN = 3;
static const int classes = 2;
static const int coords = 0;
static const int relative = 1;

static const int small_buf_w = network_input_w / 32;
static const int small_buf_h = network_input_h / 32;
static const int medium_buf_w = network_input_w / 16;
static const int medium_buf_h = network_input_h / 16;
static const int big_buf_w = network_input_w / 8;
static const int big_buf_h = network_input_h / 8;

static const int small_buf_size = small_buf_w * small_buf_h * (3*(classes+5));
static const int medium_buf_size = medium_buf_w * medium_buf_h * (3*(classes+5));
static const int big_buf_size = big_buf_w * big_buf_h * (3*(classes+5));

static const int small_layer_size = small_buf_w * small_buf_h;
static const int medium_layer_size = medium_buf_w * medium_buf_h;
static const int big_layer_size = big_buf_w * big_buf_h;
static const int THREAD_NUM = 3;

extern int g_detect_number;

static float anchor[18] = { 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326 };

/*******************    simple pose begin    ************************/
float heat_maps[15][27][48];
float paf_maps[26][27][48];

float *heatmaps = NULL;
float *pafmaps = NULL;

const int POSE_OUT_DIM_W = 48;
const int POSE_OUT_DIM_H = 27;

#if 0
typedef struct __nn_body_keypoints{
    cv::Point  point;
    float    score;
    int      map;
}body_keypoints;

typedef struct __nn_one_pose_detect
{
    body_keypoints                one_pose_keypoints[15];
    int                           uid;             //人体跟踪id,
    int                           track_id;
    cv::Rect2d                    person_rect;     //人体矩形框坐标
    float                         paf_map[48 * 27 * 26];
}one_pose_detect_t;

typedef struct __nn_simple_pose_detect
{
    int                           detNum;
    one_pose_detect_t*            pose;
    
}simple_pose_detect_t;

static void linspace(double x1, double x2, int n, std::vector<double> &result)
{
    double d = (x2 - x1) / (n - 1);
    for (int i = 0; i < n; ++i) {
        result.push_back(x1 + i * d);
    }
}

static int zip(std::vector<double> &src1, std::vector<double> &src2,
               std::vector<std::array<double, 2>> &result)
{
    if (src1.size() != src2.size()) {
        LOGE("src1.size() != src2.size()");
        return 0;
    }
    for (unsigned int i = 0; i < src1.size(); ++i) {
        std::array<double, 2> tmp;
        tmp[0] = src1[i];
        tmp[1] = src2[i];
        result.push_back(tmp);
    }
    return 1;
}

static std::vector<float> operator*(std::vector<float> &vec, double n)
{
    std::vector<float> result;
    for (auto start = vec.begin(); start != vec.end(); start++) {
        result.push_back(*start * n);
    }
    return result;
}

static std::vector<float> operator+(std::vector<float> vec1, std::vector<float> vec2)
{
    std::vector<float> result;
    if (vec1.size() != vec2.size()) {
        LOGE("vec1.size() != vec2.size()");
        return vec1;
    }
    for (unsigned int i = 0; i < vec1.size(); ++i) {
        result.push_back(vec1[i] + vec2[i]);
    }
    return result;
}

template<class T>
static std::array<T,20> operator>=(std::array<T,20> src, T parame)
{
    std::array<T,20> result;
    result.fill(0);
    for(auto i = 0; i < 20; ++i){
        result[i] = src[i] >= T(parame);
    }
    return result;
}

static int num_greater_than(std::vector<float> src, double param)
{
    int num = 0;
    for (unsigned int  i = 0; i < src.size(); ++i) {
        if (src[i] > param) {
            ++num;
        }
    }
    return num;
}

static bool not_in_connection(int i, int j, Connection connection)
{
    for(auto tmp : connection){
        if(i == tmp[3] || j == tmp[4]){
            return false;
        }
    }
    return true;
}

template<class T>
static bool not_in_vector(T i, std::vector<T> v)
{
    for(auto start = v.begin(); start != v.end(); start++){
        if(i==*start)
            return false;
    }
    return true;
}

static void splitbuffer(vector<cv::Mat> &mat,float *buf, int cols, int rows, int channels)
{
    for(int i = 0; i < channels; ++i){
        mat[i] = cv::Mat(rows,cols,CV_32F,buf + i * rows * cols);
    }
}

static cv::Point modify_result(cv::Mat &heat_map, cv::Point point)
{
    int tmp_x = point.x * 8;
    int tmp_y = point.y * 8;
    int offset_x = point.x;
    int offset_y = point.y;
    cv::Mat resize_img = heat_map;
    float tmp = resize_img.at<float>(tmp_y, tmp_x);
    for (int cols = -8; cols < 8; cols++) {
        for (int rows = -8; rows < 8; rows++) {
            int x = (offset_x * 8 + cols) < 0 ? 0 : (offset_x * 8 + cols);
            x = x >= resize_img.cols ? resize_img.cols-1 : x;
            int y = (offset_y * 8 + rows) < 0 ? 0 : (offset_y * 8 + rows);
            y = y >= resize_img.rows ? resize_img.rows-1 : y;

            int x_1 = x - 1 < 0 ? 0 : x - 1;
            x_1 = x_1 >= resize_img.cols ? resize_img.cols-1 : x_1;

            int x_11 = x + 1 < 0 ? 0 : x + 1;
            x_11 = x_11 >= resize_img.cols ? resize_img.cols-1 : x_11;

            int y_1 = y - 1 < 0 ? 0 : y - 1;
            y_1 = y_1 >= resize_img.rows ? resize_img.rows-1 : y_1;

            int y_11 = y + 1 < 0 ? 0 : y + 1;
            y_11 = y_11 >= resize_img.rows ? resize_img.rows-1 : y_11;

            if (resize_img.at<float>(y, x) >= resize_img.at<float>(y_1, x) &&
                resize_img.at<float>(y, x) >= resize_img.at<float>(y_11, x) &&
                resize_img.at<float>(y, x) >= resize_img.at<float>(y, x_1) &&
                resize_img.at<float>(y, x) >= resize_img.at<float>(y, x_11)) {

                if (resize_img.at<float>(y, x) > tmp) {
                    tmp = resize_img.at<float>(y, x);
                    tmp_x = point.x * 8 + cols;
                    tmp_x = tmp_x < 0 ? 0 : tmp_x;
                    tmp_x = tmp_x >= heat_map.cols * 8 ? heat_map.cols * 8 - 1: tmp_x;
                    tmp_y = point.y * 8 + rows;
                    tmp_y = tmp_y < 0 ? 0 : tmp_y;
                    tmp_y = tmp_y >= heat_map.rows * 8 ? heat_map.rows * 8 - 1: tmp_y;
                }
            }
        }
    }
    cv::Point result = cv::Point(tmp_x,tmp_y);
    return result;
}

static int mat_process(simple_pose_detect_t* result, vector<cv::Mat> &heat_map, vector<cv::Mat> &paf_map, float* pafmap) 
{
    vector<Peaks> all_peaks;
    int peak_counter = 0;
    // #pragma omp parallel for result
    for(unsigned int i = 0; i < 14; ++i) {
        Peaks peaks;

        for(int rows = 1; rows < heat_map[i].rows -1; ++rows) {
            for (int cols = 1; cols < heat_map[i].cols - 1; ++cols) {
                float tmp = heat_map[i].at<float>(rows, cols);
                if (tmp > HEAT_MAP_POINTS_THRES) {
                    if (tmp >= heat_map[i].at<float>(rows - 1, cols)
                        && tmp >= heat_map[i].at<float>(rows + 1, cols)
                        && tmp >= heat_map[i].at<float>(rows, cols - 1)
                        && tmp >= heat_map[i].at<float>(rows, cols + 1)) {
                        Peak p(cv::Point(cols, rows), tmp, peak_counter,i);
                        peaks.push_back(p);
                        ++peak_counter;
                    }
                }
            }
        }

        all_peaks.push_back(peaks);
    }

    std::vector<Connection> connection_all;
    std::vector<int> special_k;
    int mid_num = 10;

    // #pragma omp parallel for
    for(int k = 0; k < 13; ++k) {
        cv::Mat score_mid1 = paf_map[mapIdx[k][0] - 19];
        cv::Mat score_mid2 = paf_map[mapIdx[k][1] - 19];
        Peaks candA = all_peaks[limbSeq[k][0] - 1];
        Peaks candB = all_peaks[limbSeq[k][1] - 1];
        auto nA = candA.size();
        auto nB = candB.size();

        if (nA != 0 && nB != 0) {
            vector<Connection_Candidate> connection_candidate;
            for (unsigned int i = 0; i < nA; ++i) {
                for (unsigned int j = 0; j < nB; ++j) {
                    double vec[2];
                    vec[0] = std::get<0>(candB[j]).x - std::get<0>(candA[i]).x;
                    vec[1] = std::get<0>(candB[j]).y - std::get<0>(candA[i]).y;
                    double norm = sqrt(pow(vec[0], 2) + pow(vec[1], 2));
                    vec[0] /= norm;
                    vec[1] /= norm;
                    vector<double> line1, line2;
                    vector<std::array<double, 2>> startend;
                    linspace(std::get<0>(candA[i]).x, std::get<0>(candB[j]).x, mid_num, line1);
                    linspace(std::get<0>(candA[i]).y, std::get<0>(candB[j]).y, mid_num, line2);
                    int ret = zip(line1, line2, startend);
                    if(!ret) {
                        return ret;
                    }
                    vector<float> vec_x, vec_y;
                    for(unsigned int I = 0; I < startend.size(); ++I){
                        int rows, cols;
                        rows = int(round(startend[I][1]));
                        cols = int(round(startend[I][0]));
                        vec_x.push_back(score_mid1.at<float>(rows, cols));
                        vec_y.push_back(score_mid2.at<float>(rows, cols));
                    }
                    std::vector<float> score_midpts = vec_x * vec[0] + vec_y * vec[1];
                    double score_with_dist_prior = std::accumulate(score_midpts.begin(),
                                                                   score_midpts.end(),
                                                                   double(0)) /score_midpts.size()
                                                   + (0.5 * 384 / norm - 1 < double(0) ? 0.5 * 216 / norm - 1 : double(0));
                    bool criterion1, criterion2;
                    criterion1 = num_greater_than(score_midpts, PAF_MAP_THRES) > (0.4 * score_midpts.size());
                    criterion2 = score_with_dist_prior > 0;
                    if (criterion1 && criterion2) {
                        Connection_Candidate tmp(i, j, score_with_dist_prior,
                                                 score_with_dist_prior
                                                 + std::get<1>(candA[i])
                                                 + std::get<1>(candB[j]));
                        connection_candidate.push_back(tmp);
                    }
                }
            }
            std::sort(connection_candidate.begin(), connection_candidate.end(),
                      [](Connection_Candidate a, Connection_Candidate b) {
                          return std::get<2>(a) > std::get<2>(b);});
            Connection connection;
            for (auto c : connection_candidate) {
                int i, j;
                double s;
                i = std::get<0>(c);
                j = std::get<1>(c);
                s = std::get<2>(c);
                if(not_in_connection(i, j, connection)){
                    Connection_member mem;
                    mem[0] = std::get<2>(candA[i]);
                    mem[1] = std::get<2>(candB[j]);
                    mem[2] = s;
                    mem[3] = i;
                    mem[4] = j;
                    connection.push_back(mem);
                    if(connection.size() >= MIN(nA,nB)){
                        break;
                    }
                }
            }
            connection_all.push_back(connection);
        }
        else{
            Connection connection;
            special_k.push_back(k);
            connection_all.push_back(connection);
        }
    }
    //todo: subset maybe subset * -1
    std::vector<std::array<double,20>> subset;
    std::vector<Peak> candidate;
    for(auto start : all_peaks){
        for(auto s : start){
            candidate.push_back(s);
        }
    }

    // #pragma omp parallel for
    for(int k = 0; k < 13; ++k) {
        if(not_in_vector(k,special_k)){
            std::vector<double> partAs, partBs;
            for(auto start : connection_all[k]){

                partAs.push_back(start[0]);
                partBs.push_back(start[1]);

            }
            int indexA = limbSeq[k][0] - 1;
            int indexB = limbSeq[k][1] - 1;

            auto tmp_connection_size = connection_all[k].size();
            for(unsigned int i = 0; i < tmp_connection_size; ++i){
                int found = 0;
                int subset_idx[2] = {-1, -1};
                for(unsigned int j = 0; j < subset.size(); ++j){
                    if (subset[j][indexA] == partAs[i] || subset[j][indexB] == partBs[i]){
                        subset_idx[found] = j;
                        ++found;
                    }
                }

                if(found == 1){
                    int j = subset_idx[0];
                    if(subset[j][indexB] != partBs[i]){
                        subset[j][indexB] = partBs[i];
                        subset[j][subset[j].size()-1] += 1;
                        subset[j][subset[j].size()-2] += std::get<1>(candidate[partBs[i]])
                                                        + connection_all[k][i][2];
                    }
                }
                else if(found == 2){
                    //NC_MESSAGE("Warning found == 2 ...");
                    int j1 = subset_idx[0];
                    int j2 = subset_idx[1];
                    std::array<double,20> arr_j1, arr_j2;
                    std::array<int,18> membership;
                    int over2_num = 0;
                    membership.fill(0);
                    arr_j1 = subset[j1];
                    arr_j2 = subset[j2];
                    for(int t = 0; t < 14; ++t){
                        membership[t] = int(arr_j1[t] >= double(0)) + int(arr_j2[t] >= double(0));
                        //NC_MESSAGE("membership[%s]: %d", std::to_string(t).c_str(), membership[t]);
                        if(membership[t] == 2)
                            over2_num++;
                    }
                    if(over2_num == 0){
                        for(unsigned int t = 0; t < subset[j1].size() - 2; ++t){
                            subset[j1][t] += (subset[j2][t] + 1);
                        }
                        subset[j1][subset[j1].size()-1] += subset[j2][subset[j2].size()-1];
                        subset[j1][subset[j1].size()-2] += subset[j2][subset[j2].size()-2];
                        subset[j1][subset[j1].size()-2] += connection_all[k][i][2];
                        subset.erase(subset.begin()+j2);
                    }
                    else{
                        subset[j1][indexB] = partBs[i];
                        subset[j1][subset[j1].size()-1] += 1;
                        subset[j1][subset[j1].size()-2] += std::get<1>(candidate[partBs[i]])
                                                           + connection_all[k][i][2];
                    }
                }
                else if(!found && k < 17){
                    //todo: maybe row * -1
                    std::array<double,20> row;
                    row.fill(-1);
                    row[indexA] = partAs[i];
                    row[indexB] = partBs[i];
                    row[row.size() - 1] = 2;
                    row[row.size() - 2] = std::get<1>(candidate[connection_all[k][i][0]])
                                          + std::get<1>(candidate[connection_all[k][i][1]])
                                          + connection_all[k][i][2];
                    subset.push_back(row);
                }
            }
        }
    }
    // #pragma omp parallel for
    for (unsigned int n = 0; n < subset.size(); ++n)
    {
        simple_pose_detect_t* person_pose;

        int keypoint_num = 0;
        for(int i = 0; i < 14; ++i){
            int index = subset[n][i];

            if(index == -1){
                person_pose->pose[n].one_pose_keypoints[i].point = cv::Point(0,0);
                person_pose->pose[n].one_pose_keypoints[i].score = 0.0f;
                person_pose->pose[n].one_pose_keypoints[i].map = -1;
                continue;
            }
            cv::Point p(std::get<0>(candidate[index]).x, std::get<0>(candidate[index]).y);
            person_pose->pose[n].one_pose_keypoints[i].point = p;
            person_pose->pose[n].one_pose_keypoints[i].score = std::get<1>(candidate[index]);
            person_pose->pose[n].one_pose_keypoints[i].map = std::get<3>(candidate[index]);
            keypoint_num++;
        }
        if(keypoint_num > 5) {
            for(int i = 0; i < 14; ++i)
            {
            //std::memcpy(person_pose->pose.paf_map, pafmap, sizeof(float) * POSE_OUT_DIM_W * POSE_OUT_DIM_H * 26);
            //result.push_back(person_pose);
                result->pose[n].one_pose_keypoints[i].point = person_pose->pose[n].one_pose_keypoints[i].point;
                result->pose[n].one_pose_keypoints[i].score = person_pose->pose[n].one_pose_keypoints[i].score; 
                result->pose[n].one_pose_keypoints[i].map   = person_pose->pose[n].one_pose_keypoints[i].map;
            }
        }
    }

    return 1;
}
static void check_result(vector<cv::Mat> &heat_map, simple_pose_detect_t* result)
{
    for( int loop=0; loop<15; loop++) {
        //cv::INTER_NEAREST//INTER_CUBIC
        cv::resize(heat_map[loop], heat_map[loop], cv::Size(0, 0), 8.0, 8.0, cv::INTER_CUBIC);
    }

    for (size_t i = 0; i < result.size(); i++) {
        for (int j = 0; j < 14; ++j) {
            if (result[i].one_pose_keypoints[j].map >= 0) {
                cv::Point p = modify_result(heat_map[result[i].one_pose_keypoints[j].map],
                                            result[i].one_pose_keypoints[j].point);
                result[i].one_pose_keypoints[j].point.x = p.x * img_scale;
                result[i].one_pose_keypoints[j].point.y = p.y * img_scale;
            }
            else {
                result[i].one_pose_keypoints[j].point.x = result[i].one_pose_keypoints[j].point.x * 8;
                result[i].one_pose_keypoints[j].point.y = result[i].one_pose_keypoints[j].point.y * 8;
            }
        }

        cv::Point nose = result[i].one_pose_keypoints[0].point;
        cv::Point neck = result[i].one_pose_keypoints[1].point;
        if(nose.x > 0 && neck.x > 0) {
            cv::Point head;
            head.x = nose.x;
            head.y= nose.y + 0.6 * (nose.y - neck.y);
            result[i].one_pose_keypoints[14].point = head;
            result[i].one_pose_keypoints[14].score =  0.4 * result[i].one_pose_keypoints[0].score + 0.6 * result[i].one_pose_keypoints[1].score;
            result[i].one_pose_keypoints[14].map = 1;
        }

        result[i].track_id = -1;
        result[i].uid = -1;
    }
}

#endif

/*******************    simple pose end   ************************/

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

/******
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

*********/


/*******************   person detect **************************

static int person_det_nms_comparator(const void *pa, const void *pb)
{
    detection a = *(detection *)pa;
    detection b = *(detection *)pb;
    float diff = 0;
    if (b.sort_class >= 0) {
        diff = a.prob[b.sort_class] - b.prob[b.sort_class];
    }
    else {
        diff = a.objectness - b.objectness;
    }
    if (diff < 0) return 1;
    else if (diff > 0) return -1;
    return 0;
}

static void person_det_do_nms_sort(detection *dets, int total, int classes, float thresh)
{
    int i, j, k;
    k = total - 1;
    for (i = 0; i <= k; ++i)
    {
        if (dets[i].objectness == 0)
        {
            detection swap = dets[i];
            dets[i] = dets[k];
            dets[k] = swap;
            --k;
            --i;
        }
    }
    total = k + 1;

    for (k = 0; k < classes; ++k)
    {
        for (i = 0; i < total; ++i)
        {
            dets[i].sort_class = k;
        }
        qsort(dets, total, sizeof(detection), person_det_nms_comparator);
        for (i = 0; i < total; ++i)
        {
            if (dets[i].prob[k] == 0) continue;
            box a = dets[i].bbox;
            for (j = i + 1; j < total; ++j)
            {
                box b = dets[j].bbox;
                if (box_iou(a, b) > thresh)
                {
                    dets[j].prob[k] = 0;
                }
            }
        }
    }
}

static box get_yolo_box(float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride)
{
    box b;
    b.x = (i + x[index + 0 * stride]) / lw;
    b.y = (j + x[index + 1 * stride]) / lh;
    b.w = exp(x[index + 2 * stride]) * biases[2 * n] / w;
    b.h = exp(x[index + 3 * stride]) * biases[2 * n + 1] / h;
    return b;
}

static void get_yolo_detections(int idx_layer, float *predictions, int layerW, int layerH, int layerN, int classes, int iOutputs,
        float *anchor, int *mask,int w, int h, int netw, int neth, float thresh, int relative, /*detection/vector<detection*> &dets)
{
    int i, j;

    for (i = 0; i < layerW*layerH; ++i)
    {
        int row = i / layerW;
        int col = i % layerW;
        int n = idx_layer;
        int obj_index = entry_index(layerW, layerH, classes, iOutputs, 0, n*layerW*layerH + i, 4);
        float objectness = predictions[obj_index];
        if (objectness <= thresh) continue;
        detection *dets1 = (detection*)calloc(1, sizeof(detection));
        dets1->prob = (float*)calloc(classes, sizeof(float));
        int box_index = entry_index(layerW, layerH, classes, iOutputs, 0, n*layerW*layerH + i, 0);

        dets1->bbox = get_yolo_box(predictions, anchor, mask[n], box_index, col, row, layerW, layerH, netw, neth, layerW*layerH);
        dets1->objectness = objectness;
        dets1->classes = classes;
        for (j = 0; j < classes; ++j)
        {
            int class_index = entry_index(layerW, layerH, classes, iOutputs, 0, n*layerW*layerH + i, 4 + 1 + j);
            float prob = objectness * predictions[class_index];
            dets1->prob[j] = (prob > thresh) ? prob : 0;
        }
        dets.push_back(dets1);
    }
}

void* postprocess_aml_person_detect(nn_output *pout)
{
    unsigned int j;
    unsigned int sz;
    float *buffer = NULL;

    static aml_person_detect_out_t aml_person_detect_result;
    memset(&aml_person_detect_result,0,sizeof(point_t));

    //float *buf13x13 = (float*)malloc(small_buf_w*small_buf_h*21*sizeof(float));

    float *buf13x13 = (float*)malloc(small_buf_w * small_buf_h * 21 * sizeof(float));
    float *buf26x26 = (float*)malloc(medium_buf_w * medium_buf_h * 21 * sizeof(float));
    float *buf52x52 = (float*)malloc(big_buf_w * big_buf_h * 21 * sizeof(float));

    for (j = 0; j < pout->num; j++)
    {
        buffer = (float *)pout->out[j].buf;
        sz= pout->out[j].size;

        switch (j)
        {
            case 0:
                memcpy(buf13x13,buffer,sz);
                break;
            case 1:
                memcpy(buf26x26,buffer,sz);
                break;
            case 2:
                memcpy(buf52x52,buffer,sz);
                break;
            default:
                break;
        }
    }

    float *small_buf = buf13x13;
    float *medium_buf = buf26x26;
    float *big_buf = buf52x52;
    vector<detection*> dets;

    detection *dets_sum = NULL;

    int mask[3];

    float nms = 0.5;
    float thresh = 0.5;

    int index = 0;

    for (int i = 0; i < layerN; ++i)
    {
        //small featrue map activate
        if (small_buf != NULL)
        {
            index = i * small_layer_size * (4 + classes + 1);
            activate_array(small_buf + index, 2 * small_layer_size);
            index = i * small_layer_size * (4+classes+1) + 4 * small_layer_size;
            activate_array(small_buf + index, (1+classes) * small_layer_size);

            mask[0] = 6;
            mask[1] = 7;
            mask[2] = 8;
            get_yolo_detections(i,small_buf, small_buf_w, small_buf_h, layerN, classes, small_buf_size,
            anchor, mask, src_w, src_h, network_input_w, network_input_h, thresh, relative, dets);

        }

        //medium feature map activate
        if (medium_buf != NULL)
        {
            index = i * medium_layer_size * (4 + classes + 1);
            activate_array(medium_buf + index, 2 * medium_layer_size);
            index = i * medium_layer_size * (4+classes+1) + 4 * medium_layer_size;
            activate_array(medium_buf + index, (1+classes) * medium_layer_size);

            mask[0] = 3;
            mask[1] = 4;
            mask[2] = 5;
            get_yolo_detections(i,medium_buf, medium_buf_w, medium_buf_h, layerN, classes, medium_buf_size,
            anchor, mask, src_w, src_h, network_input_w, network_input_h, thresh, relative, dets);
        }

        //big feature map activate
        if (big_buf != NULL)
        {
            index = i * big_layer_size * (4 + classes + 1);
            activate_array(big_buf + index, 2 * big_layer_size);
            index = i * big_layer_size * (4+classes+1) + 4 * big_layer_size;
            activate_array(big_buf + index, (1+classes) * big_layer_size);

            mask[0] = 0;
            mask[1] = 1;
            mask[2] = 2;
            get_yolo_detections(i,big_buf, big_buf_w, big_buf_h, layerN, classes, big_buf_size,
                        anchor, mask, src_w, src_h, network_input_w, network_input_h, thresh, relative, dets);
        }
    }

    unsigned int total = dets.size();

    dets_sum = (detection*)malloc((total) * sizeof(detection));
    for (unsigned int i = 0; i < total; i++)
    {
        memcpy(dets_sum + i, dets[i], sizeof(detection));
    }

    person_det_do_nms_sort(dets_sum, (int)total, classes, nms);
    int idx = 0;
    for (unsigned int i = 0; i < total; ++i)
    {
        for (unsigned int j = 0; j < classes; ++j)
        {
            if (j != 1)
                continue;
            if (dets_sum[i].prob[j] > thresh && idx < g_detect_number)
            {
                box b = dets_sum[i].bbox;
                aml_person_detect_result.pBox[idx].score = dets_sum[i].prob[j];
                aml_person_detect_result.pBox[idx].x = (b.x - b.w / 2);
                aml_person_detect_result.pBox[idx].y = (b.y - b.h / 2);
                aml_person_detect_result.pBox[idx].w = b.w;
                aml_person_detect_result.pBox[idx].h = b.h;
                idx++;
                //result.push_back(r);
            }
        }
    }
    aml_person_detect_result.detNum = idx;

    for (unsigned int i = 0; i < dets.size(); i++)
    {
        if (dets[i])
        {
            if (dets[i]->prob)
            {
                free(dets[i]->prob);
                dets[i]->prob = NULL;
            }
            free(dets[i]);
            dets[i] = NULL;
        }
    }

    if (dets_sum)
    {
        free(dets_sum);
        dets_sum = NULL;
    }
    if (buf26x26)
    {
        free(buf26x26);
        buf26x26 = NULL;
    }
    if (buf13x13)
    {
        free(buf13x13);
        buf13x13 = NULL;
    }

    if (buf52x52)
    {
        free(buf52x52);
        buf52x52 = NULL;
    }

    dets.clear();

    //aml_person_do_post_process(&aml_person_detect_result);
    return (void*)(&aml_person_detect_result);
}


*****************************************************************************/
/*****************************************************************************/
/*****************************************************************************/
/*****************************************************************************/
/*****************************************************************************

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




*******************     body pose detect postprocess      ********************/
/****************************************************************************/
#ifdef USE_OPENCV
extern "C"
struct SPKeyPoint{
    SPKeyPoint(Point point,float probability){
        this->id = -1;
        this->point = point;
        this->probability = probability;
    }
    int id;
    Point point;
    float probability;
};
const int nPoints = 18;
/*
const std::string keypointsMapping[] = {
    "Nose", "Neck",
    "R-Sho", "R-Elb", "R-Wr",
    "L-Sho", "L-Elb", "L-Wr",
    "R-Hip", "R-Knee", "R-Ank",
    "L-Hip", "L-Knee", "L-Ank",
    "R-Eye", "L-Eye", "R-Ear", "L-Ear"
};*/

void getKeyPoints(Mat& probMap, double thresholdval, std::vector<SPKeyPoint>& keyPoints)
{
    Mat smoothProbMap;
    GaussianBlur( probMap, smoothProbMap, Size( 3, 3 ), 0, 0 );

    Mat maskedProbMap;
    threshold(smoothProbMap,maskedProbMap,thresholdval,255,THRESH_BINARY);

    maskedProbMap.convertTo(maskedProbMap,CV_8U,1);

    std::vector<std::vector<Point> > contours;
    findContours(maskedProbMap,contours,RETR_TREE,CHAIN_APPROX_SIMPLE);

    for (unsigned int i = 0; i < contours.size();++i)
    {
        Mat blobMask = Mat::zeros(smoothProbMap.rows,smoothProbMap.cols,smoothProbMap.type());

        fillConvexPoly(blobMask,contours[i],Scalar(1));

        double maxVal;
        Point maxLoc;

        minMaxLoc(smoothProbMap.mul(blobMask),0,&maxVal,0,&maxLoc);

        keyPoints.push_back(SPKeyPoint(maxLoc, probMap.at<float>(maxLoc.y,maxLoc.x)));
    }
}

void splitNetOutputBlobToParts(const Size& targetSize,std::vector<Mat>& netOutputParts,float* bout)
{
    int nParts = 57;
    int h = 60;
    int w = 80;

    for (int i = 0; i< nParts;++i)
    {
        Mat part(h, w, CV_32F, (bout+4800*i)); //&output[i][0][0]
        Mat resizedPart;

        resize(part,resizedPart,targetSize);
        netOutputParts.push_back(resizedPart);
    }
}

extern "C"
void* post_posenet(float* pbody,unsigned int size)
{
    std::vector<Mat> netOutputParts;
    int keyPointId = 0;
    float prob;
    std::vector<std::vector<SPKeyPoint > > detectedKeypoints;
    std::vector<SPKeyPoint> keyPointsList;
    static body_pose_out_t *bodyout = NULL;
    if (bodyout == NULL)
    {
        bodyout = (body_pose_out_t*)malloc(sizeof(body_pose_out_t));
    }
    if (bodyout == NULL)
    {
        return NULL;
    }
    splitNetOutputBlobToParts(Size(640,480),netOutputParts,pbody);
    for (int i = 0; i < nPoints;++i)
    {
        std::vector<SPKeyPoint> keyPoints;
        getKeyPoints(netOutputParts[i],0.1,keyPoints);
        prob = 0;
        //std::cout << "Keypoints - " << keypointsMapping[i] << " : " << keyPoints << std::endl;
        for (unsigned int j = 0; j< keyPoints.size();++j,++keyPointId)
        {
            keyPoints[j].id = keyPointId;
            //printf("keypoint++\n");
            if (prob < keyPoints[j].probability)
            {
                //printf("i:%d,x:%d,y:%d,prob:%f\n",i,keyPoints[j].point.x,keyPoints[j].point.y,keyPoints[j].probability);
                bodyout->bpos[i].valid = 1;
                bodyout->bpos[i].pos.x = (float)keyPoints[j].point.x;
                bodyout->bpos[i].pos.y = (float)keyPoints[j].point.y;
                prob = keyPoints[j].probability;
            }
        }
        detectedKeypoints.push_back(keyPoints);
        keyPointsList.insert(keyPointsList.end(),keyPoints.begin(),keyPoints.end());
    }

    return (void*)bodyout;
}

void* postprocess_bodypose(nn_output *pout)
{
    float *bodyout = (float *)pout->out[0].buf;
    unsigned int sz = pout->out[0].size;
    return post_posenet(bodyout,sz);
}
#endif

/****************************************************************************/
/*******************     body pose detect postprocess      ********************/


/*******************     head detect postprocess      ********************/
/****************************************************************************/

/****************************************************************************/
/*******************     head detect postprocess       ********************/
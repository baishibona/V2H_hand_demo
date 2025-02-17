/***********************************************************************************************************************
* Copyright (C) 2023 Renesas Electronics Corporation. All rights reserved.
***********************************************************************************************************************/
/***********************************************************************************************************************
* File Name    : main.cpp
* Version      : 1.00
* Description  : RZ/V2H DRP-AI Sample Application for MMPose HRNet + Megvii-Base Detection YOLOX with MIPI/USB Camera
***********************************************************************************************************************/

/*****************************************
* Includes
******************************************/
/*DRPAI Driver Header*/
#include <linux/drpai.h>
/*Definition of Macros & other variables*/
#include "define.h"
#include "define_color.h"
/*USB camera control*/
#include "camera.h"
/*Image control*/
#include "image.h"
/*Wayland control*/
#include "wayland.h"
/*YOLOX Post-Processing*/
#include "box.h"
/*DRP-AI Control*/
#include "drpai_ctl.h"
/*Mutual exclusion*/
#include <mutex>
#include "spdlog/spdlog.h"
#include "spdlog/sinks/basic_file_sink.h"

#include <stdio.h>
#include <termios.h>
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>
#include <cmath>



using namespace std;
/*****************************************
* Global Variables
******************************************/
/*Multithreading*/
static sem_t terminate_req_sem;
static pthread_t ai_inf_thread;
static pthread_t kbhit_thread;
static pthread_t capture_thread;
static pthread_t img_thread;
static pthread_t hdmi_thread;
static mutex mtx;

/*Flags*/
static atomic<uint8_t> inference_start (0);
static atomic<uint8_t> img_obj_ready   (0);
static atomic<uint8_t> hdmi_obj_ready   (0);

/*Global Variables*/
static float drpai_output_buf0[num_inf_out];
static float drpai_output_buf1[INF_OUT_SIZE];
static uint64_t capture_address;
static uint8_t buf_id;
static Image img;

/*AI Inference for DRPAI*/
static int drpai_fd0 = -1;
static int drpai_fd1 = -1;
static drpai_handle_t *drpai_hdl0 = NULL;
static drpai_data_t drpai_data0;
static drpai_handle_t *drpai_hdl1 = NULL;
static drpai_data_t drpai_data1;
static double yolox_drpai_time = 0;
static double hrnet_drpai_time = 0;
#ifdef DISP_AI_FRAME_RATE
static double ai_fps = 0;
static double cap_fps = 0;
static double proc_time_capture = 0;
static uint32_t array_cap_time[30] = {1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000};
#endif /* DISP_AI_FRAME_RATE */
static double yolox_proc_time = 0;
static double hrnet_proc_time = 0;
static uint32_t disp_time = 0;
static uint32_t array_drp_time[30] = {1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000};
static uint32_t array_disp_time[30] = {1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000};
static int32_t drp_max_freq;
static int32_t drpai_freq;
static uint32_t ai_time = 0;
static double array_hrnet_drpai_time[NUM_MAX_PERSON];
static double array_hrnet_proc_time[NUM_MAX_PERSON];
static float hrnet_preds[NUM_OUTPUT_C][3];
static uint16_t id_x[NUM_OUTPUT_C][NUM_MAX_PERSON];
static uint16_t id_y[NUM_OUTPUT_C][NUM_MAX_PERSON];
static uint16_t id_x_local[NUM_OUTPUT_C][NUM_MAX_PERSON]; /*To be used only in Inference Threads*/
static uint16_t id_y_local[NUM_OUTPUT_C][NUM_MAX_PERSON]; /*To be used only in Inference Threads*/

static int16_t cropx[NUM_MAX_PERSON];
static int16_t cropy[NUM_MAX_PERSON];
static int16_t croph[NUM_MAX_PERSON];
static int16_t cropw[NUM_MAX_PERSON];
static float lowest_kpt_score[NUM_MAX_PERSON];
static float lowest_kpt_score_local[NUM_MAX_PERSON]; /*To be used only in Inference Threads*/

/*YOLOX*/
static uint32_t bcount = 0;
static uint32_t ppl_count_local = 0; /*To be used only in Inference Threads*/
static uint32_t ppl_count = 0;
static vector<detection> det_res;
static vector<detection> det_ppl;

static Wayland wayland;
static vector<detection> det;
static Camera* capture = NULL;

// Serial 
int serial_port;
// Hand gesture 
unsigned char ONE[] = {235, 144, 1, 15, 18, 206, 5, 0, 0, 0, 0, 0, 0, 192, 3, 45, 0, 0, 0, 229};
unsigned char TWO[] = {235, 144, 1, 15, 18, 206, 5, 0, 0, 0, 0, 192, 3, 192, 3, 45, 0, 0, 0, 168};
unsigned char THREE[] = {235, 144, 1, 15, 18, 206, 5, 192, 3, 192, 3, 192, 3, 167, 1, 133, 2, 0, 0, 109};
unsigned char FOUR[] = {235, 144, 1, 15, 18, 206, 5, 192, 3, 192, 3, 192, 3, 192, 3, 45, 0, 0, 0, 46};
unsigned char FIVE[] = {235, 144, 1, 15, 18, 206, 5, 192, 3, 192, 3, 192, 3, 192, 3, 177, 3, 165, 3, 93};
unsigned char DEX_HAND_CMD[20];

#define PI 3.14159265

/*****************************************
* Function Name : timedifference_msec
* Description   : compute the time differences in ms between two moments
* Arguments     : t0 = start time
*                 t1 = stop time
* Return value  : the time difference in ms
******************************************/
static double timedifference_msec(struct timespec t0, struct timespec t1)
{
    return (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1000000.0;
}

/*****************************************
* Function Name : wait_join
* Description   : waits for a fixed amount of time for the thread to exit
* Arguments     : p_join_thread = thread that the function waits for to Exit
*                 join_time = the timeout time for the thread for exiting
* Return value  : 0 if successful
*                 not 0 otherwise
******************************************/
static int8_t wait_join(pthread_t *p_join_thread, uint32_t join_time)
{
    int8_t ret_err;
    struct timespec join_timeout;
    ret_err = clock_gettime(CLOCK_REALTIME, &join_timeout);
    if ( 0 == ret_err )
    {
        join_timeout.tv_sec += join_time;
        ret_err = pthread_timedjoin_np(*p_join_thread, NULL, &join_timeout);
    }
    return ret_err;
}

/*****************************************
 * Function Name : ceil3
 * Description   : ceil num specifiy digit
 * Arguments     : num number
 *               : base ceil digit
 * Return value  : int32_t result
 ******************************************/
static int32_t ceil3(int32_t num, int32_t base)
{
    double x = (double)(num) / (double)(base);
    double y = ceil(x) * (double)(base);
    return (int32_t)(y);
}

/*****************************************
* Function Name : get_result
* Description   : Get DRP-AI Output from memory via DRP-AI Driver
* Arguments     : drpai_fd = file descriptor of DRP-AI Driver
*                 output_addr = memory start address of DRP-AI output
*                 output_size = output data size
* Return value  : 0 if succeeded
*                 not 0 otherwise
******************************************/
static int8_t get_result(int drpai_fd, uint64_t output_addr, uint32_t output_size)
{
    drpai_data_t drpai_data;
    float drpai_buf[BUF_SIZE];
    drpai_data.address = output_addr;
    drpai_data.size = output_size;
    int32_t i = 0;
    int8_t ret = 0;

    errno = 0;
    /* Assign the memory address and size to be read */
    ret = ioctl(drpai_fd, DRPAI_ASSIGN, &drpai_data);
    if (-1 == ret)
    {
        fprintf(stderr, "[ERROR] Failed to run DRPAI_ASSIGN: errno=%d\n", errno);
        return -1;
    }

    /* Read the memory via DRP-AI Driver and store the output to buffer */
    for (i = 0; i < (drpai_data.size/BUF_SIZE); i++)
    {
        errno = 0;
        ret = read(drpai_fd, drpai_buf, BUF_SIZE);
        if ( -1 == ret )
        {
            fprintf(stderr, "[ERROR] Failed to read via DRP-AI Driver: errno=%d\n", errno);
            return -1;
        }

        memcpy(&drpai_output_buf0[BUF_SIZE/sizeof(float)*i], drpai_buf, BUF_SIZE);
    }

    if ( 0 != (drpai_data.size % BUF_SIZE))
    {
        errno = 0;
        ret = read(drpai_fd, drpai_buf, (drpai_data.size % BUF_SIZE));
        if ( -1 == ret)
        {
            fprintf(stderr, "[ERROR] Failed to read via DRP-AI Driver: errno=%d\n", errno);
            return -1;
        }

        memcpy(&drpai_output_buf0[(drpai_data.size - (drpai_data.size%BUF_SIZE))/sizeof(float)], drpai_buf, (drpai_data.size % BUF_SIZE));
    }
    return 0;
}

/*****************************************
* Function Name : sigmoid
* Description   : Helper function for YOLO Post Processing
* Arguments     : x = input argument for the calculation
* Return value  : sigmoid result of input x
******************************************/
static double sigmoid(double x)
{
    return 1.0/(1.0 + exp(-x));
}

/*****************************************
* Function Name : softmax
* Description   : Helper function for YOLO Post Processing
* Arguments     : val[] = array to be computed softmax
* Return value  : -
******************************************/
static void softmax(float val[NUM_CLASS])
{
    float max_num = -FLT_MAX;
    float sum = 0;
    int32_t i;
    for ( i = 0 ; i<NUM_CLASS ; i++ )
    {
        max_num = max(max_num, val[i]);
    }

    for ( i = 0 ; i<NUM_CLASS ; i++ )
    {
        val[i]= (float) exp(val[i] - max_num);
        sum+= val[i];
    }

    for ( i = 0 ; i<NUM_CLASS ; i++ )
    {
        val[i]= val[i]/sum;
    }
    return;
}

/*****************************************
* Function Name : index
* Description   : Get the index of the bounding box attributes based on the input offset.
* Arguments     : n = output layer number.
*                 offs = offset to access the bounding box attributesd.
*                 channel = channel to access each bounding box attribute.
* Return value  : index to access the bounding box attribute.
******************************************/
int32_t index(uint8_t n, int32_t offs, int32_t channel)
{
    uint8_t num_grid = num_grids[n];
    return offs + channel * num_grid * num_grid;
}

/*****************************************
* Function Name : offset
* Description   : Get the offset nuber to access the bounding box attributes
*                 To get the actual value of bounding box attributes, use index() after this function.
* Arguments     : n = output layer number [0~2].
*                 b = Number to indicate which bounding box in the region [0~2]
*                 y = Number to indicate which region [0~13]
*                 x = Number to indicate which region [0~13]
* Return value  : offset to access the bounding box attributes.
******************************************/
int32_t offset(uint8_t n, int32_t b, int32_t y, int32_t x)
{
    uint8_t num = num_grids[n];
    uint32_t prev_layer_num = 0;
    int32_t i = 0;

    for (i = 0 ; i < n; i++)
    {
        prev_layer_num += NUM_BB *(NUM_CLASS + 5)* num_grids[i] * num_grids[i];
    }
    return prev_layer_num + b *(NUM_CLASS + 5)* num * num + y * num + x;
}

/*****************************************
* Function Name : R_Post_Proc
* Description   : Process CPU post-processing for YoloX
* Arguments     : floatarr = drpai output address
*                 det = detected boxes details
*                 box_count = total number of boxes
* Return value  : -
******************************************/
static void R_Post_Proc(float* floatarr, vector<detection>& det, uint32_t* box_count)
{
    uint32_t count = 0;
    uint32_t BoundingBoxCount = 0;
    /*Memory Access*/
    /* Following variables are required for correct_region_boxes in Darknet implementation*/
    /* Note: This implementation refers to the "darknet detector test" */
    vector<detection> det_buff;
    float new_w, new_h;
    float correct_w = 1.;
    float correct_h = 1.;
    if ((float) (MODEL_IN_W / correct_w) < (float) (MODEL_IN_H/correct_h) )
    {
        new_w = (float) MODEL_IN_W;
        new_h = correct_h * MODEL_IN_W / correct_w;
    }
    else
    {
        new_w = correct_w * MODEL_IN_H / correct_h;
        new_h = MODEL_IN_H;
    }

    int32_t n = 0;
    int32_t b = 0;
    int32_t y = 0;
    int32_t x = 0;
    int32_t offs = 0;
    int32_t i = 0;
    float tx = 0;
    float ty = 0;
    float tw = 0;
    float th = 0;
    float tc = 0;
    float center_x = 0;
    float center_y = 0;
    float box_w = 0;
    float box_h = 0;
    float objectness = 0;
    uint8_t num_grid = 0;
    uint8_t anchor_offset = 0;
    float classes[NUM_CLASS];
    float max_pred = 0;
    int32_t pred_class = -1;
    float probability = 0;
    detection d;
    /* Clear the detected result list */
    det.clear();

    //YOLOX
    int stride = 0;
    vector<int> strides = {8, 16, 32};

    for (n = 0; n<NUM_INF_OUT_LAYER; n++)
    {
        num_grid = num_grids[n];
        anchor_offset = 2 * NUM_BB * (NUM_INF_OUT_LAYER - (n + 1));

        for (b = 0;b<NUM_BB;b++)
        {
           stride = strides[n];
            for (y = 0;y<num_grid;y++)
            {
                for (x = 0;x<num_grid;x++)
                {
                    offs = offset(n, b, y, x);
                    tc = floatarr[index(n, offs, 4)];

                    objectness = tc;

                    if (objectness > TH_PROB)
                    {
                        /* Get the class prediction */
                        for (i = 0;i < NUM_CLASS;i++)
                        {
                            classes[i] = floatarr[index(n, offs, 5+i)];
                        }

                        max_pred = 0;
                        pred_class = -1;
                        for (i = 0; i < NUM_CLASS; i++)
                        {
                            if (classes[i] > max_pred)
                            {
                                pred_class = i;
                                max_pred = classes[i];
                            }
                        }

                        /* Store the result into the list if the probability is more than the threshold */
                        probability = max_pred * objectness;
                        if (probability > TH_PROB)
                        {
                            if (pred_class == PERSON_LABEL_NUM)    //hand = 1
							{
                                tx = floatarr[offs];
                                ty = floatarr[index(n, offs, 1)];
                                tw = floatarr[index(n, offs, 2)];
                                th = floatarr[index(n, offs, 3)];

                                /* Compute the bounding box */
                                /*get_yolo_box/get_region_box in paper implementation*/
                                center_x = (tx+ float(x))* stride;
                                center_y = (ty+ float(y))* stride;
                                center_x = center_x  / (float) MODEL_IN_W;
                                center_y = center_y  / (float) MODEL_IN_H;
                                box_w = exp(tw) * stride;
                                box_h = exp(th) * stride;
                                box_w = box_w / (float) MODEL_IN_W;
                                box_h = box_h / (float) MODEL_IN_H;
                                
                                /* Adjustment for size */
                                /* correct_yolo/region_boxes */
                                center_x = (center_x - (MODEL_IN_W - new_w) / 2. / MODEL_IN_W) / ((float) new_w / MODEL_IN_W);
                                center_y = (center_y - (MODEL_IN_H - new_h) / 2. / MODEL_IN_H) / ((float) new_h / MODEL_IN_H);
                                box_w *= (float) (MODEL_IN_W / new_w);
                                box_h *= (float) (MODEL_IN_H / new_h);

                                center_x = round(center_x * DRPAI_IN_WIDTH*(DRPAI_IN_WIDTH/MODEL_IN_W));
                                center_y = round(center_y * DRPAI_IN_HEIGHT*(DRPAI_IN_HEIGHT/MODEL_IN_H));
                                box_w = round(box_w * DRPAI_IN_WIDTH*(DRPAI_IN_WIDTH/MODEL_IN_W));
                                box_h = round(box_h * DRPAI_IN_HEIGHT*(DRPAI_IN_HEIGHT/MODEL_IN_H));
                                
                                Box bb = {center_x, center_y, box_w, box_h};
                                d = {bb, pred_class, probability};
                                det_buff.push_back(d);
                                count++;
                            }
                            BoundingBoxCount++;
                        }
                    }
                }
            }
        }
    }
    /* Non-Maximum Supression filter */
    filter_boxes_nms(det_buff, det_buff.size(), TH_NMS);
    *box_count = count;
    /* Log Output */
    spdlog::info("YOLOX Result-------------------------------------");
    int iBoxCount=0;
    for(i = 0; i < det_buff.size(); i++)
    {
        /* Skip the overlapped bounding boxes */
        if (det_buff[i].prob == 0) continue;
        spdlog::info(" Bounding Box Number : {}",i+1);
        spdlog::info(" Bounding Box        : (X, Y, W, H) = ({}, {}, {}, {})", (int)det_buff[i].bbox.x, (int)det_buff[i].bbox.y, (int)det_buff[i].bbox.w, (int)det_buff[i].bbox.h);
        spdlog::info(" Detected Class      : {} (Class {})", label_file_map[det_buff[i].c].c_str(), det_buff[i].c);
        spdlog::info(" Probability         : {} %", (std::round((det_buff[i].prob*100) * 10) / 10));
        iBoxCount++;
    }
    spdlog::info(" Bounding Box Count  : {}", BoundingBoxCount);
    spdlog::info(" hand Count        : {}", iBoxCount);

    mtx.lock();
    /* Clear the detected result list */
    det.clear();
    copy(det_buff.begin(), det_buff.end(), back_inserter(det));
    mtx.unlock();
    return ;
}

/*****************************************
* Function Name : people_counter
* Description   : Function to count the real number of people detected and does not exceeds the maximum number
* Arguments     : det = detected boxes details
*                 ppl = detected people details
*                 box_count = total number of boxes
*                 ppl_count = actual number of people
* Return value  : -
******************************************/
static void people_counter(vector<detection>& det, vector<detection>& ppl, uint32_t box_count, uint32_t* ppl_count)
{
    mtx.lock();
    uint32_t count = 0;
    ppl.clear();
    for(uint32_t i = 0; i<box_count; i++)
    {
        if(0 == det[i].prob)
        {
            continue;
        }
        else
        {
            ppl.push_back(det[i]);
            count++;
            if(count > NUM_MAX_PERSON-1)
            {
                break;
            }
        }
    }
    *ppl_count = count;
    mtx.unlock();
}

/*****************************************
* Function Name : offset_hrnet
* Description   : Get the offset number to access the HRNet attributes
* Arguments     : b = Number to indicate which region [0~17]
*                 y = Number to indicate which region [0~64]
*                 x = Number to indicate which region [0~48]
* Return value  : offset to access the HRNet attributes.
*******************************************/
static int32_t offset_hrnet(int32_t b, int32_t y, int32_t x)
{
    return b * NUM_OUTPUT_W * NUM_OUTPUT_H + y * NUM_OUTPUT_W + x;
}

/*****************************************
* Function Name : sign
* Description   : Get the sign of the input value
* Arguments     : x = input value
* Return value  : returns the sign, 1 if positive -1 if not
*******************************************/
static int8_t sign(int32_t x)
{
    return x > 0 ? 1 : -1;
}

/*****************************************
* Function Name : R_Post_Proc_HRNet
* Description   : CPU post-processing for HRNet
*                 Microsoft COCO: Common Objects in Context' ECCV'2014
*                 More details can be found in the `paper
*                 <https://arxiv.org/abs/1405.0312>
*                 COCO Keypoint Indexes:
*                 0: 'nose',
*                 1: 'left_eye',
*                 2: 'right_eye',
*                 3: 'left_ear',
*                 4: 'right_ear',
*                 5: 'left_shoulder',
*                 6: 'right_shoulder',
*                 7: 'left_elbow',
*                 8: 'right_elbow',
*                 9: 'left_wrist',
*                 10: 'right_wrist',
*                 11: 'left_hip',
*                 12: 'right_hip',
*                 13: 'left_knee',
*                 14: 'right_knee',
*                 15: 'left_ankle',
*                 16: 'right_ankle'
* Arguments     : floatarr = drpai output address
*                 n_pers = number of the person detected
* Return value  : -
******************************************/
static void R_Post_Proc_HRNet(float* floatarr, uint8_t n_pers)
{
    mtx.lock();
    float score;
    int32_t b = 0;
    int32_t y = 0;
    int32_t x = 0;
    int32_t offs = 0;

    float center[] = {(float)(cropw[n_pers] / 2 -1), (float)(croph[n_pers] / 2 - 1)};
    int8_t ind_x = -1;
    int8_t ind_y = -1;
    float max_val = -1;
    float scale_x = 0;
    float scale_y = 0;
    float coords_x = 0;
    float coords_y = 0;
    float diff_x;
    float diff_y;
    int8_t i;
    
    for(b = 0; b < NUM_OUTPUT_C; b++)
    {
        float scale[] = {(float)(cropw[n_pers] / 200.0), (float)(croph[n_pers] / 200.0)};
        ind_x = -1;
        ind_y = -1;
        max_val = -1;
        for(y = 0; y < NUM_OUTPUT_H; y++)
        {
            for(x = 0; x < NUM_OUTPUT_W; x++)
            {
                offs = offset_hrnet(b, y, x);
                if (max_val < floatarr[offs])
                {
                    /*Update the maximum value and indices*/
                    max_val = floatarr[offs];
                    ind_x = x;
                    ind_y = y;
                }
            }
        }
        if (0 > max_val)
        {
            ind_x = -1;
            ind_y = -1;
            goto not_detect;
        }
        hrnet_preds[b][0] = float(ind_x);
        hrnet_preds[b][1] = float(ind_y);
        hrnet_preds[b][2] = max_val;
        offs = offset_hrnet(b, ind_y, ind_x);
        if ((ind_y > 1) && (ind_y < NUM_OUTPUT_H -1))
        {
            if ((ind_x > 1) && (ind_x < (NUM_OUTPUT_W -1)))
            {
                diff_x = floatarr[offs + 1] - floatarr[offs - 1];
                diff_y = floatarr[offs + NUM_OUTPUT_W] - floatarr[offs - NUM_OUTPUT_W];
                hrnet_preds[b][0] += sign(diff_x) * 0.25;
                hrnet_preds[b][1] += sign(diff_y) * 0.25;
            }
        }

        /*transform_preds*/
        scale[0] *= 200;
        scale[1] *= 200;
        //udp (Unbiased Data Processing) = False
        scale_x = scale[0] / (NUM_OUTPUT_W);
        scale_y = scale[1] / (NUM_OUTPUT_H);
        coords_x = hrnet_preds[b][0];
        coords_y = hrnet_preds[b][1];
        hrnet_preds[b][0] = (coords_x * scale_x) + center[0] - (scale[0] * 0.5);
        hrnet_preds[b][1] = (coords_y * scale_y) + center[1] - (scale[1] * 0.5);
    }
    /* Clear the score in preparation for the update. */
    lowest_kpt_score_local[n_pers] = 0;
    score = 1;
    for (i = 0; i < NUM_OUTPUT_C; i++)
    {
        /* Adopt the lowest score. */
        if (hrnet_preds[i][2] < score)
        {
            score = hrnet_preds[i][2];
        }
    }
    /* Update the score for display thread. */
    lowest_kpt_score_local[n_pers] = score;
    /* HRnet Logout. */
    spdlog::info("HRNet Result-------------------------------------");
    for (i = 0; i < NUM_OUTPUT_C; i++)
    {
        spdlog::info("  ID {}: ({}, {}): {}%", i, (std::round((hrnet_preds[i][0]) * 100) / 100), (std::round((hrnet_preds[i][1]) * 100) / 100), (std::round((hrnet_preds[i][2]*100) * 10) / 10));
    }
    goto end;

not_detect:
    lowest_kpt_score_local[n_pers] = 0;
    goto end;

end:
    mtx.unlock();
    return;
}

/*****************************************
* Function Name : R_HRNet_Coord_Convert
* Description   : Convert the post processing result into drawable coordinates
* Arguments     : n_pers = number of the detected person
* Return value  : -
******************************************/
static void R_HRNet_Coord_Convert(uint8_t n_pers)
{
    /* Render skeleton on image and print their details */
    int32_t posx;
    int32_t posy;
    int8_t i;
    mtx.lock();

    for (i = 0; i < NUM_OUTPUT_C; i++)
    {
#if (0) == INF_YOLOX_SKIP
        /* Conversion from input image coordinates to display image coordinates. */
        /* +0.5 is for rounding.*/
        posx = (int32_t)(hrnet_preds[i][0] + 0.5) + cropx[n_pers] + OUTPUT_ADJ_X;
        posy = (int32_t)(hrnet_preds[i][1] + 0.5) + cropy[n_pers] + OUTPUT_ADJ_Y;
        /* Make sure the coordinates are not off the screen. */
        posx = (posx < 0) ? 0 : posx;
        posx = (posx > IMREAD_IMAGE_WIDTH - KEY_POINT_SIZE -1 ) ? IMREAD_IMAGE_WIDTH -KEY_POINT_SIZE -1 : posx;
        posy = (posy < 0) ? 0 : posy;
        posy = (posy > IMREAD_IMAGE_HEIGHT -KEY_POINT_SIZE -1) ? IMREAD_IMAGE_HEIGHT -KEY_POINT_SIZE -1 : posy;
#else
        /* Conversion from input image coordinates to display image coordinates. */
        /* +0.5 is for rounding.                                                 */
        posx = (int32_t)(hrnet_preds[i][0] / CROPPED_IMAGE_WIDTH  * CROPPED_IMAGE_WIDTH  + 0.5) + OUTPUT_LEFT + OUTPUT_ADJ_X;
        posy = (int32_t)(hrnet_preds[i][1] / CROPPED_IMAGE_HEIGHT * CROPPED_IMAGE_HEIGHT + 0.5) + OUTPUT_TOP  + OUTPUT_ADJ_Y;
        /* Make sure the coordinates are not off the screen. */
        posx    = (posx < OUTPUT_LEFT) ? OUTPUT_LEFT : posx;
        posy    = (posy < OUTPUT_TOP)  ? OUTPUT_TOP  : posy;
        posx = (posx > OUTPUT_LEFT + CROPPED_IMAGE_WIDTH  - 1) ? (OUTPUT_LEFT + CROPPED_IMAGE_WIDTH   - 1) : posx;
        posy = (posy > OUTPUT_TOP  + CROPPED_IMAGE_HEIGHT - 1) ? (OUTPUT_TOP  + CROPPED_IMAGE_HEIGHT  - 1) : posy;
#endif
        id_x_local[i][n_pers] = posx;
        id_y_local[i][n_pers] = posy;
    }
    mtx.unlock();
    return;
}

/*****************************************
* Function Name : draw_skeleton
* Description   : Draw Complete Skeleton on image.
* Arguments     : -
* Return value  : -
******************************************/
static void draw_skeleton(void)
{
    int32_t sk_id;
    uint8_t v;
    uint8_t i;
    float   thre_kpt = TH_KPT;
#if (1) == INF_YOLOX_SKIP
    thre_kpt = TH_KPT_YOLOX_SKIP;
#endif

    mtx.lock();

#if (1) == INF_YOLOX_SKIP
    i=0;
    img.draw_rect(cropx[i], cropy[i], cropw[i], croph[i]-1, YELLOW_DATA);
    img.draw_rect(cropx[i]+1, cropy[i]+1, cropw[i]-2, croph[i]-3, YELLOW_DATA);
#endif

    if(ppl_count > 0)
    {
        for(i=0; i < ppl_count; i++)
        {    
            /*Check If All Key Points Were Detected: If Over Threshold, It will Draw Complete Skeleton*/
            if (lowest_kpt_score[i] > thre_kpt)
            {
                /* Draw limb */
                for (sk_id = 0; sk_id < NUM_LIMB; sk_id++)
                {
                    uint8_t sk[] = {skeleton[sk_id][0], skeleton[sk_id][1]};
                    int pos1[] = {id_x[sk[0]][i], id_y[sk[0]][i]};
                    int pos2[] = {id_x[sk[1]][i], id_y[sk[1]][i]};
                    
                    if ((0 < pos1[0]) && (MIPI_WIDTH > pos1[0])
                        && (0 < pos1[1]) && (MIPI_WIDTH > pos1[1]))
                    {
                        if ((0 < pos2[0]) && (MIPI_WIDTH > pos2[0])
                            && (0 < pos2[1]) && (MIPI_WIDTH > pos2[1]))
                        {
                            img.draw_line2(pos1[0], pos1[1], pos2[0],pos2[1], YELLOW_DATA);
                        }
                    }
                }
        
                /*Draw Rectangle As Key Points*/
                for(v = 0; v < NUM_OUTPUT_C; v++)
                {
                    /*Draw Rectangles On Each Skeleton Key Points*/
                    img.draw_rect(id_x[v][i], id_y[v][i], KEY_POINT_SIZE, KEY_POINT_SIZE, RED_DATA);
                    img.draw_rect(id_x[v][i], id_y[v][i], KEY_POINT_SIZE+1, KEY_POINT_SIZE+1, RED_DATA);
                }
            }
        }
    }
    mtx.unlock();
    return;
}

/*****************************************
* Function Name : draw_bounding_box
* Description   : Draw bounding box on image.
* Arguments     : -
* Return value  : 0 if succeeded
*               not 0 otherwise
******************************************/
void draw_bounding_box(void)
{
    vector<detection> det_buff;
    stringstream stream;
    string result_str;
    int32_t i = 0;
    uint32_t color=0;
 
    mtx.lock();
    copy(det_res.begin(), det_res.end(), back_inserter(det_buff));
    mtx.unlock();

    /* Draw bounding box on RGB image. */
    for (i = 0; i < det_buff.size(); i++)
    {
        /* Skip the overlapped bounding boxes */
        if (det_buff[i].prob == 0) continue;
        
        color = box_color[det_buff[i].c];
        /* Clear string stream for bounding box labels */
        stream.str("");
        /* Draw the bounding box on the image */
        stream << fixed << setprecision(2) << det_buff[i].prob;
        result_str = label_file_map[det_buff[i].c]+ " "+ stream.str();
        img.draw_rect_box((int)det_buff[i].bbox.x, (int)det_buff[i].bbox.y, (int)det_buff[i].bbox.w, (int)det_buff[i].bbox.h, result_str.c_str(),color);
    }
    return;
}

double get_norm(const std::vector<float> vec) {
    double sum = 0.0;
    for(float v : vec) {
        sum += v * v;
    }
    return std::sqrt(sum);

}

double angleBetweenVectors(const std::vector<double>& v1, const std::vector<double>& v2) {
    if (v1.size() != v2.size()) {
        return 0.0;
    }

    double dotProduct = 0.0;
    double mag1 = 0.0;
    double mag2 = 0.0;

    for (size_t i = 0; i < v1.size(); ++i) {
        dotProduct += v1[i] * v2[i];
        mag1 += v1[i] * v1[i];
        mag2 += v2[i] * v2[i];
    }

    mag1 = std::sqrt(mag1);
    mag2 = std::sqrt(mag2);

    if (mag1 == 0.0 || mag2 == 0.0) {
        return 0.0;
    }

    double cosineTheta = dotProduct / (mag1 * mag2);
    double angleRadians = std::acos(cosineTheta);

    // Convert to degrees
    double angleDegrees = angleRadians * 180.0 / PI;

    return angleDegrees;
}

/* Round angles given step size */
void round_angles(vector<float>& raw_angles, float step_size) {

    for(int i=0;i<raw_angles.size();i++) {
        int rounded_number = int(raw_angles[i]/step_size) * step_size;
        raw_angles[i] = rounded_number;
    }
    raw_angles[5] = 160;
    return;
}


/* Calculate Angles based on 2d hand landmarks */
vector<float> calculate_angles(uint16_t id_x_local[NUM_OUTPUT_C][NUM_MAX_PERSON], uint16_t id_y_local[NUM_OUTPUT_C][NUM_MAX_PERSON]) {
    vector<float> res = {180.0, 180.0, 180.0, 180.0, 50.0, 165.0}; // dafault angel
    res[0] = (angleBetweenVectors({float(id_x_local[20][0]) - float(id_x_local[18][0]), float(id_y_local[20][0]) - float(id_y_local[18][0])}, {float(id_x_local[17][0]) - float(id_x_local[18][0]), float(id_y_local[17][0]) - float(id_y_local[18][0])}) - 20) * 1.25;
    // get_norm({float(id_x_local[20][0]) - float(id_x_local[18][0]), float(id_y_local[20][0] - id_y_local[18][0])}) / get_norm({float(id_x_local[17][0]) - float(id_x_local[18][0]), float(id_y_local[17][0] - id_y_local[18][0])});
    res[1] = (angleBetweenVectors({float(id_x_local[16][0]) - float(id_x_local[14][0]), float(id_y_local[16][0]) - float(id_y_local[14][0])}, {float(id_x_local[13][0]) - float(id_x_local[14][0]), float(id_y_local[13][0]) - float(id_y_local[14][0])}) - 20) * 1.25;
    // get_norm({float(id_x_local[16][0]) - float(id_x_local[14][0]), float(id_y_local[16][0] - id_y_local[14][0])}) / get_norm({float(id_x_local[13][0]) - float(id_x_local[14][0]), float(id_y_local[13][0] - id_y_local[14][0])});
    res[2] = (angleBetweenVectors({float(id_x_local[12][0]) - float(id_x_local[10][0]), float(id_y_local[12][0]) - float(id_y_local[10][0])}, {float(id_x_local[9][0]) - float(id_x_local[10][0]), float(id_y_local[9][0]) - float(id_y_local[10][0])}) - 20) * 1.25;
    // get_norm({float(id_x_local[12][0]) - float(id_x_local[10][0]), float(id_y_local[12][0] - id_y_local[10][0])}) / get_norm({float(id_x_local[9][0]) - float(id_x_local[10][0]), float(id_y_local[9][0] - id_y_local[10][0])});
    res[3] = (angleBetweenVectors({float(id_x_local[8][0]) - float(id_x_local[6][0]), float(id_y_local[8][0]) - float(id_y_local[6][0])}, {float(id_x_local[5][0]) - float(id_x_local[6][0]), float(id_y_local[5][0]) - float(id_y_local[6][0])}) - 20) * 1.25;
    // get_norm({float(id_x_local[8][0]) - float(id_x_local[6][0]), float(id_y_local[8][0] - id_y_local[6][0])}) / get_norm({float(id_x_local[5][0]) - float(id_x_local[6][0]), float(id_y_local[5][0] - id_y_local[6][0])});
    res[4] = (angleBetweenVectors({float(id_x_local[4][0]) - float(id_x_local[2][0]), float(id_y_local[4][0]) - float(id_y_local[2][0])}, {float(id_x_local[1][0]) - float(id_x_local[2][0]), float(id_y_local[1][0]) - float(id_y_local[2][0])}) - 100.0) * 1.25 - 30;
    res[5] = angleBetweenVectors({float(id_x_local[2][0]) - float(id_x_local[1][0]), float(id_y_local[2][0]) - float(id_y_local[1][0])}, {float(id_x_local[5][0]) - float(id_x_local[1][0]), float(id_y_local[5][0]) - float(id_y_local[1][0])}) * 2.5 + 70.0;
    round_angles(res, 5.0);
    return res;
}

/* serial commands set dex hand by angles */
vector<char> get_serial_commands_based_on_angles(vector<float> angles) {
    vector<char> res = {235, 144, 1, 15, 18, 206, 5};
    // little finger
    int value = int((angles[0] - 19.0) / 167.7 * 1000.0);
    value = std::max(value, 0);
    value = std::min(value, 1000);
    res.push_back(value & 0xff);
    res.push_back((value & 0xff00) >> 8);
    // ring finger
    value = int((angles[1] - 19.0) / 167.7 * 1000.0);
    value = std::max(value, 0);
    value = std::min(value, 1000);
    res.push_back(value & 0xff);
    res.push_back((value & 0xff00) >> 8);
    // middle finger
    value = int((angles[2] - 19.0) / 167.7 * 1000.0);
    value = std::max(value, 0);
    value = std::min(value, 1000);
    res.push_back(value & 0xff);
    res.push_back((value & 0xff00) >> 8);
    // index finger
    value = int((angles[3] - 19.0) / 167.7 * 1000.0);
    value = std::max(value, 0);
    value = std::min(value, 1000);
    res.push_back(value & 0xff);
    res.push_back((value & 0xff00) >> 8);
    // thumb bending
    value = int((angles[4] + 13.0) / 66.6 * 1000.0);
    value = std::max(value, 0);
    value = std::min(value, 1000);
    res.push_back(value & 0xff);
    res.push_back((value & 0xff00) >> 8);
    // thumb rotate
    value = int((angles[5] - 90.0) / 75.0 * 1000.0);
    value = std::max(value, 0);
    value = std::min(value, 1000);
    res.push_back(value & 0xff);
    res.push_back((value & 0xff00) >> 8);

    // checksum
    char checksum = std::accumulate(res.begin() + 2, res.end(), 0) & 0xff;
    res.push_back(checksum);
    printf("command: ");
    for(char c : res) {
        printf("%d ", c);
    }
    printf("\n");

    return res;
}

/*****************************************
* Function Name : print_result
* Description   : print the result on display.
* Arguments     : -
* Return value  : 0 if succeeded
*               not 0 otherwise
******************************************/
int8_t print_result(Image* img)
{
#ifdef DEBUG_TIME_FLG
    using namespace std;
    chrono::system_clock::time_point start, end;
    start = chrono::system_clock::now();
#endif // DEBUG_TIME_FLG

    int32_t index = 0;
    stringstream stream;
    string str = "";
    string DispStr = "";
    
#if (0) == INF_YOLOX_SKIP
    /* Draw Inference YOLOX Time Result on RGB image.*/
    stream.str("");
    if (yolox_drpai_time< 10){
       DispStr = "YOLOX           Pre-Proc + Inference Time (DRP-AI) :  ";
    }else{
       DispStr = "YOLOX           Pre-Proc + Inference Time (DRP-AI) : ";
    }
    stream << DispStr << std::setw(3) << std::fixed << std::setprecision(1) << std::round(yolox_drpai_time * 10) / 10 << "msec";
    str = stream.str();
    index++;
    img->write_string_rgb(str, 2, TEXT_WIDTH_OFFSET_R,  LINE_HEIGHT_OFFSET + (LINE_HEIGHT * index), CHAR_SCALE_LARGE, 0x00FF00u);
 
    /* Draw Post-Proc YOLOX Time on RGB image.*/
    stream.str("");
    if (yolox_proc_time< 10){
       DispStr = "Post-Proc Time (CPU) :  ";
    }else{
       DispStr = "Post-Proc Time (CPU) : ";
    }
    stream << DispStr << std::setw(3) << std::fixed << std::setprecision(1) << std::round(yolox_proc_time * 10) / 10 << "msec";
    str = stream.str();
    index++;
    img->write_string_rgb(str, 2, TEXT_WIDTH_OFFSET_R, LINE_HEIGHT_OFFSET + (LINE_HEIGHT * index), CHAR_SCALE_LARGE, 0x00FF00u);
#endif
    
    /* Draw Inference HRNet Time Result on RGB image.*/
    stream.str("");
    if (hrnet_drpai_time< 10){
       DispStr = "  Total Pre-Proc + Inference Time (DRP-AI) :  ";
    }else{
       DispStr = "  Total Pre-Proc + Inference Time (DRP-AI) : ";
    }
    stream << "HRNet x " << (uint32_t)ppl_count << DispStr << std::setw(3) << std::fixed << std::setprecision(1) << std::round(hrnet_drpai_time * 10) / 10 << "msec";
    str = stream.str();
    index++;
    img->write_string_rgb(str, 2, TEXT_WIDTH_OFFSET_R,  LINE_HEIGHT_OFFSET + (LINE_HEIGHT * index), CHAR_SCALE_LARGE, 0xFFF000u);
 
    /* Draw Post-Proc HRNet Time on RGB image.*/
    stream.str("");
    if (hrnet_proc_time< 10){
       DispStr = "Total Post-Proc Time (CPU) :  ";
    }else{
       DispStr = "Total Post-Proc Time (CPU) : ";
    }
    stream << DispStr << std::setw(3) << std::fixed << std::setprecision(1) << std::round(hrnet_proc_time * 10) / 10 << "msec";
    str = stream.str();
    index++;
    img->write_string_rgb(str, 2, TEXT_WIDTH_OFFSET_R, LINE_HEIGHT_OFFSET + (LINE_HEIGHT * index), CHAR_SCALE_LARGE, 0xFFF000u);

#ifdef DISP_AI_FRAME_RATE
    /* Draw AI/Camera Frame Rate on RGB image.*/
    stream.str("");
    stream << "AI/Camera Frame Rate: " << std::setw(3) << (uint32_t)ai_fps << "/" << (uint32_t)cap_fps << "fps";
    str = stream.str();
    img->write_string_rgb(str, 1, TEXT_WIDTH_OFFSET_L, LINE_HEIGHT_OFFSET + (LINE_HEIGHT * 2), CHAR_SCALE_LARGE, WHITE_DATA);
#endif /* DISP_AI_FRAME_RATE */

    vector<float> hand_angles = calculate_angles(id_x_local, id_y_local);
    vector<char> dex_hand_commands = get_serial_commands_based_on_angles(hand_angles);
    std::copy(dex_hand_commands.begin(), dex_hand_commands.end(), DEX_HAND_CMD);
    write(serial_port, DEX_HAND_CMD, sizeof(DEX_HAND_CMD));
    printf("sending dex hand commands through serial.\n");
    for(float a : hand_angles) {
        printf("%f ,", a);
    }
    printf("\n");
    // Print angles
    stream.str("");
    stream << "hand angles: " << hand_angles[0] << ", " << hand_angles[1] << ", " << hand_angles[2] << ", " << hand_angles[3] << ", " << hand_angles[4] << ", " << hand_angles[5];
    str = stream.str();
    index++;
    img->write_string_rgb(str, 2, TEXT_WIDTH_OFFSET_R, LINE_HEIGHT_OFFSET + (LINE_HEIGHT * index), CHAR_SCALE_LARGE, 0xFFF000u);


#ifdef DEBUG_TIME_FLG
    end = chrono::system_clock::now();
    double time = static_cast<double>(chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.0);
    printf("Draw Text Time            : %lf[ms]\n", time);
#endif // DEBUG_TIME_FLG

    return 0;
}

/*****************************************
* Function Name : R_Inf_Thread
* Description   : Executes the DRP-AI inference thread
* Arguments     : threadid = thread identification
* Return value  : -
******************************************/
void *R_Inf_Thread(void *threadid)
{
    /*Semaphore Variable*/
    int32_t inf_sem_check = 0;
    int32_t inf_cnt = -1;
    /*Inference Variables*/
    fd_set rfds;
    fd_set rfds0;
    struct timespec tv;
    int8_t inf_status = 0;
    drpai_status_t drpai_status0;
    drpai_status_t drpai_status1;
    /*Variable for checking return value*/
    int8_t ret = 0;
    /*Variable for Performance Measurement*/
    double total_hrnet_drpai_time = 0;
    double total_hrnet_proc_time = 0;
    timespec yolox_sta_time;
    timespec yolox_end_time;
    static struct timespec yolox_drp_start_time;
    static struct timespec yolox_drp_end_time;
    timespec hrnet_sta_time;
    timespec hrnet_end_time;
	static struct timespec hrnet_drp_start_time;
    static struct timespec hrnet_drp_end_time;

	static struct timespec inf_start_time;
    static struct timespec inf_end_time;
    static struct timespec drp_prev_time = { .tv_sec = 0, .tv_nsec = 0, };
    /*HRNet Modify Parameters*/
    drpai_crop_t crop_param;
    static string drpai_param_file;
    uint32_t drp_param_info_size;
    uint8_t i;

    printf("Inference Thread Starting\n");

    /*DRP-AI Output Memory Preparation*/
    /*HRNet*/
    drpai_data0.address = drpai_hdl0->data_inout.data_out_addr;
    drpai_data0.size = drpai_hdl0->data_inout.data_out_size;
    /*YOLOX*/
    drpai_data1.address = drpai_hdl1->data_inout.data_out_addr;
    drpai_data1.size = drpai_hdl1->data_inout.data_out_size;

    printf("Inference Loop Starting\n");
    /*Inference Loop Start*/
    while(1)
    {
        inf_cnt++;
        spdlog::info("[START] Start DRP-AI Inference...");
        spdlog::info("Inference ----------- No. {}", (inf_cnt + 1));
        while(1)
        {
            /*Gets the Termination request semaphore value. If different then 1 Termination was requested*/
            /*Checks if sem_getvalue is executed wihtout issue*/
            errno = 0;
            ret = sem_getvalue(&terminate_req_sem, &inf_sem_check);
            if (0 != ret)
            {
                fprintf(stderr, "[ERROR] Failed to get Semaphore Value: errno=%d\n", errno);
                goto err;
            }
            /*Checks the semaphore value*/
            if (1 != inf_sem_check)
            {
                goto ai_inf_end;
            }
            /*Checks if image frame from Capture Thread is ready.*/
            if (inference_start.load())
            {
                break;
            }
            usleep(WAIT_TIME);
        }
        ppl_count_local = 0;
        memset(array_hrnet_drpai_time,0,sizeof(array_hrnet_drpai_time));
        memset(array_hrnet_proc_time,0,sizeof(array_hrnet_proc_time));

#if (0) == INF_YOLOX_SKIP
        /*Gets inference starting time*/
        ret = timespec_get(&yolox_drp_start_time, TIME_UTC);
        if (0 == ret)
        {
            fprintf(stderr, "[ERROR] Failed to get Inference Start Time\n");
            goto err;
        }
        /*Start DRP-AI Driver*/
        errno = 0;
        ret = start_drpai(drpai_hdl0, (uintptr_t) capture_address, drp_max_freq, drpai_freq);
        if (0 > ret)
        {
            fprintf(stderr, "[ERROR] Failed to run DRPAI_START: errno=%d\n", errno);
            goto err;
        }
#else
        ret = 1;  /* YOLOX Skip*/
#endif
        
        while(1)
        {
            
            errno = 0;
#if (0) == INF_YOLOX_SKIP
            ret = sem_getvalue(&terminate_req_sem, &inf_sem_check);
            if (0 != ret)
            {
                fprintf(stderr, "[ERROR] Failed to get Semaphore Value: errno=%d\n", errno);
                goto err;
            }
            /*Checks the semaphore value*/
            if (1 != inf_sem_check)
            {
                goto ai_inf_end;
            }
            
            /*Setup pselect settings*/
            FD_ZERO(&rfds0);
            FD_SET(drpai_fd0, &rfds0);
            tv.tv_sec = DRPAI_TIMEOUT;
            tv.tv_nsec = 0;

            /*Wait Till The DRP-AI Ends*/
            ret = pselect(drpai_fd0+1, &rfds0, NULL, NULL, &tv, NULL);
#else
            ret = 1;  /* YOLOX Skip*/
#endif
            if (0 == ret)
            {
                fprintf(stderr, "[ERROR] DRP-AI Inference pselect() Timeout: errno=%d\n", errno);
                goto err;
            }
            else if (0 > ret)
            {
                fprintf(stderr, "[ERROR] DRP-AI Inference pselect() Error: errno=%d\n", errno);
                ret = ioctl(drpai_fd0, DRPAI_GET_STATUS, &drpai_status0);
                if (-1 == ret)
                {
                    fprintf(stderr, "[ERROR] Failed to run DRPAI_GET_STATUS : errno=%d\n", errno);
                }
                goto err;
            }
            else
            {

#if (0) == INF_YOLOX_SKIP
                /*Gets AI Inference End Time*/
                ret = timespec_get(&yolox_drp_end_time, TIME_UTC);
                if ( 0 == ret)
                {
                    fprintf(stderr, "[ERROR] Failed to Get Inference End Time\n");
                    goto err;
                }

                /*Checks if DRPAI Inference ended without issue*/
                inf_status = ioctl(drpai_fd0, DRPAI_GET_STATUS, &drpai_status0);
#else
                inf_status = 0;  /* YOLOX Skip*/
#endif
                if (0 == inf_status)
                {
                
#if (0) == INF_YOLOX_SKIP
                    /*Process to read the DRPAI output data.*/
                    ret = get_result(drpai_fd0, drpai_data0.address, num_inf_out * sizeof(drpai_output_buf0[0]));
                    if (0 != ret)
                    {
                        fprintf(stderr, "[ERROR] Failed to get result from memory.\n");
                        goto err;
                    }

                    /* YOLOX R_Post_Proc time start*/
                    ret = timespec_get(&yolox_sta_time, TIME_UTC);
                    if (0 == ret)
                    {
                        fprintf(stderr, "[ERROR] Failed to get R_Post_Proc Start Time\n");
                        goto err;
                    }
                    /*Preparation For Post-Processing*/
                    bcount = 0;
                    det_res.clear();
                    /*CPU Post-Processing For YOLOX*/
                    R_Post_Proc(&drpai_output_buf0[0], det_res, &bcount);
                    /*Count the Number of People Detected*/
                    ppl_count_local = 0;
                    people_counter(det_res, det_ppl, bcount, &ppl_count_local);
                    /* YOLOX R_Post_Proc time end*/
                    ret = timespec_get(&yolox_end_time, TIME_UTC);
                    if (0 == ret)
                    {
                        fprintf(stderr, "[ERROR] Failed to get R_Post_Proc end Time\n");
                        goto err;
                    }
                    yolox_proc_time  = (timedifference_msec(yolox_sta_time, yolox_end_time) * TIME_COEF);
#else
                    ppl_count_local = 1;  /* YOLOX Skip*/
#endif
                    
                    /*If Person is detected run HRNet for Pose Estimation three times*/
                    if(ppl_count_local > 0)
                    {
                        for(i = 0; i < ppl_count_local; i++)
                        {
#if (0) == INF_YOLOX_SKIP
                            croph[i] = det_ppl[i].bbox.h + CROP_ADJ_X;
                            cropw[i] = det_ppl[i].bbox.w + CROP_ADJ_Y;
#else
                            /* YOLOX Skip*/
                            croph[i] = CROPPED_IMAGE_HEIGHT;
                            cropw[i] = CROPPED_IMAGE_WIDTH;
#endif
                            /*Checks that cropping height and width does not exceeds image dimension*/
                            if(croph[i] < 1)
                            {
                                croph[i] = 1;
                            }
                            else if(croph[i] > IMREAD_IMAGE_HEIGHT)
                            {
                                croph[i] = IMREAD_IMAGE_HEIGHT;
                            }
                            else
                            {
                                /*Do Nothing*/
                            }
                            if(cropw[i] < 1)
                            {
                                cropw[i] = 1;
                            }
                            else if(cropw[i] > IMREAD_IMAGE_WIDTH)
                            {
                                cropw[i] = IMREAD_IMAGE_WIDTH;
                            }
                            else
                            {
                                /*Do Nothing*/
                            }
#if (0) == INF_YOLOX_SKIP
                            /*Compute Cropping Y Position based on Detection Result*/
                            /*If Negative Cropping Position*/
                            if(det_ppl[i].bbox.y < (croph[i]/2))
                            {
                                cropy[i] = 0;
                            }
                            else if(det_ppl[i].bbox.y > (IMREAD_IMAGE_HEIGHT-croph[i]/2)) /*If Exceeds Image Area*/
                            {
                                cropy[i] = IMREAD_IMAGE_HEIGHT-croph[i];
                            }
                            else
                            {
                                cropy[i] = (int16_t)det_ppl[i].bbox.y - croph[i]/2;
                            }
                            /*Compute Cropping X Position based on Detection Result*/
                            /*If Negative Cropping Position*/
                            if(det_ppl[i].bbox.x < (cropw[i]/2))
                            {
                                cropx[i] = 0;
                            }
                            else if(det_ppl[i].bbox.x > (IMREAD_IMAGE_WIDTH-cropw[i]/2)) /*If Exceeds Image Area*/
                            {
                                cropx[i] = IMREAD_IMAGE_WIDTH-cropw[i];
                            }
                            else
                            {
                                cropx[i] = (int16_t)det_ppl[i].bbox.x - cropw[i]/2;
                            }
#else
                            cropx[i] = OUTPUT_LEFT;
                            cropy[i] = 0;
#endif
                            /*Checks that combined cropping position with width and height does not exceed the image dimension*/
                            if(cropx[i] + cropw[i] > IMREAD_IMAGE_WIDTH)
                            {
                                cropw[i] = IMREAD_IMAGE_WIDTH - cropx[i];
                            }
                            if(cropy[i] + croph[i] > IMREAD_IMAGE_HEIGHT)
                            {
                                croph[i] = IMREAD_IMAGE_HEIGHT - cropy[i];
                            }
                            /*Change HRNet Crop Parameters*/
                            crop_param.img_owidth = (uint16_t)cropw[i];
                            crop_param.img_oheight = (uint16_t)croph[i];
                            crop_param.pos_x = (uint16_t)cropx[i];
                            crop_param.pos_y = (uint16_t)cropy[i];
                            crop_param.obj.address = drpai_hdl1->drpai_address.drp_param_addr;
                            crop_param.obj.size = drpai_hdl1->drpai_address.drp_param_size;
                            ret = ioctl(drpai_fd1, DRPAI_PREPOST_CROP, &crop_param);
                            if (0 != ret)
                            {
                                fprintf(stderr, "[ERROR] Failed to DRPAI prepost crop: errno=%d\n", errno);
                                goto err;
                            }
                            /*Get Inference Start Time*/
                            ret = timespec_get(&hrnet_drp_start_time, TIME_UTC);
                            if ( 0 == ret)
                            {
                                fprintf(stderr, "[ERROR] Failed to Get Inference henrt Start Time\n");
                                goto err;
                            }
                            /*Start DRP-AI Driver*/
                            ret = start_drpai(drpai_hdl1, (uintptr_t) capture_address, drp_max_freq, drpai_freq);
                            if (0 > ret)
                            {
                                fprintf(stderr, "[ERROR] Failed to run DRPAI_START: errno=%d\n", errno);
                                goto err;
                            }

                            while(1)
                            {
                                /*Gets The Termination Request Semaphore Value, If Different Then 1 Termination Was Requested*/
                                /*Checks If sem_getvalue Is Executed Without Issue*/
                                ret = sem_getvalue(&terminate_req_sem, &inf_sem_check);
                                if (0 != ret)
                                {
                                    fprintf(stderr, "[ERROR] Failed to get Semaphore Value: errno=%d\n", errno);
                                    goto err;
                                }
                                /*Checks The Semaphore Value*/
                                if (1 != inf_sem_check)
                                {
                                    goto ai_inf_end;
                                }
                                /*Wait Till The DRP-AI Ends*/
                                FD_ZERO(&rfds);
                                FD_SET(drpai_fd1, &rfds);
                                tv.tv_sec = DRPAI_TIMEOUT;
                                tv.tv_nsec = 0;
                                ret = pselect(drpai_fd1+1, &rfds, NULL, NULL, &tv, NULL);
                                if(ret == 0)
                                {
                                    /*Do Nothing*/
                                }
                                else if(ret < 0)
                                {
                                    fprintf(stderr, "[ERROR] DRPAI Inference pselect() Error: %d\n", errno);
                                    goto err;
                                }
                                else
                                {
                                    /*Gets AI Inference End Time*/
                                    ret = timespec_get(&hrnet_drp_end_time, TIME_UTC);
                                    if ( 0 == ret)
                                    {
                                        fprintf(stderr, "[ERROR] Failed to Get Inference hrnet End Time\n");
                                        goto err;
                                    }
                                    array_hrnet_drpai_time[i] = array_hrnet_drpai_time[i] = (timedifference_msec(hrnet_drp_start_time, hrnet_drp_end_time)*TIME_COEF);

                                    /*Checks If DRPAI Inference Ended Without Issue*/
                                    inf_status = ioctl(drpai_fd1, DRPAI_GET_STATUS, &drpai_status1);
                                    if(inf_status == 0)
                                    {
                                        /*Get DRPAI Output Data*/
                                        inf_status = ioctl(drpai_fd1, DRPAI_ASSIGN, &drpai_data1);
                                        if (inf_status != 0)
                                        {
                                            fprintf(stderr, "[ERROR] Failed to run DRPAI_ASSIGN: errno=%d\n", errno);
                                                goto err;
                                        }
                                        inf_status = read(drpai_fd1, &drpai_output_buf1[0], INF_OUT_SIZE * sizeof(drpai_output_buf1[0]));
                                        if(inf_status < 0)
                                        {
                                            fprintf(stderr, "[ERROR] Failed to read via DRP-AI Driver: errno=%d\n", errno);
                                            goto err;
                                        }

                                        /* HRNet R_Post_Proc time start*/
                                        ret = timespec_get(&hrnet_sta_time, TIME_UTC);
                                        if (0 == ret)
                                        {
                                            fprintf(stderr, "[ERROR] Failed to get R_Post_Proc_HRNet Start Time\n");
                                            goto err;
                                        }
                                        /*CPU Post Processing For HRNet & Display the Results*/
                                        R_Post_Proc_HRNet(&drpai_output_buf1[0],i);

                                        if(lowest_kpt_score_local[i] > 0)
                                        {
                                            R_HRNet_Coord_Convert(i);
                                        }

                                        /* HRNet R_Post_Proc time end*/
                                        ret = timespec_get(&hrnet_end_time, TIME_UTC);
                                        if (0 == ret)
                                        {
                                            fprintf(stderr, "[ERROR] Failed to get R_Post_Proc_HRNet end Time\n");
                                            goto err;
                                        }
                                        array_hrnet_proc_time[i] = array_hrnet_proc_time[i] = (timedifference_msec(hrnet_sta_time, hrnet_end_time) * TIME_COEF);
                                        break;
                                    }
                                    else //inf_status != 0
                                    {
                                            fprintf(stderr, "[ERROR] DRPAI Internal Error: errno=%d\n", errno);
                                            goto err;
                                    }
                                }
                            }
                        }
                    }

                    /*Copy data for Display Thread*/
                    ppl_count = 0;
                    memcpy(lowest_kpt_score,lowest_kpt_score_local,sizeof(lowest_kpt_score_local));
                    memcpy(id_x, id_x_local, sizeof(id_x_local));
                    memcpy(id_y, id_y_local,sizeof(id_y_local));
                    ppl_count = ppl_count_local;

                    /* R_Post_Proc time end*/
                    ret = timespec_get(&inf_end_time, TIME_UTC);
                    if (0 == ret)
                    {
                        fprintf(stderr, "[ERROR] Failed to Get R_Post_Proc End Time\n");
                        goto err;
                    }
                    
                    break;

                }
                else
                {
                    /* inf_status != 0 */
                    fprintf(stderr, "[ERROR] DRPAI Internal Error: errno=%d\n", errno);
                    goto err;
                }
            }
        }

#if (0) == INF_YOLOX_SKIP
        /*Display Processing YOLOX Time On Log File*/
        yolox_drpai_time = (timedifference_msec(yolox_drp_start_time, yolox_drp_end_time) * TIME_COEF);
        spdlog::info("YOLOX");
        spdlog::info(" Pre-Proc + Inference Time (DRP-AI): {} [ms]", std::round(yolox_drpai_time * 10) / 10);
        spdlog::info(" Post-Proc Time (CPU): {} [ms]", std::round(yolox_proc_time * 10) / 10);
#endif

        /*Display Processing HRNet Time On Log File*/
        /*Display Processing Time On Console*/
        total_hrnet_drpai_time = 0;
        total_hrnet_proc_time = 0;
        for(uint8_t i = 0; i < ppl_count_local; i++)
        {
            total_hrnet_drpai_time += array_hrnet_drpai_time[i];
            total_hrnet_proc_time += array_hrnet_proc_time[i];
        }
        hrnet_drpai_time = total_hrnet_drpai_time;
        hrnet_proc_time = total_hrnet_proc_time ;
        spdlog::info("HRNet");
        spdlog::info(" Total Pre-Proc + Inference Time (DRP-AI): {} [ms]", std::round(hrnet_drpai_time * 10) / 10);
        spdlog::info(" Total Post-Proc Time (CPU): {} [ms]", std::round(hrnet_proc_time * 10) / 10);
        
        /*Display Processing Frame Rate On Log File*/
        ai_time = (uint32_t)((timedifference_msec(drp_prev_time, inf_end_time) * TIME_COEF));
        int idx = inf_cnt % SIZE_OF_ARRAY(array_drp_time);
        array_drp_time[idx] = ai_time;
        drp_prev_time = inf_end_time;
#ifdef DISP_AI_FRAME_RATE
        int arraySum = std::accumulate(array_drp_time, array_drp_time + SIZE_OF_ARRAY(array_drp_time), 0);
        double arrayAvg = 1.0 * arraySum / SIZE_OF_ARRAY(array_drp_time);
        ai_fps = 1.0 / arrayAvg * 1000.0 + 0.5;
        spdlog::info("AI Frame Rate {} [fps]", (int32_t)ai_fps);
#endif /* DISP_AI_FRAME_RATE */

        inference_start.store(0);
    }
    /*End of Inference Loop*/

/*Error Processing*/
err:
    /*Set Termination Request Semaphore to 0*/
    sem_trywait(&terminate_req_sem);
    goto ai_inf_end;
/*AI Thread Termination*/
ai_inf_end:
    /*To terminate the loop in Capture Thread.*/
    printf("AI Inference Thread Terminated\n");
    pthread_exit(NULL);
}

/*****************************************
* Function Name : R_Capture_Thread
* Description   : Executes the V4L2 capture with Capture thread.
* Arguments     : threadid = thread identification
* Return value  : -
******************************************/
void *R_Capture_Thread(void *threadid)
{
    Camera* capture = (Camera*) threadid;
    /*Semaphore Variable*/
    int32_t capture_sem_check = 0;
    /*First Loop Flag*/
    uint64_t capture_addr = 0;
    int32_t _cap_offset;
    int32_t _img_offset;
    int8_t ret = 0;
    int32_t counter = 0;
    uint8_t * img_buffer;
    uint8_t * img_buffer0;
    uint8_t capture_stabe_cnt = 8;  // Counter to wait for the camera to stabilize
    int32_t cap_cnt = -1;
#ifdef DISP_AI_FRAME_RATE
    static struct timespec capture_time;
    static struct timespec capture_time_prev = { .tv_sec = 0, .tv_nsec = 0, };
#endif /* DISP_AI_FRAME_RATE */

#if (0) == INPUT_CAM_TYPE
    double elapsed_time_last_disp = 0;
    double target_disp_fps = 15.0;
#endif

    printf("Capture Thread Starting\n");

    img_buffer0 = (uint8_t *)capture->drpai_buf->mem;
    if (MAP_FAILED == img_buffer0)
    {
        fprintf(stderr, "[ERROR] Failed to mmap\n");
        goto err;
    }
#if (1) == DRPAI_INPUT_PADDING
    /** Fill buffer with the brightness 114. */
    for( uint32_t i = 0; i < CAM_IMAGE_WIDTH * CAM_IMAGE_WIDTH * CAM_IMAGE_CHANNEL_YUY2; i += 4 )
    {
        /// Y =  0.299R + 0.587G + 0.114B
        img_buffer0[i]   = 114;    
        img_buffer0[i+2] = 114;
        /// U = -0.169R - 0.331G + 0.500B + 128
        img_buffer0[i+1] = 128;
        /// V =  0.500R - 0.419G - 0.081B + 128
        img_buffer0[i+3] = 128;
    }
#endif  /* (1) == DRPAI_INPUT_PADDING */
    capture_address = capture->drpai_buf->phy_addr;

    while(1)
    {
        /*Gets the Termination request semaphore value. If different then 1 Termination was requested*/
        /*Checks if sem_getvalue is executed wihtout issue*/
        errno = 0;
        ret = sem_getvalue(&terminate_req_sem, &capture_sem_check);
        if (0 != ret)
        {
            fprintf(stderr, "[ERROR] Failed to get Semaphore Value: errno=%d\n", errno);
            goto err;
        }
        /*Checks the semaphore value*/
        if (1 != capture_sem_check)
        {
            goto capture_end;
        }

        /* Capture USB camera image and stop updating the capture buffer */
        capture_addr = (uint32_t)capture->capture_image();

#ifdef DISP_AI_FRAME_RATE
        cap_cnt++;
        ret = timespec_get(&capture_time, TIME_UTC);
        proc_time_capture = (timedifference_msec(capture_time_prev, capture_time) * TIME_COEF);
        capture_time_prev = capture_time;

        int idx = cap_cnt % SIZE_OF_ARRAY(array_cap_time);
        array_cap_time[idx] = (uint32_t)proc_time_capture;
        int arraySum = std::accumulate(array_cap_time, array_cap_time + SIZE_OF_ARRAY(array_cap_time), 0);
        double arrayAvg = 1.0 * arraySum / SIZE_OF_ARRAY(array_cap_time);
        cap_fps = 1.0 / arrayAvg * 1000.0 + 0.5;
#endif /* DISP_AI_FRAME_RATE */

        if (capture_addr == 0)
        {
            fprintf(stderr, "[ERROR] Failed to capture image from camera.\n");
            goto err;
        }
        else
        {
            /* Do not process until the camera stabilizes, because the image is unreliable until the camera stabilizes. */
            if( capture_stabe_cnt > 0 )
            {
                capture_stabe_cnt--;
            }
            else
            {
                img_buffer = capture->get_img();
                if (!inference_start.load())
                {
                    /* Copy captured image to Image object. This will be used in Display Thread. */
                    memcpy(img_buffer0, img_buffer, capture->get_size());
                    /* Flush capture image area cache */
                    ret = capture->video_buffer_flush_dmabuf(capture->drpai_buf->idx, capture->drpai_buf->size);
                    if (0 != ret)
                    {
                        goto err;
                    }
                    inference_start.store(1); /* Flag for AI Inference Thread. */
                }

#if (0) == INPUT_CAM_TYPE
                /**  
                 * To stabilize the Capture thread and AI Inference thread when this application uses the USB camera, control the display frame rate to 15 fps.
                 * In details, controls the number of frames to be sent the Image thread.
                 * This thread just has to send the frame to the Image thread every 66.6 msec (=1000[msec]/15[fps]).
                 */
                elapsed_time_last_disp += proc_time_capture;
                if( 1000 / target_disp_fps <= elapsed_time_last_disp )
                {
                    elapsed_time_last_disp = fmod(elapsed_time_last_disp, 1000 / target_disp_fps);

                    if (!img_obj_ready.load())
                    {
                        img.camera_to_image(img_buffer, capture->get_size());
                        ret = capture->video_buffer_flush_dmabuf(capture->wayland_buf->idx, capture->wayland_buf->size);
                        if (0 != ret)
                        {
                            goto err;
                        }
                        img_obj_ready.store(1); /* Flag for Display Thread. */
                    }
                }
#else
                if (!img_obj_ready.load())
                {
                    img.camera_to_image(img_buffer, capture->get_size());
                    ret = capture->video_buffer_flush_dmabuf(capture->wayland_buf->idx, capture->wayland_buf->size);
                    if (0 != ret)
                    {
                        goto err;
                    }
                    img_obj_ready.store(1); /* Flag for Display Thread. */
                }
#endif
            }
        }

        /* IMPORTANT: Place back the image buffer to the capture queue */
        ret = capture->capture_qbuf();
        if (0 != ret)
        {
            fprintf(stderr, "[ERROR] Failed to enqueue capture buffer.\n");
            goto err;
        }
    } /*End of Loop*/

/*Error Processing*/
err:
    sem_trywait(&terminate_req_sem);
    goto capture_end;

capture_end:
    /*To terminate the loop in AI Inference Thread.*/
    inference_start.store(1);

    printf("Capture Thread Terminated\n");
    pthread_exit(NULL);
}

/*****************************************
* Function Name : R_Img_Thread
* Description   : Executes img proc with img thread
* Arguments     : threadid = thread identification
* Return value  : -
******************************************/
void *R_Img_Thread(void *threadid)
{
    /*Semaphore Variable*/
    int32_t hdmi_sem_check = 0;
    /*Variable for checking return value*/
    int8_t ret = 0;
    double img_proc_time = 0;
    int32_t disp_cnt = 0;
    bool padding = false;
#ifdef CAM_INPUT_VGA
    padding = false;
#endif // CAM_INPUT_VGA
    timespec start_time;
    timespec end_time;

    printf("Image Thread Starting\n");
    while(1)
    {
        /*Gets The Termination Request Semaphore Value, If Different Then 1 Termination Is Requested*/
        /*Checks If sem_getvalue Is Executed Without Issue*/
        errno = 0;
        ret = sem_getvalue(&terminate_req_sem, &hdmi_sem_check);
        if (0 != ret)
        {
            fprintf(stderr, "[ERROR] Failed to get Semaphore Value: errno=%d\n", errno);
            goto err;
        }
        /*Checks the semaphore value*/
        if (1 != hdmi_sem_check)
        {
            goto hdmi_end;
        }
        /* Check img_obj_ready flag which is set in Capture Thread. */
        if (img_obj_ready.load())
        {
            ret = timespec_get(&start_time, TIME_UTC);
            if (0 == ret)
            {
                fprintf(stderr, "[ERROR] Failed to get Display Start Time\n");
                goto err;
            }
            
            /* Draw Complete Skeleton. */
            draw_skeleton();

            /* Convert YUYV image to BGRA format. */
            img.convert_format();

            /* Convert output image size. */
            img.convert_size(CAM_IMAGE_WIDTH, DRPAI_OUT_WIDTH, padding);

#if (0) == INF_YOLOX_SKIP
            /* Draw bounding box on image. */
            draw_bounding_box();
#endif            
            /*displays AI Inference Results on display.*/
            print_result(&img);

            buf_id = img.get_buf_id();
            img_obj_ready.store(0);

            if (!hdmi_obj_ready.load())
            {
                hdmi_obj_ready.store(1); /* Flag for AI Inference Thread. */
            }
            
            ret = timespec_get(&end_time, TIME_UTC);
            if (0 == ret)
            {
                fprintf(stderr, "[ERROR] Failed to Get Display End Time\n");
                goto err;
            }
            img_proc_time = (timedifference_msec(start_time, end_time) * TIME_COEF);
            
#ifdef DEBUG_TIME_FLG
            printf("Img Proc Time             : %lf[ms]\n", img_proc_time);
#endif
        }
        usleep(WAIT_TIME); //wait 1 tick time
    } /*End Of Loop*/

/*Error Processing*/
err:
    /*Set Termination Request Semaphore To 0*/
    sem_trywait(&terminate_req_sem);
    goto hdmi_end;

hdmi_end:
    /*To terminate the loop in Capture Thread.*/
    img_obj_ready.store(0);
    printf("Img Thread Terminated\n");
    pthread_exit(NULL);
}
/*****************************************
* Function Name : R_Display_Thread
* Description   : Executes the HDMI Display with Display thread
* Arguments     : threadid = thread identification
* Return value  : -
******************************************/
void *R_Display_Thread(void *threadid)
{
    /*Semaphore Variable*/
    int32_t hdmi_sem_check = 0;
    /*Variable for checking return value*/
    int8_t ret = 0;
    double disp_proc_time = 0;
    int32_t disp_cnt = 0;

    timespec start_time;
    timespec end_time;
    static struct timespec disp_prev_time = { .tv_sec = 0, .tv_nsec = 0, };

    /* Initialize waylad */
    ret = wayland.init(capture->wayland_buf->idx, IMAGE_OUTPUT_WIDTH, IMAGE_OUTPUT_HEIGHT, IMAGE_CHANNEL_BGRA);
    if(0 != ret)
    {
        fprintf(stderr, "[ERROR] Failed to initialize Image for Wayland\n");
        goto err;
    }

    printf("Display Thread Starting\n");
    while(1)
    {
        /*Gets The Termination Request Semaphore Value, If Different Then 1 Termination Is Requested*/
        /*Checks If sem_getvalue Is Executed Without Issue*/
        errno = 0;
        ret = sem_getvalue(&terminate_req_sem, &hdmi_sem_check);
        if (0 != ret)
        {
            fprintf(stderr, "[ERROR] Failed to get Semaphore Value: errno=%d\n", errno);
            goto err;
        }
        /*Checks the semaphore value*/
        if (1 != hdmi_sem_check)
        {
            goto hdmi_end;
        }
        /* Check hdmi_obj_ready flag which is set in Capture Thread. */
        if (hdmi_obj_ready.load())
        {
            ret = timespec_get(&start_time, TIME_UTC);
            if (0 == ret)
            {
                fprintf(stderr, "[ERROR] Failed to get Display Start Time\n");
                goto err;
            }
            /*Update Wayland*/
            wayland.commit(img.get_img(buf_id), NULL);

            hdmi_obj_ready.store(0);
            ret = timespec_get(&end_time, TIME_UTC);
            if (0 == ret)
            {
                fprintf(stderr, "[ERROR] Failed to Get Display End Time\n");
                goto err;
            }
            disp_proc_time = (timedifference_msec(start_time, end_time) * TIME_COEF);
            disp_time = (uint32_t)((timedifference_msec(disp_prev_time, end_time) * TIME_COEF));
            int idx = disp_cnt++ % SIZE_OF_ARRAY(array_disp_time);
            array_disp_time[idx] = disp_time;
            disp_prev_time = end_time;
#ifdef DEBUG_TIME_FLG
            /* Draw Disp Frame Rate on RGB image.*/
            int arraySum = std::accumulate(array_disp_time, array_disp_time + SIZE_OF_ARRAY(array_disp_time), 0);
            double arrayAvg = 1.0 * arraySum / SIZE_OF_ARRAY(array_disp_time);
            double disp_fps = 1.0 / arrayAvg * 1000.0;

            printf("Disp Proc Time            : %lf[ms]\n", disp_proc_time);
            printf("Disp Frame Rate           : %lf[fps]\n", disp_fps);
            printf("Dipslay ------------------------------ No. %d\n", disp_cnt);
#endif
        }
        usleep(WAIT_TIME); //wait 1 tick time
    } /*End Of Loop*/

/*Error Processing*/
err:
    /*Set Termination Request Semaphore To 0*/
    sem_trywait(&terminate_req_sem);
    goto hdmi_end;

hdmi_end:
    /*To terminate the loop in Capture Thread.*/
    hdmi_obj_ready.store(0);
    printf("Display Thread Terminated\n");
    pthread_exit(NULL);
}

/*****************************************
* Function Name : R_Kbhit_Thread
* Description   : Executes the Keyboard hit thread (checks if enter key is hit)
* Arguments     : threadid = thread identification
* Return value  : -
******************************************/
void *R_Kbhit_Thread(void *threadid)
{
    /*Semaphore Variable*/
    int32_t kh_sem_check = 0;
    /*Variable to store the getchar() value*/
    int32_t c = 0;
    /*Variable for checking return value*/
    int8_t ret = 0;

    printf("Key Hit Thread Starting\n");

    printf("************************************************\n");
    printf("* Press ENTER key to quit. *\n");
    printf("************************************************\n");

    /*Set Standard Input to Non Blocking*/
    errno = 0;
    ret = fcntl(0, F_SETFL, O_NONBLOCK);
    if (-1 == ret)
    {
        fprintf(stderr, "[ERROR] Failed to run fctnl(): errno=%d\n", errno);
        goto err;
    }

    while(1)
    {
        /*Gets the Termination request semaphore value. If different then 1 Termination was requested*/
        /*Checks if sem_getvalue is executed wihtout issue*/
        errno = 0;
        ret = sem_getvalue(&terminate_req_sem, &kh_sem_check);
        if (0 != ret)
        {
            fprintf(stderr, "[ERROR] Failed to get Semaphore Value: errno=%d\n", errno);
            goto err;
        }
        /*Checks the semaphore value*/
        if (1 != kh_sem_check)
        {
            goto key_hit_end;
        }

        c = getchar();
        if (EOF != c)
        {
            /* When key is pressed. */
            printf("key Detected.\n");
            goto err;
        }
        else
        {
            /* When nothing is pressed. */
            usleep(WAIT_TIME);
        }
    }

/*Error Processing*/
err:
    /*Set Termination Request Semaphore to 0*/
    sem_trywait(&terminate_req_sem);
    goto key_hit_end;

key_hit_end:
    printf("Key Hit Thread Terminated\n");
    pthread_exit(NULL);
}

/*****************************************
* Function Name : R_Main_Process
* Description   : Runs the main process loop
* Arguments     : -
* Return value  : 0 if succeeded
*                 not 0 otherwise
******************************************/
int8_t R_Main_Process()
{
    /*Main Process Variables*/
    int8_t main_ret = 0;
    /*Semaphore Related*/
    int32_t sem_check = 0;
    /*Variable for checking return value*/
    int8_t ret = 0;

    printf("Main Loop Starts\n");
    while(1)
    {
        /*Gets the Termination request semaphore value. If different then 1 Termination was requested*/
        errno = 0;
        ret = sem_getvalue(&terminate_req_sem, &sem_check);
        if (0 != ret)
        {
            fprintf(stderr, "[ERROR] Failed to get Semaphore Value: errno=%d\n", errno);
            goto err;
        }
        /*Checks the semaphore value*/
        if (1 != sem_check)
        {
            goto main_proc_end;
        }
        /*Wait for 1 TICK.*/
        usleep(WAIT_TIME);
    }

/*Error Processing*/
err:
    sem_trywait(&terminate_req_sem);
    main_ret = 1;
    goto main_proc_end;
/*Main Processing Termination*/
main_proc_end:
    printf("Main Process Terminated\n");
    return main_ret;
}

/* Initialize Serial */
int Init_Serial() {
    serial_port = open("/dev/ttyUSB0", O_RDWR);
    struct termios tty;
    // Read in existing settings, and handle any error
    if(tcgetattr(serial_port, &tty) != 0) {
        printf("Error %i from tcgetattr: %s\n", errno, strerror(errno));
        return 1;
    }
    tty.c_cflag &= ~PARENB; // Clear parity bit, disabling parity (most common)
    tty.c_cflag &= ~CSTOPB; // Clear stop field, only one stop bit used in communication (most common)
    tty.c_cflag &= ~CSIZE; // Clear all bits that set the data size
    tty.c_cflag |= CS8; // 8 bits per byte (most common)
    tty.c_cflag &= ~CRTSCTS; // Disable RTS/CTS hardware flow control (most common)
    tty.c_cflag |= CREAD | CLOCAL; // Turn on READ & ignore ctrl lines (CLOCAL = 1)

    tty.c_lflag &= ~ICANON;
    tty.c_lflag &= ~ECHO; // Disable echo
    tty.c_lflag &= ~ECHOE; // Disable erasure
    tty.c_lflag &= ~ECHONL; // Disable new-line echo
    tty.c_lflag &= ~ISIG; // Disable interpretation of INTR, QUIT and SUSP
    tty.c_iflag &= ~(IXON | IXOFF | IXANY); // Turn off s/w flow ctrl
    tty.c_iflag &= ~(IGNBRK|BRKINT|PARMRK|ISTRIP|INLCR|IGNCR|ICRNL); // Disable any special handling of received bytes

    tty.c_oflag &= ~OPOST; // Prevent special interpretation of output bytes (e.g. newline chars)
    tty.c_oflag &= ~ONLCR; // Prevent conversion of newline to carriage return/line feed
    // tty.c_oflag &= ~OXTABS; // Prevent conversion of tabs to spaces (NOT PRESENT ON LINUX)
    // tty.c_oflag &= ~ONOEOT; // Prevent removal of C-d chars (0x004) in output (NOT PRESENT ON LINUX)

    tty.c_cc[VTIME] = 1;    // Wait for up to 1s (10 deciseconds), returning as soon as any data is received.
    tty.c_cc[VMIN] = 0;

    // Set in/out baud rate to be 9600
    cfsetispeed(&tty, B115200);
    cfsetospeed(&tty, B115200);
    printf("Finish setting up serial, 115200");

    // Save tty settings, also checking for error
    if (tcsetattr(serial_port, TCSANOW, &tty) != 0) {
      printf("Error %i from tcsetattr: %s\n", errno, strerror(errno));
      return 1;
    }

    return 0;
}

int32_t main(int32_t argc, char * argv[])
{
    /* Log File Setting */
    auto now = std::chrono::system_clock::now();
    auto tm_time = spdlog::details::os::localtime(std::chrono::system_clock::to_time_t(now));
    char date_buf[64];
    char time_buf[128];
    memset(time_buf,0,sizeof(time_buf));
    std::strftime(date_buf, sizeof(date_buf), "%Y-%m-%d_%H-%M-%S", &tm_time);
    sprintf(time_buf,"logs/%s_app_hrnet_yolox_cam.log",date_buf);
    auto logger = spdlog::basic_logger_mt("logger", time_buf);
    spdlog::set_default_logger(logger);

    /* DRP-AI Frequency Setting */
    if (2 <= argc)
    {
        drp_max_freq = atoi(argv[1]);
    }
    else
    {
        drp_max_freq = DRP_MAX_FREQ_DEF;
    }
    if (3 <= argc)
    {
        drpai_freq = atoi(argv[2]);
    }
    else
    {
        drpai_freq = DRPAI_FREQ_DEF;
    }

    int8_t main_proc = 0;
    int8_t ret = 0;
    int8_t ret_main = 0;
    /*Multithreading Variables*/
    int32_t create_thread_ai = -1;
    int32_t create_thread_key = -1;
    int32_t create_thread_capture = -1;
    int32_t create_thread_img = -1;
    int32_t create_thread_hdmi = -1;
    int32_t sem_create = -1;


    printf("RZ/V2H DRP-AI Sample Application\n");
    printf("Model : MMPose HRNet | %s  \n", AI1_DESC_NAME);
#if (0) == INF_YOLOX_SKIP
    printf("Model : Megvii-Base Detection YOLOX | %s\n", AI0_DESC_NAME);
#endif
    printf("Input : %s\n", INPUT_CAM_NAME);
    spdlog::info("************************************************");
    spdlog::info("  RZ/V2H DRP-AI Sample Application");
    spdlog::info("  Model : MMPose HRNet with YOLOX | {} {}", AI1_DESC_NAME,AI0_DESC_NAME);
    spdlog::info("  Input : {}", INPUT_CAM_NAME);
    spdlog::info("************************************************");
    printf("Argument : <DRP0_max_freq_factor> = %d\n", drp_max_freq);
    printf("Argument : <AI-MAC_freq_factor> = %d\n", drpai_freq);

    if(Init_Serial() != 0) {
        return 1;
    }
    write(serial_port, ONE, sizeof(TWO));
    
    /*DRP-AI Driver Open*/
    /*For YOLOX*/
    errno = 0;
#if (0) == INF_YOLOX_SKIP
    drpai_fd0 = open("/dev/drpai0", O_RDWR);
    if (0 > drpai_fd0)
    {
        fprintf(stderr, "[ERROR] Failed to open DRP-AI Driver: errno=%d\n", errno);
        return -1;
    }
#endif
    /*For HRNet*/
    errno = 0;
    drpai_fd1 = open("/dev/drpai0", O_RDWR);
    if (0 > drpai_fd1)
    {
        fprintf(stderr, "[ERROR] Failed to open DRP-AI Driver: errno=%d\n", errno);
        return -1;
    }

#if (0) == INF_YOLOX_SKIP
    /* Get DRP-AI Memory Area Address via DRP-AI Driver */
    ret = ioctl(drpai_fd0, DRPAI_GET_DRPAI_AREA, &drpai_data0);	
    if (ret == -1)
    {
        fprintf(stderr, "[ERROR] Failed to get DRP-AI Memory Area: errno=%d\n", errno);
        ret_main = -1;
        goto end_close_drpai;
    }

    /* Load DRP-AI Data from Filesystem to Memory via DRP-AI Driver */
    /* YOLOX */
    drpai_hdl0 = load_drpai_obj_dynamic(drpai_fd0, AI0_DESC_NAME, drpai_data0.address);
    if (NULL == drpai_hdl0)
    {
        fprintf(stderr, "[ERROR] Failed to load DRP-AI Data\n");
        ret_main = -1;
        goto end_close_drpai;
    }
#endif
    
    /* HRNet */
    /* Get DRP-AI Memory Area Address via DRP-AI Driver */
#if (0) == INF_YOLOX_SKIP
    drpai_data1.address = (drpai_hdl0->data_inout.start_address + drpai_hdl0->data_inout.object_files_size + 0x1000000) & 0xFFFFFFFFFF000000;
#else
    /* Get DRP-AI Memory Area Address via DRP-AI Driver */
    ret = ioctl(drpai_fd1, DRPAI_GET_DRPAI_AREA, &drpai_data1);	
    if (ret == -1)
    {
        fprintf(stderr, "[ERROR] Failed to get DRP-AI Memory Area: errno=%d\n", errno);
        ret_main = -1;
        goto end_close_drpai;
    }
#endif
    drpai_hdl1 = load_drpai_obj_dynamic(drpai_fd1, AI1_DESC_NAME, drpai_data1.address);
    if (NULL == drpai_hdl1)
    {
        fprintf(stderr, "[ERROR] Failed to load DRP-AI Data\n");
        ret_main = -1;
        goto end_close_drpai;
    }

    /* Create Camera Instance */
    capture = new Camera();

    /* Init and Start Camera */
    ret = capture->start_camera();
    if (0 != ret)
    {
        fprintf(stderr, "[ERROR] Failed to initialize Camera.\n");
        delete capture;
        ret_main = ret;
        goto end_close_drpai;
    }

    /*Initialize Image object.*/
    ret = img.init(CAM_IMAGE_WIDTH, CAM_IMAGE_HEIGHT, CAM_IMAGE_CHANNEL_YUY2, IMAGE_OUTPUT_WIDTH, IMAGE_OUTPUT_HEIGHT, IMAGE_CHANNEL_BGRA, capture->wayland_buf->mem);
    if (0 != ret)
    {
        fprintf(stderr, "[ERROR] Failed to initialize Image object.\n");
        ret_main = ret;
        goto end_close_camera;
    }
    
    /*Termination Request Semaphore Initialization*/
    /*Initialized value at 1.*/
    sem_create = sem_init(&terminate_req_sem, 0, 1);
    if (0 != sem_create)
    {
        fprintf(stderr, "[ERROR] Failed to Initialize Termination Request Semaphore.\n");
        ret_main = -1;
        goto end_threads;
    }

    /*Create Key Hit Thread*/
    create_thread_key = pthread_create(&kbhit_thread, NULL, R_Kbhit_Thread, NULL);
    if (0 != create_thread_key)
    {
        fprintf(stderr, "[ERROR] Failed to create Key Hit Thread.\n");
        ret_main = -1;
        goto end_threads;
    }

    /*Create Inference Thread*/
    create_thread_ai = pthread_create(&ai_inf_thread, NULL, R_Inf_Thread, NULL);
    if (0 != create_thread_ai)
    {
        sem_trywait(&terminate_req_sem);
        fprintf(stderr, "[ERROR] Failed to create AI Inference Thread.\n");
        ret_main = -1;
        goto end_threads;
    }

    /*Create Capture Thread*/
    create_thread_capture = pthread_create(&capture_thread, NULL, R_Capture_Thread, (void *) capture);
    if (0 != create_thread_capture)
    {
        sem_trywait(&terminate_req_sem);
        fprintf(stderr, "[ERROR] Failed to create Capture Thread.\n");
        ret_main = -1;
        goto end_threads;
    }

    /*Create Image Thread*/
    create_thread_img = pthread_create(&img_thread, NULL, R_Img_Thread, NULL);
    if(0 != create_thread_img)
    {
        sem_trywait(&terminate_req_sem);
        fprintf(stderr, "[ERROR] Failed to create Image Thread.\n");
        ret_main = -1;
        goto end_threads;
    }

	/*Create Display Thread*/
    create_thread_hdmi = pthread_create(&hdmi_thread, NULL, R_Display_Thread, NULL);
    if(0 != create_thread_hdmi)
    {
        sem_trywait(&terminate_req_sem);
        fprintf(stderr, "[ERROR] Failed to create Display Thread.\n");
        ret_main = -1;
        goto end_threads;
    }

    /*Main Processing*/
    main_proc = R_Main_Process();
    if (0 != main_proc)
    {
        fprintf(stderr, "[ERROR] Error during Main Process\n");
        ret_main = -1;
    }
    goto end_threads;

end_threads:
    if(0 == create_thread_hdmi)
    {
        ret = wait_join(&hdmi_thread, DISPLAY_THREAD_TIMEOUT);
        if (0 != ret)
        {
            fprintf(stderr, "[ERROR] Failed to exit Display Thread on time.\n");
            ret_main = -1;
        }
    }
    if(0 == create_thread_img)
    {
        ret = wait_join(&img_thread, DISPLAY_THREAD_TIMEOUT);
        if (0 != ret)
        {
            fprintf(stderr, "[ERROR] Failed to exit Image Thread on time.\n");
            ret_main = -1;
        }
    }
    if (0 == create_thread_capture)
    {
        ret = wait_join(&capture_thread, CAPTURE_TIMEOUT);
        if (0 != ret)
        {
            fprintf(stderr, "[ERROR] Failed to exit Capture Thread on time.\n");
            ret_main = -1;
        }
    }
    if (0 == create_thread_ai)
    {
        ret = wait_join(&ai_inf_thread, AI_THREAD_TIMEOUT);
        if (0 != ret)
        {
            fprintf(stderr, "[ERROR] Failed to exit AI Inference Thread on time.\n");
            ret_main = -1;
        }
    }
    if (0 == create_thread_key)
    {
        ret = wait_join(&kbhit_thread, KEY_THREAD_TIMEOUT);
        if (0 != ret)
        {
            fprintf(stderr, "[ERROR] Failed to exit Key Hit Thread on time.\n");
            ret_main = -1;
        }
    }

    /*Delete Terminate Request Semaphore.*/
    if (0 == sem_create)
    {
        sem_destroy(&terminate_req_sem);
    }

    /* Exit waylad */
    wayland.exit();
    goto end_close_camera;

end_close_camera:
    /*Close USB Camera.*/
    ret = capture->close_camera();
    if (0 != ret)
    {
        fprintf(stderr, "[ERROR] Failed to close Camera.\n");
        ret_main = -1;
    }
    delete capture;
    goto end_close_drpai;

end_close_drpai:
    if (NULL != drpai_hdl0)
    {
        unload_drpai_object_dynamic(drpai_hdl0);
        drpai_hdl0 = NULL;
    }

    /*Close DRP-AI Driver.*/
    if (0 < drpai_fd0)
    {
        errno = 0;
        ret = close(drpai_fd0);
        if (0 != ret)
        {
            fprintf(stderr, "[ERROR] Failed to close DRP-AI Driver: errno=%d\n", errno);
            ret_main = -1;
        }
    }
    if (NULL != drpai_hdl1)
    {
        unload_drpai_object_dynamic(drpai_hdl1);
        drpai_hdl1 = NULL;
    }
    if (0 < drpai_fd1)
    {
        errno = 0;
        ret = close(drpai_fd1);
        if (0 != ret)
        {
            fprintf(stderr, "[ERROR] Failed to close DRP-AI Driver: errno=%d\n", errno);
            ret_main = -1;
        }
    }
    goto end_main;

end_main:
    printf("Application End\n");
    return ret_main;
}

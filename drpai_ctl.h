/***********************************************************************************************************************
* Copyright (C) 2023 Renesas Electronics Corporation. All rights reserved.
***********************************************************************************************************************/
/***********************************************************************************************************************
* File Name    : drpai_ctl.h
* Version      : 1.00
* Description  : RZ/V2H DRP-AI Sample Application for MMPose HRNet + Megvii-Base Detection YOLOX with MIPI/USB Camera
***********************************************************************************************************************/
#ifndef _DRPAI_CTL_H_
#define _DRPAI_CTL_H_

/*****************************************
* Includes
******************************************/
/*DRPAI Driver Header*/
#include <linux/drpai.h>
/*Definition of Macros & other variables*/
#include "define.h"

/*****************************************
* Typedef
******************************************/
/* For relocatable support of DRP-AI Object files */
typedef struct
{
    std::string*  directory_name;
    uint64_t      start_address;
    unsigned long object_files_size;
    unsigned long data_in_addr;
    unsigned long data_in_size;
    unsigned long data_out_addr;
    unsigned long data_out_size;
} st_drpai_data_t;

typedef struct
{
    unsigned long desc_aimac_addr;
    unsigned long desc_aimac_size;
    unsigned long desc_drp_addr;
    unsigned long desc_drp_size;
    unsigned long drp_param_addr;
    unsigned long drp_param_size;
    unsigned long data_in_addr;
    unsigned long data_in_size;
    unsigned long data_addr;
    unsigned long data_size;
    unsigned long work_addr;
    unsigned long work_size;
    unsigned long data_out_addr;
    unsigned long data_out_size;
    unsigned long drp_config_addr;
    unsigned long drp_config_size;
    unsigned long weight_addr;
    unsigned long weight_size;
    unsigned long aimac_cmd_addr;
    unsigned long aimac_cmd_size;
    unsigned long aimac_param_desc_addr;
    unsigned long aimac_param_desc_size;
    unsigned long aimac_param_cmd_addr;
    unsigned long aimac_param_cmd_size;
} st_addr_info_t;

typedef struct
{
    int8_t          drpai_fd;
    st_drpai_data_t data_inout;
    st_addr_info_t  drpai_address;
} drpai_handle_t;

drpai_handle_t* load_drpai_obj_dynamic(int8_t drpai_fd, std::string directory_name, uint64_t start_address);
int8_t start_drpai(drpai_handle_t* drpai_obj_info, uint64_t data_in, uint32_t max_freq_in, uint32_t freq_in);
void unload_drpai_object_dynamic(drpai_handle_t* drpai_hdl);

#endif // !_DRPAI_CTL_H_

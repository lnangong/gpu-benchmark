/**
 *	gpu_opencl.h
 *
 *	Copyright (c) 2014, Long(Ryan) Nangong.
 *	All right reserved.
 *
 *      Email: lnangong@hawk.iit.edu
 */


#ifndef _GPU_H_
#define _GPU_H_

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

//Select the OpenCL device to use (can be GPU, CPU, or Accelerator such as Intel Xeon Phi)
#define OPENCL_DEVICE_SELECTION CL_DEVICE_TYPE_ALL
#define MAX_SOURCE_SIZE (0x100000)

//Matrix parameters
unsigned int MATRIX_WIDTH;
unsigned int MATRIX_HIGHT;
unsigned int size_A;
unsigned int size_B;
unsigned int size_C;

//OpenCL parameters
char *source_str;
size_t source_size;

char str_temp[1024];
cl_platform_id platform_id;
cl_device_id device_id;
cl_uint num_devices;
cl_uint num_platforms;
cl_uint max_units;
cl_uint max_work_item_dims;
size_t max_work_group_size;
cl_int stat;

cl_context context;
cl_command_queue command_queue;
cl_program program;
cl_kernel kernel_1;
cl_kernel kernel_2;

cl_mem src_A;
cl_mem src_B;
cl_mem res_C;
cl_mem src_a;
cl_mem src_b;
cl_mem res_c;

#endif

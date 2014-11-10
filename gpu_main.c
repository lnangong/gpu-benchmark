/**********************************************************
	GPU benchmarking Program (OpenCL)
	--Test GPU speed in terms of floating point operations 
	per second and integer operations per second, GPU memory
	bandwidth (Host to device, device to host, device to devie)
	
	Copyright (c) 2014, Long(Ryan) Nangong.
 	All right reserved.
 
 	Email: lnangong@hawk.iit.edu
***********************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include "gpu_opencl.h"



double execTime(cl_event event){
	cl_ulong time_start, time_end;
        //Kernel start time
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
        //Kernel ends time
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

	double elapse = time_end - time_start;
        return (double) 1.0e-6 * elapse; // Return elapse time in milliseconds

}

void computePerform(double executionTime, int flag){
	if(!flag){
		printf("GPU speed in floating point operation.\n");
	}else
		printf("GPU speed in integer operation.\n");

        printf("Execution time in milliseconds = %.2f ms\n", executionTime);

        double size = (double) MATRIX_WIDTH;
	double speed = (double) 100000 * 2 * pow(size,3.0)  / (executionTime * 1.0e-3);	
	if(!flag){
		printf("GPU speed:%.2f [Giga FLOPS]\n",(double) 1.0e-9 * speed);
	}else
		printf("GPU speed:%.2f [Giga IOPS]\n",(double) 1.0e-9 * speed);

}

void matrixFloat(float *matrix, int size){
	for(int i=0; i < size; i++){
                matrix[i] =(float)rand() / 32768.0;
        }

}

void matrixInt(int *matrix, int size){
	for(int i=0; i < size; i++){
                matrix[i] = rand() % 100;
        }

}

void MatrixFloat(float *A, float *B){
	
	srand(time(NULL));
	matrixFloat(A,size_A);
	matrixFloat(B,size_B);

	if(MATRIX_WIDTH <= 8){
		int row;
		printf("A=\n");
		for(row = 0; row < size_A; row++) {
			if(row % MATRIX_WIDTH == 0) printf("\n");
			printf("%f ",A[row]);
		}
		printf("\nB=\n");
		for(row = 0; row < size_B; row++) {
			if(row % MATRIX_WIDTH == 0) printf("\n");
			printf("%f ",B[row]);
		}
	}

}

void MatrixInt(int *A, int *B){
	
	srand(time(NULL));
	matrixInt(A,size_A);
	matrixInt(B,size_B);

	if(MATRIX_WIDTH <= 8){
		int row;
		printf("A=\n");
		for(row = 0; row < size_A; row++) {
			if(row % MATRIX_WIDTH == 0) printf("\n");
			printf("%d ",A[row]);
		}
		printf("\nB=\n");
		for(row = 0; row < size_B; row++) {
			if(row % MATRIX_WIDTH == 0) printf("\n");
			printf("%d ",B[row]);
		}
	}

}


void getPlatformInfo(){
	/********* Get platform/device info ********/
	stat = clGetPlatformIDs(1, &platform_id, &num_platforms);
	if(stat == CL_SUCCESS) printf("number of platforms is %d\n",num_platforms);
	else printf("Error getting platform IDs\n");

	stat = clGetPlatformInfo(platform_id,CL_PLATFORM_NAME, sizeof(str_temp), str_temp,NULL);
	if(stat == CL_SUCCESS) printf("platform name is %s\n",str_temp);
	else printf("Error getting platform name\n");

	stat = clGetPlatformInfo(platform_id, CL_PLATFORM_VERSION, sizeof(str_temp), str_temp,NULL);
	if(stat == CL_SUCCESS) printf("platform version is %s\n",str_temp);
	else printf("Error getting platform version\n");

	stat = clGetDeviceIDs( platform_id, OPENCL_DEVICE_SELECTION, 1, &device_id, &num_devices);
	if(stat == CL_SUCCESS) printf("number of devices is %d\n", num_devices);
	else printf("Error getting device IDs\n");

	stat = clGetDeviceInfo(device_id,CL_DEVICE_NAME, sizeof(str_temp), str_temp,NULL);
	if(stat == CL_SUCCESS) printf("device name is %s\n",str_temp);
	else printf("Error getting device name\n");

	stat = clGetDeviceInfo(device_id,CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &max_units,NULL);
	if(stat == CL_SUCCESS) printf("maximun compute units are %d\n",max_units);
	else printf("Error getting device compute units\n");

	stat = clGetDeviceInfo(device_id,CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &max_work_item_dims,NULL);
        if(stat == CL_SUCCESS) printf("maximun work item dims are %d\n",max_work_item_dims);
        else printf("Error getting device maximum work item dims\n");

	stat = clGetDeviceInfo(device_id,CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size,NULL);
	if(stat == CL_SUCCESS) printf("maximun work group size is %zd\n",max_work_group_size);
	else printf("Error getting device maximum work group size\n");

}

void loadKernelSourceCode(){
/******* Load the kernel source code *******/
	FILE *fp;
	
	fp = fopen("gpu_kernel.cl", "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose(fp);

}

void initOpencl(){
/**************Initialize OpenCL***************/
	// Create an OpenCL context
	context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &stat);
	if(stat != CL_SUCCESS) printf("Context error!\n");

	// Create a command queue
	command_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &stat);
	if(stat != CL_SUCCESS) printf("Command queue error!\n");

	// Create a program from the kernel source
	program = clCreateProgramWithSource(context, 1, 
	    (const char **)&source_str, (const size_t *)&source_size, &stat);
	if(stat != CL_SUCCESS) printf("Create program error!\n");

	// Build the program
	stat = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
	if(stat != CL_SUCCESS) printf("Build program error!\n");

	// Create the OpenCL kernel
	kernel_1 = clCreateKernel(program, "matrix_mult_float", &stat);
	if(stat != CL_SUCCESS) printf("Create kernel error!\n");
	// Create the OpenCL kernel
	kernel_2 = clCreateKernel(program, "matrix_mult_int", &stat);
	if(stat != CL_SUCCESS) printf("Create kernel error!\n");

}

void memoryBenchmark(char *message, int size){
/*************GPU Memory Read/Write Bandwidth Test******************/
	// Set buffer for memory bandwidth test
	cl_mem mem_buf = clCreateBuffer(context, CL_MEM_READ_WRITE,
             size*sizeof(char), NULL, &stat);

	int memcpy_iteration;
	double elapsedTime;
	cl_event event;
	
	memcpy_iteration = (int)pow(10,5);

	// Memory write from Host to Device 
	for(int i = 0; i < memcpy_iteration; i++){
                stat = clEnqueueWriteBuffer(command_queue, mem_buf, CL_FALSE, 0,
            		size*sizeof(char), message, 0, NULL, &event);
		if(stat != CL_SUCCESS) printf("Memory buffer error!\n");

		clWaitForEvents(1,&event);
                elapsedTime += execTime(event);
        }
	printf("memcpy time = %.2f ms\n",elapsedTime);

	if(size == 1){ 
		printf("Host to Device memory write bandwidth [1Byte message] = %.3f MB/sec\n", 
			(double)size * memcpy_iteration * 1.0e-6 / (elapsedTime * 1.0e-3));
	}else if(size == 1024){ 
		printf("Host to Device memory write bandwidth [1KByte message] = %.3f MB/sec\n", 
			(double)size * memcpy_iteration * 1.0e-6 / (elapsedTime * 1.0e-3));
	}else{
		printf("Host to Device memory write bandwidth [1MByte message] = %.3f MB/sec\n", 
			(double)size * memcpy_iteration * 1.0e-6 / (elapsedTime * 1.0e-3));
	}

	// Memory read from Device to Host 
	for(int i = 0; i < memcpy_iteration; i++){
         	stat = clEnqueueReadBuffer(command_queue, mem_buf, CL_FALSE, 0, 
	    		size*sizeof(char), message, 0, NULL, &event);
       		if(stat != CL_SUCCESS) printf("Memory buffer error!\n");

                clWaitForEvents(1,&event);
                elapsedTime += execTime(event);
        }
        printf("memcpy time = %.2f ms\n",elapsedTime);

        if(size == 1){
                printf("Device to Host memory read bandwidth [1Byte message] = %.3f MB/sec\n",
                        (double)size * memcpy_iteration * 1.0e-6 / (elapsedTime * 1.0e-3));
        }else if(size == 1024){
                printf("Device to Host memory read bandwidth [1KByte message] = %.3f MB/sec\n",
                        (double)size * memcpy_iteration * 1.0e-6 / (elapsedTime * 1.0e-3));
        }else{
                printf("Device to Host memory read bandwidth [1MByte message] = %.3f MB/sec\n",
                        (double)size * memcpy_iteration * 1.0e-6 / (elapsedTime * 1.0e-3));
        }

	// Set memory copy buffer
	cl_mem mem_copy_buf = clCreateBuffer(context, CL_MEM_READ_WRITE,
             size*sizeof(char), NULL, &stat);

	// Memory Device to Device copy
	for(int i = 0; i < memcpy_iteration; i++){
         	stat = clEnqueueCopyBuffer(command_queue, mem_buf, mem_copy_buf, 0, 0, 
	    		size*sizeof(char), 0, NULL, &event);
       		if(stat != CL_SUCCESS) printf("Memory buffer error!\n");

                clWaitForEvents(1,&event);
                elapsedTime += execTime(event);
        }
        printf("memcpy time = %.2f ms\n",elapsedTime);

        if(size == 1){
                printf("Device to Device memory copy bandwidth [1Byte message] = %.3f MB/sec\n",
                        (double)size * memcpy_iteration * 1.0e-6 / (elapsedTime * 1.0e-3));
        }else if(size == 1024){
                printf("Device to Device memory copy bandwidth [1KByte message] = %.3f MB/sec\n",
                        (double)size * memcpy_iteration * 1.0e-6 / (elapsedTime * 1.0e-3));
        }else{
                printf("Device to Device memory copy bandwidth [1MByte message] = %.3f MB/sec\n",
                        (double)size * memcpy_iteration * 1.0e-6 / (elapsedTime * 1.0e-3));
        }

	//release event object
	clReleaseEvent(event);
	clFinish(command_queue);

}

void setMemBuffer(float *A, float *B, int *a, int *b){
	// Create memory buffers on the device for each vector 
	src_A = clCreateBuffer(context, CL_MEM_READ_ONLY, 
	     size_A * sizeof(float), NULL, &stat);
	src_B = clCreateBuffer(context, CL_MEM_READ_ONLY,
	     size_B * sizeof(float), NULL, &stat);
	res_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
	     size_C * sizeof(float), NULL, &stat);
	if(stat != CL_SUCCESS) printf("Memory buffer error!\n");

	// Copy the A and B to their respective memory buffers
	stat = clEnqueueWriteBuffer(command_queue, src_A, CL_TRUE, 0,
	    size_A * sizeof(float), A, 0, NULL, NULL);
	stat = clEnqueueWriteBuffer(command_queue, src_B, CL_TRUE, 0, 
	    size_B * sizeof(float), B, 0, NULL, NULL);
	if(stat != CL_SUCCESS) printf("Copy memory buffer error!\n");
	
	// Create memory buffers on the device for each vector 
	src_a = clCreateBuffer(context, CL_MEM_READ_ONLY, 
	     size_A * sizeof(int), NULL, &stat);
	src_b = clCreateBuffer(context, CL_MEM_READ_ONLY,
	     size_B * sizeof(int), NULL, &stat);
	res_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
	     size_C * sizeof(int), NULL, &stat);
	if(stat != CL_SUCCESS) printf("Memory buffer error!\n");

	// Copy the A and B to their respective memory buffers
	stat = clEnqueueWriteBuffer(command_queue, src_a, CL_TRUE, 0,
	    size_A * sizeof(int), a, 0, NULL, NULL);
	stat = clEnqueueWriteBuffer(command_queue, src_b, CL_TRUE, 0, 
	    size_B * sizeof(int), b, 0, NULL, NULL);
	if(stat != CL_SUCCESS) printf("Copy memory buffer error!\n");

}

	
void kernelBenchmark(int  flag){
	/****************** Launch Kernel ******************/
	cl_event event;
	size_t global_work_size[2], local_work_size[2];
	size_t group_size;
	double elapse;

	// Query the optimal work group size
	clGetKernelWorkGroupInfo(kernel_1, device_id, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, 
		sizeof(size_t), &group_size, NULL);
	printf("optimal local work size = %zd\n",group_size);

	global_work_size[0] = MATRIX_WIDTH;
	global_work_size[1] = MATRIX_HIGHT;
	local_work_size[0] = 32;
	local_work_size[1] = 32;

	if(!flag){
		// Set the arguments of the kernel 1
		cl_uint size = MATRIX_WIDTH;
		stat = clSetKernelArg(kernel_1, 0, sizeof(cl_mem), (void *)&src_A);
		stat = clSetKernelArg(kernel_1, 1, sizeof(cl_mem), (void *)&src_B);
		stat = clSetKernelArg(kernel_1, 2, sizeof(cl_mem), (void *)&res_C);
		stat = clSetKernelArg(kernel_1, 3, sizeof(cl_uint), (void *)&size);
		if(stat != CL_SUCCESS) printf("Set kernel arguments error!\n");
		
		//Ensure to have executed all enqueued tasks
		clFinish(command_queue);

		// Execute the OpenCL kernel 1
		stat = clEnqueueNDRangeKernel(command_queue, kernel_1, 2, NULL, 
		    global_work_size, local_work_size, 0, NULL, &event);
		if(stat != CL_SUCCESS) printf("Execute kernel error!\n");

		// Waiting for kernel 1 to complete
		clWaitForEvents(1,&event);
		elapse = execTime(event); // Get kernel execution time
		// Performance evaluation
		computePerform(elapse,0);
	}
	
	else if(flag){
		// Set the arguments of the kernel 2
		cl_uint size = MATRIX_WIDTH;
		stat = clSetKernelArg(kernel_2, 0, sizeof(cl_mem), (void *)&src_a);
		stat = clSetKernelArg(kernel_2, 1, sizeof(cl_mem), (void *)&src_b);
		stat = clSetKernelArg(kernel_2, 2, sizeof(cl_mem), (void *)&res_c);
		stat = clSetKernelArg(kernel_2, 3, sizeof(cl_uint), (void *)&size);
		if(stat != CL_SUCCESS) printf("Set kernel arguments error!\n");
		
		//Ensure to have executed all enqueued tasks
		clFinish(command_queue);

		// Execute the OpenCL kernel 2
		stat = clEnqueueNDRangeKernel(command_queue, kernel_2, 2, NULL, 
		    global_work_size, local_work_size, 0, NULL, &event);
		if(stat != CL_SUCCESS) printf("Execute kernel error!\n");

		// Waiting for kernel 2 to complete
		clWaitForEvents(1,&event);
		elapse = execTime(event); // Get kernel execution time
		// Performance evaluation
		computePerform(elapse,1);
	}
/*************debugging util**********************************************	
	// Read the memory buffer C on the device to the local variable C
	stat = clEnqueueReadBuffer(command_queue, res_c, CL_TRUE, 0, 
	    size_C * sizeof(float), C, 0, NULL, NULL);
	if(stat != CL_SUCCESS) printf("Read memory buffer error!\n");

	// Display the result to debug
	if(MATRIX_WIDTH <= 8){
        	int row;
                printf("\nResult C=\n");
                for(row = 0; row < size_A; row++) {
                        if(row % MATRIX_WIDTH == 0) printf("\n");
                        printf("%f ",C[row]);
                }
        }
************************************************************************/	
	//release event object
	clReleaseEvent(event);

}


int main(int argc, char **argv){
	// Test messages
	char *msg_1B = (char*)malloc(sizeof(char));
	char *msg_1KB = (char*)malloc(sizeof(char)*1024);
	char *msg_1MB = (char*)malloc(sizeof(char)*pow(1024,2.0));
	memset(msg_1B,'a',1);
	memset(msg_1KB,'a',1024);
	memset(msg_1MB,'a',(size_t)pow(1024,2.0));

	getPlatformInfo();

	//Allocate memory for Matrix A and B
	MATRIX_WIDTH = 160; //max_units * sqrt((double)max_work_group_size);
	MATRIX_HIGHT = 160; //max_units * sqrt((double)max_work_group_size);
	size_A = MATRIX_WIDTH * MATRIX_HIGHT;
	size_B = size_A;
	size_C = size_A;
	float *A = (float*)malloc(sizeof(float) * size_A);
        float *B = (float*)malloc(sizeof(float) * size_B);
	float *C = (float*)malloc(sizeof(float) * size_C);
	int *a = (int*)malloc(sizeof(int) * size_A);
        int *b = (int*)malloc(sizeof(int) * size_B);
	int *c = (int*)malloc(sizeof(int) * size_C);
	int flag;
	MatrixFloat(A,B);
	MatrixInt(a,b);

	loadKernelSourceCode();
	initOpencl();

	printf("\nBenchmark result......\n");
	//memory benchmark
	int msg_size = 1;
	if (argc == 1){
		printf("command option missing, please refer to the Readme document!\n");
	}
	else if(strcmp(argv[1],"-m1b") == 0){
		memoryBenchmark(msg_1B, msg_size);
	}
	else if(strcmp(argv[1],"-m1kb") == 0){
		msg_size = 1024;
		memoryBenchmark(msg_1KB, msg_size);
	}
	else if(strcmp(argv[1],"-m1mb") == 0){
		msg_size = 1024 * 1024;
		memoryBenchmark(msg_1MB, msg_size);
	}
	else if(strcmp(argv[1],"-sf") == 0){
		printf("\nWait a moment...\n");
		//kernel benchmark
		setMemBuffer( A, B, a, b );
		flag = 0;
		kernelBenchmark(flag);
	}
	else if(strcmp(argv[1],"-si") == 0){
		printf("\nWait a moment...\n");
                //kernel benchmark
                setMemBuffer( A, B, a, b );
		flag = 1;
                kernelBenchmark(flag);
	}else{
		printf("Option not recognized, please refer to the Readme document!\n");
	}

	//Clean up
	clFlush(command_queue);
	clFinish(command_queue);
	clReleaseKernel(kernel_1);
	clReleaseKernel(kernel_2);
	clReleaseProgram(program);
	clReleaseMemObject(src_a);
	clReleaseMemObject(src_b);
	clReleaseMemObject(res_c);
	clReleaseMemObject(src_A);
	clReleaseMemObject(src_B);
	clReleaseMemObject(res_C);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);
	free(A);
	free(B);
	free(C);
	free(a);
	free(b);
	free(c);

	return 0;
}


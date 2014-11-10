/**
 * 	gpu_kernel.cl
 *
 *	Copyright (c) 2014, Long(Ryan) Nangong.
 *	All right reserved.
 *
 *      Email: lnangong@hawk.iit.edu
 */


#define ITERATION 100000

__kernel void matrix_mult_float(__global float *A, __global float *B, __global float *C, int size) {

	//get the index of the current processing element
	int idx = get_global_id(0);
	int idy = get_global_id(1);

	// Do the operation
	if((idx < size) && (idy < size)){
		float sum = 0;
		for(int offset = 0; offset < size; offset++)
			for(int i = 0; i < ITERATION; i++)
				sum += A[idx * size + offset] * B[offset * size + idy];
		
		C[idx * size + idy] = sum;
	}

}

__kernel void matrix_mult_int(__global int *A, __global int *B, __global int *C, int size) {

	//get the index of the current processing element
	int idx = get_global_id(0);
	int idy = get_global_id(1);

	// Do the operation
	if((idx < size) && (idy < size)){
		int sum = 0;
		for(int offset = 0; offset < size; offset++)
			for(int i = 0; i < ITERATION; i++)
				sum += A[idx * size + offset] * B[offset * size + idy];
		
		C[idx * size + idy] = sum;
	}

}

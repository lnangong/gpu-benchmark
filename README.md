GPU-Benchmarking-OpenCL
=======================
GPU:
You may need to download and install the OpenCL platform before running this benchmarking program.
Please download Nvidia OpenCL platform from https://developer.nvidia.com/cuda-downloads 
or Intel OpenCL plaform from https://software.intel.com/en-us/intel-opencl
based on your Graphic card's vendor.

Run memory benchmarking {host->device/device->host/device->device}
Test with 1byte message
#./gpu_start -m1b
Test with 1kbyte message
#./gpu_start -m1kb
Test with 1Mbyte message
#./gpu_start -m1mb

Run GPU speed benchmarking 
Floating point operation test
#./gpu_start -sf
Integer operation test
#./gpu_start -si

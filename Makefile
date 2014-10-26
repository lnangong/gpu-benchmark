OPENCLINC= -I/usr/local/cuda/include
OPENCLLIB= -L/usr/lib -L/usr/lib64

OS := $(shell uname)
OPTIONS:= 

ifeq ($(OS),Darwin)
	OPTIONS += -framework OpenCL
else
	OPTIONS += -l OpenCL
endif

all: gpu_start

gpu_start: gpu_main.c gpu_opencl.h
	gcc -std=c99 -Wall -g -o gpu_start gpu_main.c ${OPENCLINC} ${OPENCLLIB} $(OPTIONS) -lm


clean:
	rm gpu_start

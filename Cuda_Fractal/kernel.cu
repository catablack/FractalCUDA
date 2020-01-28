#pragma once
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/timeb.h>
#include "tinycthread.h"
#include "util.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/* This should be conveted into a GPU kernel */
__global__ void generate_image(unsigned char* dev_image, unsigned char* dev_colormap)
{
	int row, col, index, iteration;
	double c_re, c_im, x, y, x_new;

	unsigned char* image = dev_image;
	unsigned char* colormap = dev_colormap;

	int width = WIDTH;
	int height = HEIGHT;
	int max = MAX_ITERATION;

	int blockId = blockIdx.y * gridDim.x + blockIdx.x; // global block id
	index = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;// pixel index for the thread 

	if (index >= width * height) return;

	row = index / WIDTH;
	col = index % WIDTH;

	c_re = (col - width / 2.0) * 4.0 / width;
	c_im = (row - height / 2.0) * 4.0 / width;

	x = 0, y = 0;
	iteration = 0;
	while (x * x + y * y <= 4 && iteration < max) {
		x_new = x * x - y * y + c_re;
		y = 2 * x * y + c_im;
		x = x_new;
		iteration++;
	}

	if (iteration > max) {
		iteration = max;
	}

	image[4 * index + 0] = colormap[iteration * 3 + 0];
	image[4 * index + 1] = colormap[iteration * 3 + 1];
	image[4 * index + 2] = colormap[iteration * 3 + 2];
	image[4 * index + 3] = 255;

}

int main(int argc, char** argv) {
	
	double times[REPEAT];
	struct timeb start[REPEAT], end[REPEAT];
	char path[255];

	unsigned char* colormap;
	unsigned char* image;

	cudaMallocManaged(&colormap, (MAX_ITERATION + 1) * 3);
	cudaMallocManaged(&image, WIDTH * HEIGHT * 4);

	init_colormap(MAX_ITERATION, colormap);

	dim3 grid(GRID_SIZE_X, GRID_SIZE_Y);
	dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);

	for (int i = 0; i < REPEAT; i++) {
		ftime(&start[i]);
		
		generate_image <<<grid, block >>> (image, colormap);

		cudaDeviceSynchronize();
		
		ftime(&end[i]);
		
		times[i] =  end[i].time - start[i].time + ((double)end[i].millitm - (double)start[i].millitm) / 1000.0;

		sprintf(path, IMAGE, "gpu", i);
		save_image(path, image, WIDTH, HEIGHT);
		progress("gpu", i, times[i]);

	}

	report("gpu", times);
	
	printf("\nDONE!!!");

	cudaFree(image);
	cudaFree(colormap);

	return 0;
}

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <iostream>

using namespace std;

cudaError_t histEqCudaKernel(unsigned char* image, int imageSize, unsigned char* result, float* kernelTime);
cudaError_t queryDevice(bool* cudaAvailable);

//Histogram equalization on 8-bit grayscale image using CUDA
//Sizes of image and result must agree
int histEqCuda(unsigned char* image, int imageSize, unsigned char* result) {

	float kernelTime;

	cudaError_t cudaStatus = histEqCudaKernel(image, imageSize, result, &kernelTime);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "histEqCuda failed!");
		return 1;
	}

	cout << "GPU CUDA运行时间: " << kernelTime << "ms" << endl;

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}
}

int histEqCudaBench(unsigned char* image, int imageSize, unsigned char* result, int loopCount) {

	float* kernelTime = new float [loopCount];
	for (int i = 0; i < loopCount; i++) {
		cudaError_t cudaStatus = histEqCudaKernel(image, imageSize, result, &(kernelTime[i]));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "histEqCuda failed!");
			return 1;
		}
	}

	//运行时间
	float time = 0;
	for (int i = 0; i < loopCount; i++) time += kernelTime[i];
	cout << "GPU CUDA运行时间：" << time/loopCount << "ms" << endl;

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaError_t cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}
}

bool isCudaAvailable() {
	bool cudaAvailable = false;
	cudaError_t cudaStatus = queryDevice(&cudaAvailable);
	return cudaAvailable;
}
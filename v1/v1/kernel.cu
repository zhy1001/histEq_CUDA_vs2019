
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

//One block for processing BLOCK_SIZE pixels
#define BLOCK_SIZE (4 * 256)
#define STRIDE 32

//Generate histogram on GPU
__global__ void genHist(unsigned int* input, int* hist) {

	//Using shared memory for local histogram
	__shared__ int blockHist[256];
	blockHist[threadIdx.x] = 0;
	__syncthreads();

	int x = blockIdx.x * STRIDE * blockDim.x + threadIdx.x;

	for (int i = 0; i < STRIDE; ++i) {

		int location = x + i * blockDim.x;

		unsigned int in = input[location];
		unsigned char temp0 = (in & 0xFF000000) >> 24;
		unsigned char temp1 = (in & 0x00FF0000) >> 16;
		unsigned char temp2 = (in & 0x0000FF00) >> 8;
		unsigned char temp3 = in & 0x000000FF;

		atomicAdd(&blockHist[temp0], 1);
		atomicAdd(&blockHist[temp1], 1);
		atomicAdd(&blockHist[temp2], 1);
		atomicAdd(&blockHist[temp3], 1);

	}
	__syncthreads();

	atomicAdd(&hist[threadIdx.x], blockHist[threadIdx.x]);
}

__global__ void genHist2(unsigned char* input, int numPixel, int* hist) {

	int x = blockIdx.x * blockDim.x + threadIdx.x;

	//Calculate Histogram
	if (x < numPixel) {
		atomicAdd(&hist[input[x]], 1);
	}
}

//Generate look-up table on GPU using histogram equalization
//NUMBER OF THREADS MUST BE 256
__global__ void genLUT(int* hist, int imageSize, unsigned char* lut) {

	int location = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ int cdfHist[256];
	__shared__ int tempHist[256];
	__shared__ int mincdf;

	tempHist[location] = hist[location];
	__syncthreads();

	//Accumulate
	int cdfTemp = 0;
	int i = location;
	do {
		cdfTemp += tempHist[i--];
	} while (i >= 0);
	cdfHist[location] = cdfTemp;
	__syncthreads();

	//Find minimum CDF
	if (threadIdx.x == 0) {
		int j = 0;
		while (j < 256 && cdfHist[j] == 0) {
			++j;
		}
		mincdf = j;
	}
	__syncthreads();

	//Generate look-up table
	float lutf = 0;
	if (location > mincdf) {
		lutf = 255.0 * (cdfHist[location] - cdfHist[mincdf]) / (imageSize - cdfHist[mincdf]);
	}
	//Write look-up table
	lut[location] = (unsigned char)roundf(lutf);
}

__global__ void applyLUT(unsigned int* input, unsigned char* lut, unsigned int* output) {

	//Copy look-up table from global memory to shared memory
	__shared__ unsigned char lutTemp[256];
	lutTemp[threadIdx.x] = lut[threadIdx.x];
	__syncthreads();

	int x = blockIdx.x * STRIDE * blockDim.x + threadIdx.x;

	for (int i = 0; i < STRIDE; ++i) {

		int location = x + i * blockDim.x;

		unsigned int in = input[location];

		unsigned char temp0 = lutTemp[(in & 0xFF000000) >> 24];
		unsigned char temp1 = lutTemp[(in & 0x00FF0000) >> 16];
		unsigned char temp2 = lutTemp[(in & 0x0000FF00) >> 8];
		unsigned char temp3 = lutTemp[(in & 0x000000FF)];

		unsigned int out = (((unsigned int)temp0) << 24) + (((unsigned int)temp1) << 16) + (((unsigned int)temp2) << 8) + ((unsigned int)temp3);

		output[location] = out;
	}
}

__global__ void applyLUT2(unsigned char* input, int numPixel, unsigned char* lut, unsigned char* output) {

	int x = blockIdx.x * blockDim.x + threadIdx.x;

	//Generate new gray value
	if (x < numPixel) {
		output[x] = lut[input[x]];
	}
}

//Histogram equalization on 8-bit grayscale image using CUDA
//Sizes of image and result must agree
//CUDA core function and kernel
cudaError_t histEqCudaKernel(unsigned char* image, int imageSize, unsigned char* result, float* kernelTime)
{

	cudaError_t cudaStatus;

	//Pointer to GPU memory
	unsigned char* input_gpu;
	unsigned char* output_gpu;
	int* hist;
	unsigned char* lut;

	//Choose which GPU to run on
	//Change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	//Allocate GPU global memories for image, histogram, and look-up table
	cudaStatus = cudaMalloc((void**)& input_gpu, imageSize * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)& output_gpu, imageSize * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)& hist, 256 * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)& lut, 256 * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	//Set histogram and look-up table to all 0s
	cudaStatus = cudaMemset(hist, 0, 256 * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemset failed!");
		goto Error;
	}

	cudaStatus = cudaMemset(lut, 0, 256 * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemset failed!");
		goto Error;
	}

	//Copy image (unsigned char array) from host memory to GPU memory
	cudaStatus = cudaMemcpy(input_gpu, image, imageSize * sizeof(unsigned char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d before launching CUDA kernel!\n", cudaStatus);
		goto Error;
	}

	//Variables for CUDA timing
	cudaEvent_t gpu_start, gpu_stop;
	cudaEventCreate(&gpu_start);
	cudaEventCreate(&gpu_stop);

	//Define CUDA kernel launch parameters
	//To improve memory throughput, use unsigned int to combine 4 pixels
	int gridSize = imageSize / (STRIDE * BLOCK_SIZE);
	int restPixel = imageSize % (STRIDE * BLOCK_SIZE);
	int lutOffset = gridSize * STRIDE * BLOCK_SIZE;

	//Define CUDA grids and blocks
	dim3 dimGrid(gridSize);
	dim3 dimGridOne(1);

	//For look-up table operation, use 256 blocks per grid
	dim3 dimBlock256(256);

	//Kernel Call
	cudaEventRecord(gpu_start);

	//Devide histogram calculation into 2 parts for speed
	if (gridSize > 0) genHist <<<dimGrid, dimBlock256 >>> ((unsigned int*)input_gpu, hist);

	if (restPixel != 0) {
		//Optimized block size: 256
		int gridSize2 = (restPixel - 1) / 256 + 1;
		dim3 dimGrid2(gridSize2);

		genHist2 <<<dimGrid2, dimBlock256 >>> (input_gpu + lutOffset, restPixel, hist);
		genLUT <<<dimGridOne, dimBlock256 >>> (hist, imageSize, lut);
	}

	else {
		genLUT <<<dimGridOne, dimBlock256 >>> (hist, gridSize * STRIDE * BLOCK_SIZE, lut);
	}

	//Devide look-up table reference into 2 parts for speed
	if (gridSize > 0) applyLUT <<<dimGrid, dimBlock256 >>> ((unsigned int*)input_gpu, lut, (unsigned int*)output_gpu);

	if (restPixel != 0) {
		//Optimized block size: 256
		int gridSize2 = (restPixel - 1) / 256 + 1;
		dim3 dimGrid2(gridSize2);

		applyLUT2 <<<dimGrid2, dimBlock256 >>> (input_gpu + lutOffset, restPixel, lut, output_gpu + lutOffset);
	}

	cudaEventRecord(gpu_stop);
	cudaEventSynchronize(gpu_stop);
	cudaEventElapsedTime(kernelTime, gpu_start, gpu_stop);

	cudaEventDestroy(gpu_start);
	cudaEventDestroy(gpu_stop);

	//Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "CUDA kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching CUDA kernel!\n", cudaStatus);
		goto Error;
	}

	//Copy image(unsigned char array) back from GPU memory to host memory
	cudaStatus = cudaMemcpy(result, output_gpu, imageSize * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(input_gpu);
	cudaFree(output_gpu);
	cudaFree(hist);
	cudaFree(lut);

	return cudaStatus;
}


//Query CUDA device
cudaError_t queryDevice(bool* cudaAvailable) {

	int deviceCount = 0;
	cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

	if (error_id != cudaSuccess) {
		printf("cudaGetDeviceCount returned %d\n-> %s\n",
			static_cast<int>(error_id), cudaGetErrorString(error_id));
		*cudaAvailable = false;
		return error_id;
	}

	//This function call returns false if there are no CUDA capable devices
	if (deviceCount == 0) {
		printf("There are no available device(s) that support CUDA.\n");
		*cudaAvailable = false;
		return error_id;
	}
	else {
		printf("Detected %d CUDA Capable device(s).\n", deviceCount);
		*cudaAvailable = true;

		for (int dev = 0; dev < deviceCount; ++dev) {
			cudaSetDevice(dev);
			cudaDeviceProp deviceProp;
			cudaGetDeviceProperties(&deviceProp, dev);

			printf("Device %d: %s\n\n", dev, deviceProp.name);
		}

		return error_id;
	}
}

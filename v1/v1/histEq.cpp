
#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm>
#include <chrono>

using namespace cv;
using namespace std;

//Histogram equalization on 8-bit grayscale image
//Sizes of image and result must agree
void histEq(unsigned char* image, int imageSize, unsigned char* result) {

	//统计直方图
	int hist[256] = { 0 };
	int mincdf = 255;
	for (int i = 0; i < imageSize; ++i) {
		unsigned char pixel = image[i];
		mincdf = min(mincdf, int(pixel));
		++hist[pixel];
	}

	//统计累计量
	int cdfHist[256] = { 0 };
	cdfHist[mincdf] = hist[mincdf];
	for (int i = mincdf + 1; i < 256; ++i) {
		cdfHist[i] = cdfHist[i-1] + hist[i];
	}

	//生成look-up table
	double lutTemp[256] = { 0 };
	for (int i = mincdf + 1; i < 256; ++i) {
		lutTemp[i] = 255.0 * (cdfHist[i] - cdfHist[mincdf]) / (imageSize - cdfHist[mincdf]);
	}
	unsigned char lut[256] = { 0 };
	for (int i = 0; i < 256; ++i) {
		lut[i] = round(lutTemp[i]);
	}

	//进行直方图均一化（套用LUT）
	for (int i = 0; i < imageSize; ++i) {
		result[i] = lut[image[i]];
	}
}

void histEqBench(unsigned char* image, int imageSize, unsigned char* result, int loopCount) {

	auto start = chrono::steady_clock::now();
	for (int i = 0; i < loopCount; i++) {
		histEq(image, imageSize, result);
	}
	auto end = chrono::steady_clock::now();

	//运行时间
	auto elapsed = chrono::duration_cast<chrono::microseconds>(end - start);
	cout << "CPU运行时间：" << elapsed.count() / (loopCount * 1000.0) << "ms" << endl;
}

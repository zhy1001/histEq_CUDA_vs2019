
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

using namespace cv;
using namespace std;

//读入黑白图像
Mat readImgGray(string file) {

	Mat img = imread(file, IMREAD_GRAYSCALE);
	return img;
}

//使用CPU进行Histogram Equalization
//返回处理后的图像
//基准
Mat histogramEq(Mat img, bool display) {

	//直方图均衡化
	Mat result;
	equalizeHist(img, result);

	//创建窗口显示图片
	if (display) {
		namedWindow("原图片", WINDOW_NORMAL);
		imshow("原图片", img);
		namedWindow("处理后图片", WINDOW_NORMAL);
		imshow("处理后图片", result);
		waitKey(0);
		destroyWindow("原图片");
		destroyWindow("处理后图片");
	}

	return result;
}

//Benchmark: 使用CPU
int histogramEqCPU(Mat img, int loopCount) {

	//CPU预热
	Mat warmup;
	equalizeHist(img, warmup);

	//直方图均衡化
	Mat result;
	auto start = chrono::steady_clock::now();
	for (int i = 0; i < loopCount; i++) {
		equalizeHist(img, result);
	}
	auto end = chrono::steady_clock::now();

	//运行时间
	auto elapsed = chrono::duration_cast<std::chrono::microseconds>(end - start);
	cout << "CPU运行时间：" << elapsed.count() / (loopCount * 1000.0) << "ms" << endl;

	return 0;
}

//Benchmark: 使用OpenCL
int histogramEqCL(string file, int loopCount) {

	//读入黑白图像
	UMat img;
	imread(file, IMREAD_GRAYSCALE).copyTo(img);
	if (img.empty()) {
		cout << "无法载入图片！" << endl;
		return 1;
	}

	//GPU预热
	UMat warmup;
	equalizeHist(img, warmup);

	//直方图均衡化
	UMat result;
	auto start = chrono::steady_clock::now();
	for (int i = 0; i < loopCount; i++) {
		equalizeHist(img, result);
	}
	auto end = chrono::steady_clock::now();

	//运行时间
	auto elapsed = chrono::duration_cast<chrono::microseconds>(end - start);
	cout << "GPU OpenCL运行时间：" << elapsed.count() / (loopCount * 1000.0) << "ms" << endl;

	return 0;
}

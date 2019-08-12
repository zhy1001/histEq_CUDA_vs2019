
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <iostream>

using namespace cv;
using namespace std;

Mat readImgGray(string file);
Mat histogramEq(Mat image, bool display);

void histEq(unsigned char* image, int imageSize, unsigned char* result);
void histEqBench(unsigned char* image, int imageSize, unsigned char* result, int loopCount);

int histogramEqCPU(Mat image, int loopCount);
int histogramEqCL(string file, int loopCount);

bool isCudaAvailable();
int histEqCuda(unsigned char* image, int imageSize, unsigned char* result);
int histEqCudaBench(unsigned char* image, int imageSize, unsigned char* result, int loopCount);

int main(int argc, char** argv)
{
	if (argc != 2) {
		cout << "Usage: <EXE_FILE> \"IMAGE_PATH\"" << endl;
		return 1;
	}
	else {
		cout << "从文件中载入图像: " << argv[1] << "..." << endl;
	}

	string image = argv[1];

	Mat src = readImgGray(image);
	if (src.empty()) {
		cout << "图片读取错误！程序退出！" << endl;
		return 1;
	}

	//将OpenCV Mat转换为C++数组
	int imgWidth = src.cols;
	int imgHeight = src.rows;
	int imgSize = imgWidth * imgHeight;
	unsigned char* img = new unsigned char[imgSize];
	memcpy(img, src.data, imgSize * sizeof(unsigned char));

	//执行Histogram Equalization
	cout << "运行Histogram Equalization算法，并与OpenCV内置算法比较..." << endl;
	unsigned char* resultUChar = new unsigned char[imgSize];
	histEq(img, imgSize, resultUChar);

	//将C++数组转换回OpenCV Mat
	Mat result(imgHeight, imgWidth, CV_8U);
	memcpy(result.data, resultUChar, imgSize * sizeof(unsigned char));

	Mat resultRef = histogramEq(src, false);
	if (resultRef.empty()) {
		cout << "OpenCV内置算法错误！程序退出！" << endl;
		return 1;
	}

	//与OpenCV内置算法比较结果
	bool testOK = true;
	for (int i = 0; i < imgSize; ++i) {
		int diff = abs(int(resultRef.data[i]) - int(resultUChar[i]));
		if (diff != 0) {
			testOK = false;
			break;
		}
	}
	if (testOK) cout << "PASSED! 输出图像与OpenCV一致!\n" << endl;
	else cout << "FAILED! 输出图像与OpenCV不一致!\n" << endl;

	//以下部分为性能测试
	int loopCount = 20;
	cout << "性能测试：运行算法" << loopCount << "次..." << endl;

	//在CPU上运行算法并计算时间
	cout << "正在CPU上运行算法..." << endl;
	histEqBench(img, imgSize, resultUChar, loopCount);
	cout << endl;

	//在CPU上运行OpenCV内置算法并计算时间
	cout << "正在CPU上运行OpenCV内置算法..." << endl;
	histogramEqCPU(src, loopCount);
	cout << endl;

	//显示GPU OpenCL信息并运行算法
	cout << "尝试使用GPU OpenCL运行OpenCV内置算法..." << endl;
	if (ocl::haveOpenCL()) {
		cv::ocl::Context context;
		if (!context.create(cv::ocl::Device::TYPE_GPU))
		{
			cout << "Failed creating the context..." << endl;
		}

		cout << "检测到" << context.ndevices() << "个可用GPU" << endl;
		for (int i = 0; i < context.ndevices(); i++)
		{
			cv::ocl::Device device = context.device(i);
			cout << "Name:              " << device.name() << endl;
			cout << "Available:         " << device.available() << endl;
			cout << "ImageSupport:      " << device.imageSupport() << endl;
			cout << "OpenCL_C_Version:  " << device.OpenCL_C_Version() << endl;
			cout << endl;
		}

		cout << "使用OpenCL GPU: " << context.device(0).name() << "运行OpenCV内置算法..." << endl;
		histogramEqCL(image, loopCount);
		cout << endl;
	}
	else {
		cout << "GPU OpenCL不可用！测试跳过！" << endl;
	}

	//显示GPU OpenCL信息并运行算法
	cout << "尝试使用CUDA运行算法..." << endl;
	if (isCudaAvailable()) {

		//使用CUDA运行Histogram Equalization
		cout << "默认使用GPU Device 0" << endl;
		cout << "使用CUDA运行Histogram Equalization算法，并与OpenCV内置算法比较..." << endl;
		unsigned char* resultUCharCuda = new unsigned char[imgSize];
		histEqCuda(img, imgSize, resultUCharCuda);

		//将C++数组转换回OpenCV Mat
		Mat resultCuda(imgHeight, imgWidth, CV_8U);
		memcpy(resultCuda.data, resultUCharCuda, imgSize * sizeof(unsigned char));

		//与OpenCV内置算法比较结果
		bool testOKCuda = true;
		for (int i = 0; i < imgSize; ++i) {
			int diff = abs(int(resultRef.data[i]) - int(resultUCharCuda[i]));
			if (diff != 0) {
				testOKCuda = false;
				break;
			}
		}
		if (testOKCuda) cout << "PASSED! 输出图像与OpenCV一致!\n" << endl;
		else cout << "FAILED! 输出图像与OpenCV不一致!\n" << endl;

		//namedWindow("CUDA处理后图片", WINDOW_NORMAL);
		//imshow("CUDA处理后图片", resultCuda);
		//waitKey(0);
		//destroyWindow("CUDA处理后图片");

		//使用CUDA运行算法并计算时间
		cout << "CUDA性能测试：运行算法" << loopCount << "次..." << endl;
		cout << "正在使用CUDA运行算法..." << endl;
		histEqCudaBench(img, imgSize, resultUCharCuda, loopCount);
		cout << endl;
	}
	else {

		cout << "GPU CUDA不可用！测试跳过！" << endl;
	}

	system("pause");
	return 0;
}
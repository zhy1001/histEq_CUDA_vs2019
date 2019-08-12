
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

using namespace cv;
using namespace std;

//����ڰ�ͼ��
Mat readImgGray(string file) {

	Mat img = imread(file, IMREAD_GRAYSCALE);
	return img;
}

//ʹ��CPU����Histogram Equalization
//���ش�����ͼ��
//��׼
Mat histogramEq(Mat img, bool display) {

	//ֱ��ͼ���⻯
	Mat result;
	equalizeHist(img, result);

	//����������ʾͼƬ
	if (display) {
		namedWindow("ԭͼƬ", WINDOW_NORMAL);
		imshow("ԭͼƬ", img);
		namedWindow("�����ͼƬ", WINDOW_NORMAL);
		imshow("�����ͼƬ", result);
		waitKey(0);
		destroyWindow("ԭͼƬ");
		destroyWindow("�����ͼƬ");
	}

	return result;
}

//Benchmark: ʹ��CPU
int histogramEqCPU(Mat img, int loopCount) {

	//CPUԤ��
	Mat warmup;
	equalizeHist(img, warmup);

	//ֱ��ͼ���⻯
	Mat result;
	auto start = chrono::steady_clock::now();
	for (int i = 0; i < loopCount; i++) {
		equalizeHist(img, result);
	}
	auto end = chrono::steady_clock::now();

	//����ʱ��
	auto elapsed = chrono::duration_cast<std::chrono::microseconds>(end - start);
	cout << "CPU����ʱ�䣺" << elapsed.count() / (loopCount * 1000.0) << "ms" << endl;

	return 0;
}

//Benchmark: ʹ��OpenCL
int histogramEqCL(string file, int loopCount) {

	//����ڰ�ͼ��
	UMat img;
	imread(file, IMREAD_GRAYSCALE).copyTo(img);
	if (img.empty()) {
		cout << "�޷�����ͼƬ��" << endl;
		return 1;
	}

	//GPUԤ��
	UMat warmup;
	equalizeHist(img, warmup);

	//ֱ��ͼ���⻯
	UMat result;
	auto start = chrono::steady_clock::now();
	for (int i = 0; i < loopCount; i++) {
		equalizeHist(img, result);
	}
	auto end = chrono::steady_clock::now();

	//����ʱ��
	auto elapsed = chrono::duration_cast<chrono::microseconds>(end - start);
	cout << "GPU OpenCL����ʱ�䣺" << elapsed.count() / (loopCount * 1000.0) << "ms" << endl;

	return 0;
}

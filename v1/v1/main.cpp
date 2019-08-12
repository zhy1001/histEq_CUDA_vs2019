
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
		cout << "���ļ�������ͼ��: " << argv[1] << "..." << endl;
	}

	string image = argv[1];

	Mat src = readImgGray(image);
	if (src.empty()) {
		cout << "ͼƬ��ȡ���󣡳����˳���" << endl;
		return 1;
	}

	//��OpenCV Matת��ΪC++����
	int imgWidth = src.cols;
	int imgHeight = src.rows;
	int imgSize = imgWidth * imgHeight;
	unsigned char* img = new unsigned char[imgSize];
	memcpy(img, src.data, imgSize * sizeof(unsigned char));

	//ִ��Histogram Equalization
	cout << "����Histogram Equalization�㷨������OpenCV�����㷨�Ƚ�..." << endl;
	unsigned char* resultUChar = new unsigned char[imgSize];
	histEq(img, imgSize, resultUChar);

	//��C++����ת����OpenCV Mat
	Mat result(imgHeight, imgWidth, CV_8U);
	memcpy(result.data, resultUChar, imgSize * sizeof(unsigned char));

	Mat resultRef = histogramEq(src, false);
	if (resultRef.empty()) {
		cout << "OpenCV�����㷨���󣡳����˳���" << endl;
		return 1;
	}

	//��OpenCV�����㷨�ȽϽ��
	bool testOK = true;
	for (int i = 0; i < imgSize; ++i) {
		int diff = abs(int(resultRef.data[i]) - int(resultUChar[i]));
		if (diff != 0) {
			testOK = false;
			break;
		}
	}
	if (testOK) cout << "PASSED! ���ͼ����OpenCVһ��!\n" << endl;
	else cout << "FAILED! ���ͼ����OpenCV��һ��!\n" << endl;

	//���²���Ϊ���ܲ���
	int loopCount = 20;
	cout << "���ܲ��ԣ������㷨" << loopCount << "��..." << endl;

	//��CPU�������㷨������ʱ��
	cout << "����CPU�������㷨..." << endl;
	histEqBench(img, imgSize, resultUChar, loopCount);
	cout << endl;

	//��CPU������OpenCV�����㷨������ʱ��
	cout << "����CPU������OpenCV�����㷨..." << endl;
	histogramEqCPU(src, loopCount);
	cout << endl;

	//��ʾGPU OpenCL��Ϣ�������㷨
	cout << "����ʹ��GPU OpenCL����OpenCV�����㷨..." << endl;
	if (ocl::haveOpenCL()) {
		cv::ocl::Context context;
		if (!context.create(cv::ocl::Device::TYPE_GPU))
		{
			cout << "Failed creating the context..." << endl;
		}

		cout << "��⵽" << context.ndevices() << "������GPU" << endl;
		for (int i = 0; i < context.ndevices(); i++)
		{
			cv::ocl::Device device = context.device(i);
			cout << "Name:              " << device.name() << endl;
			cout << "Available:         " << device.available() << endl;
			cout << "ImageSupport:      " << device.imageSupport() << endl;
			cout << "OpenCL_C_Version:  " << device.OpenCL_C_Version() << endl;
			cout << endl;
		}

		cout << "ʹ��OpenCL GPU: " << context.device(0).name() << "����OpenCV�����㷨..." << endl;
		histogramEqCL(image, loopCount);
		cout << endl;
	}
	else {
		cout << "GPU OpenCL�����ã�����������" << endl;
	}

	//��ʾGPU OpenCL��Ϣ�������㷨
	cout << "����ʹ��CUDA�����㷨..." << endl;
	if (isCudaAvailable()) {

		//ʹ��CUDA����Histogram Equalization
		cout << "Ĭ��ʹ��GPU Device 0" << endl;
		cout << "ʹ��CUDA����Histogram Equalization�㷨������OpenCV�����㷨�Ƚ�..." << endl;
		unsigned char* resultUCharCuda = new unsigned char[imgSize];
		histEqCuda(img, imgSize, resultUCharCuda);

		//��C++����ת����OpenCV Mat
		Mat resultCuda(imgHeight, imgWidth, CV_8U);
		memcpy(resultCuda.data, resultUCharCuda, imgSize * sizeof(unsigned char));

		//��OpenCV�����㷨�ȽϽ��
		bool testOKCuda = true;
		for (int i = 0; i < imgSize; ++i) {
			int diff = abs(int(resultRef.data[i]) - int(resultUCharCuda[i]));
			if (diff != 0) {
				testOKCuda = false;
				break;
			}
		}
		if (testOKCuda) cout << "PASSED! ���ͼ����OpenCVһ��!\n" << endl;
		else cout << "FAILED! ���ͼ����OpenCV��һ��!\n" << endl;

		//namedWindow("CUDA�����ͼƬ", WINDOW_NORMAL);
		//imshow("CUDA�����ͼƬ", resultCuda);
		//waitKey(0);
		//destroyWindow("CUDA�����ͼƬ");

		//ʹ��CUDA�����㷨������ʱ��
		cout << "CUDA���ܲ��ԣ������㷨" << loopCount << "��..." << endl;
		cout << "����ʹ��CUDA�����㷨..." << endl;
		histEqCudaBench(img, imgSize, resultUCharCuda, loopCount);
		cout << endl;
	}
	else {

		cout << "GPU CUDA�����ã�����������" << endl;
	}

	system("pause");
	return 0;
}
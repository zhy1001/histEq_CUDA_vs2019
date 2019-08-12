# histEq_CUDA_vs2019
Visual Studio 2019's solution for histogram equalization using CUDA

Input: Any image supported by OpenCV 4's imread function  
Output: Histogram equalized 8-bit grayscale image  

Optimized for NVIDIA GPU with Maxwell and later architecture. Atomic operations in shared memory are used for computing local histogram.

CUDA Toolkit and OpenCV 4 are required. And change the "opencv library" if needed.

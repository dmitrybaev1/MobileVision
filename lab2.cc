#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <arm_neon.h>
#include <chrono>

using namespace std;
using namespace cv;

void invert_rgb(const uint8_t* rgb, uint8_t* inverted_rgb, int num_pixels)
{
	cout << "inside function invert_rgb" << endl;
	auto t1 = chrono::high_resolution_clock::now();
	for(int i=0; i<num_pixels; ++i) {
		inverted_rgb[i] = ~rgb[i];
	}
	auto t2 = chrono::high_resolution_clock::now();
	auto duration = chrono::duration_cast<chrono::microseconds>(t2-t1).count();
	cout << duration << " us" << endl;
}

void invert_rgb_neon(const uint8_t* rgb, uint8_t* inverted_rgb, int num_pixels) {
	cout << "inside function invert_rgb_neon" << endl;
	// We'll use 128-bit NEON registers to process 8 pixels in parallel.
	num_pixels /= 16;
	auto t1_neon = chrono::high_resolution_clock::now();
	for(int i=0; i<num_pixels; ++i, rgb+=16, inverted_rgb+=16) {
		uint8x16_t vector = vld1q_u8(rgb);
		uint8x16_t inverted_vector = vmvnq_u8(vector);
		vst1q_u8(inverted_rgb, inverted_vector);
	}
	auto t2_neon = chrono::high_resolution_clock::now();
	auto duration_neon = chrono::duration_cast<chrono::microseconds>(t2_neon-t1_neon).count();
	cout << duration_neon << " us" << endl;
}

int main(int argc,char** argv)
{
	uint8_t * rgb_arr;
	uint8_t * inverted_arr_neon;

	if (argc != 2) {
		cout << "Usage: opencv_neon image_name" << endl;
		return -1;
	}

	Mat rgb_image;
	rgb_image = imread(argv[1], IMREAD_COLOR);
	if (!rgb_image.data) {
		cout << "Could not open the image" << endl;
		return -1;
	}
	if (rgb_image.isContinuous()) {
		rgb_arr = rgb_image.data;
	}
	else {
		cout << "data is not continuous" << endl;
		return -2;
	}

	int width = rgb_image.cols;
	int height = rgb_image.rows;
	int num_pixels = width*height;
	Mat inverted_image_neon(height, width, CV_8UC3, Scalar(0));
	inverted_arr_neon = inverted_image_neon.data;
	auto t1_neon = chrono::high_resolution_clock::now();
	invert_rgb_neon(rgb_arr, inverted_arr_neon, num_pixels);
	auto t2_neon = chrono::high_resolution_clock::now();
	auto duration_neon = chrono::duration_cast<chrono::microseconds>(t2_neon-t1_neon).count();
	cout << "invert_rgb_neon" << endl;
	cout << duration_neon << " us" << endl;

	imwrite("inverted_neon.png", inverted_image_neon);

    return 0;
}
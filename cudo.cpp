#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <algorithm>

#define MAX_THREADS_PER_BLOCK 1024

using namespace std;

void arrayPrint(int* arr, int length) {
	int i;
	for (i = 0; i < length; ++i) {
		cout << arr[i] << " ";
	}
	cout << endl;
}

void arrayFill(int* arr, int length) {
	srand(time(NULL));
	int i;
	for (i = 0; i < length; ++i) {
		arr[i] = rand();
	}
}

void supplementArrayWithMaxValue(int* arr, int sourseLength, int destLength) {
	int maxValue = *max_element(arr, arr + sourseLength);

	for (int i = sourseLength; i < destLength; i++) {
		arr[i] = maxValue + 1;
	}
}

bool comparisonArrays(int* arr1, int* arr2, int length) {
	for (int i = 0; i < length; i++) {
		if (arr1[i] != arr2[i]) {
			return false;
		}
	}
	return true;
}

int* getCopyArray(int* sourse, int length) {
	int* dest = new int[length];

	for (int i = 0; i < length; i++) {
		dest[i] = sourse[i];
	}
	return dest;
}

int powerCeil(int x) {
	if (x <= 1) return 1;
	int power = 2;
	x--;
	// x >> 1 эквивалентно x / 2
	// power <<= 1 эквивалентно power * 2
	while (x >>= 1) power <<= 1;
	return power;
}

__global__ void bitonicSortStep(int* deviceArr, int j, int k) {
	unsigned int i, ixj;
	i = threadIdx.x + blockDim.x * blockIdx.x;
	ixj = i ^ j;

	if ((ixj) > i) {
		if ((i & k) == 0) {
			/* Sort ascending */
			if (deviceArr[i] > deviceArr[ixj]) {
				/* exchange(i,ixj); */
				int temp = deviceArr[i];
				deviceArr[i] = deviceArr[ixj];
				deviceArr[ixj] = temp;
			}
		}
		if ((i & k) != 0) {
			/* Sort descending */
			if (deviceArr[i] < deviceArr[ixj]) {
				/* exchange(i,ixj); */
				int temp = deviceArr[i];
				deviceArr[i] = deviceArr[ixj];
				deviceArr[ixj] = temp;
			}
		}
	}
}

void bitonicSort(int* arr, int length) {
	int* deviceArr;
	size_t size = length * sizeof(int);

	cudaMalloc((void**)&deviceArr, size);
	cudaMemcpy(deviceArr, arr, size, cudaMemcpyHostToDevice);

	int threads = MAX_THREADS_PER_BLOCK > length ? length : MAX_THREADS_PER_BLOCK;
	int blocks = length / threads;

	for (int k = 2; k <= length; k = k << 1) {
		for (int j = k >> 1; j > 0; j = j >> 1) {
			bitonicSortStep <<<blocks, threads>>> (deviceArr, j, k);
		}
	}
	cudaMemcpy(arr, deviceArr, size, cudaMemcpyDeviceToHost);
	cudaFree(deviceArr);
}

bool isBitonic(int*v, int length) {
	bool wasDecreasing = v[length - 1] > v[0];
	int numInflections = 0;
	for (int i = 0; i < length && numInflections <= 2; i++) {
		bool isDecreasing = v[i] > v[(i + 1) % length];
		// Check if this element and next one are an inflection.
		if (wasDecreasing != isDecreasing) {
			numInflections++;
			wasDecreasing = isDecreasing;
		}
	}

	return 2 == numInflections;
}

int main(void) {
	int length = 0;
	cout << "Enter length of the array: ";
	cin >> length;

	int roundingLength = powerCeil(length);
	int* cudaArr = new int[roundingLength];
	arrayFill(cudaArr, length);
	supplementArrayWithMaxValue(cudaArr, length, roundingLength);


	cout << "start CPU std sort..." << endl;
	clock_t start, stop;
	start = clock();
	int* stdArr = getCopyArray(cudaArr, length);
	sort(stdArr, stdArr + length);
	stop = clock();
	double elapcedTime = (double)(stop - start) / CLOCKS_PER_SEC;
	cout << "time on cpu: " << elapcedTime << " sec" << endl << endl;


	cout << "start GPU bitonic sort..." << endl;
	start = clock();
	bitonicSort(cudaArr, roundingLength);
	stop = clock();
	elapcedTime = (double)(stop - start) / CLOCKS_PER_SEC;
	cout << "time on gpu: " << elapcedTime << " sec" << endl << endl;

	cout << "is_bitonic: " << isBitonic(cudaArr, length) << endl;;
	cout << "is equals: " << comparisonArrays(cudaArr, stdArr, length) << endl;
}

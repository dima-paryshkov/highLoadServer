#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <algorithm>

#define MAX_THREADS_PER_BLOCK 1024

using namespace std;

void arrayPrint(int *arr, int length)
{
	int i;
	for (i = 0; i < length; ++i)
	{
		cout << arr[i] << " ";
	}
	cout << endl;
}

void arrayFill(int *arr, int length)
{
	srand(time(NULL));
	int i;
	for (i = 0; i < length; ++i)
	{
		arr[i] = rand();
	}
}

// Дополняем длнинну массива до степени двойки
void supplementArrayWithMaxValue(int *arr, int sourseLength, int destLength)
{
	int maxValue = *max_element(arr, arr + sourseLength);

	for (int i = sourseLength; i < destLength; i++)
	{
		arr[i] = maxValue + 1;
	}
}

// Сравнение массивов
bool comparisonArrays(int *arr1, int *arr2, int length)
{
	for (int i = 0; i < length; i++)
	{
		if (arr1[i] != arr2[i])
		{
			return false;
		}
	}
	return true;
}

int *getCopyArray(int *sourse, int length)
{
	int *dest = new int[length];

	for (int i = 0; i < length; i++)
	{
		dest[i] = sourse[i];
	}
	return dest;
}

// Округление длинны массива до большей степени двойки
int powerCeil(int x)
{
	if (x <= 1)
		return 1;
	int power = 2;
	x--;
	// x >> 1 эквивалентно x / 2
	// power <<= 1 эквивалентно power * 2
	// оператор >> есть мещение битов
	// https://ru.wikipedia.org/wiki/%D0%9E%D0%BF%D0%B5%D1%80%D0%B0%D1%82%D0%BE%D1%80%D1%8B_%D0%B2_C_%D0%B8_C%2B%2B
	while (x >>= 1)
		power <<= 1;
	return power;
}

// __global__ — выполняется на GPU, вызывается с CPU.
__global__ void bitonicSortStep(int *deviceArr, int j, int k)
{
	unsigned int i, ixj;

	// https://forums.developer.nvidia.com/t/difference-between-threadidx-blockidx-statements/12161
	// Опрделяем элемент с которым работаем
	i = threadIdx.x + blockDim.x * blockIdx.x;

	// Побитовое исключающее ИЛИ (xor)
	ixj = i ^ j;

	// Сам алгоритм сортировки
	if ((ixj) > i)
	{
		if ((i & k) == 0)
		{
			/* Sort ascending */
			if (deviceArr[i] > deviceArr[ixj])
			{
				/* exchange(i,ixj); */
				int temp = deviceArr[i];
				deviceArr[i] = deviceArr[ixj];
				deviceArr[ixj] = temp;
			}
		}
		if ((i & k) != 0)
		{
			/* Sort descending */
			if (deviceArr[i] < deviceArr[ixj])
			{
				/* exchange(i,ixj); */
				int temp = deviceArr[i];
				deviceArr[i] = deviceArr[ixj];
				deviceArr[ixj] = temp;
			}
		}
	}
}

void bitonicSort(int *arr, int length)
{
	// Указатель по массив в памяти видеокарты
	int *deviceArr;
	// Длинна массива в байтах в памяти видеокарты
	size_t size = length * sizeof(int);

	// Выделяем память в пямяти видеокарты
	cudaMalloc((void **)&deviceArr, size);
	// Копируем наш массив в память видеокарты
	cudaMemcpy(deviceArr, arr, size, cudaMemcpyHostToDevice);

	// https://en.wikipedia.org/wiki/Thread_block_(CUDA_programming)#:~:text=The%20number%20of%20threads%20in,contain%20up%20to%201024%20threads.
	
	// Если длина массива меньше максимального размера блока, 
	// то размер блока устанавливаем как размер массива, иначе оставляем максимальный размер
	int threads = MAX_THREADS_PER_BLOCK > length ? length : MAX_THREADS_PER_BLOCK;

	// Определяем число блоков
	int blocks = length / threads;


	// https://habr.com/ru/post/54707/
	for (int k = 2; k <= length; k = k << 1)
	{
		for (int j = k >> 1; j > 0; j = j >> 1)
		{
			// функцию ядра, которая осуществляет вычисления
			bitonicSortStep<<<blocks, threads>>>(deviceArr, j, k);
		}
	}

	cudaMemcpy(arr, deviceArr, size, cudaMemcpyDeviceToHost);
	cudaFree(deviceArr);
}

// Проверка последовательности на то, что она явялется битонной последовательностью
bool isBitonic(int *v, int length)
{
	bool wasDecreasing = v[length - 1] > v[0];
	int numInflections = 0;
	for (int i = 0; i < length && numInflections <= 2; i++)
	{
		bool isDecreasing = v[i] > v[(i + 1) % length];
		// Check if this element and next one are an inflection.
		if (wasDecreasing != isDecreasing)
		{
			numInflections++;
			wasDecreasing = isDecreasing;
		}
	}

	return 2 == numInflections;
}

int main(void)
{
	int length = 0;
	cout << "Enter length of the array: ";
	cin >> length;

	// Округляем длину до степени двойки
	int roundingLength = powerCeil(length);

	// Выделяем память в оперативной памяти компьютера
	int *cudaArr = new int[roundingLength];

	// Заполняем массив случайными значениями
	arrayFill(cudaArr, length);

	// Дополняет массив до длины двойки максимальным значением исходного массива
	supplementArrayWithMaxValue(cudaArr, length, roundingLength);

	cout << "start CPU std sort..." << endl;
	clock_t start, stop;
	start = clock();
	int *stdArr = getCopyArray(cudaArr, length);
	sort(stdArr, stdArr + length);
	stop = clock();
	double elapcedTime = (double)(stop - start) / CLOCKS_PER_SEC;
	cout << "time on cpu: " << elapcedTime << " sec" << endl << endl;

	cout << "start GPU bitonic sort..." << endl;
	start = clock();
	// Битонная сортировка
	bitonicSort(cudaArr, roundingLength);
	stop = clock();
	elapcedTime = (double)(stop - start) / CLOCKS_PER_SEC;
	cout << "time on gpu: " << elapcedTime << " sec" << endl
		 << endl;

	cout << "is_bitonic: " << isBitonic(cudaArr, length) << endl;
	;
	cout << "is equals: " << comparisonArrays(cudaArr, stdArr, length) << endl;
}

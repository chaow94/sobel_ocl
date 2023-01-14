#include <stdio.h>
#include <stdlib.h>
// windows
#include <CL/cl.h>

#include <string.h>
#include <iostream>
#include <fstream>
#include <chrono>

cl_platform_id platform_id = NULL;
cl_uint ret_num_platforms;
cl_device_id device_id = NULL;
cl_uint ret_num_devices;
cl_context context = NULL;
cl_command_queue command_queue = NULL;
cl_mem data_in = NULL;
cl_mem data_out = NULL;
cl_mem filter_in = NULL;
cl_program program = NULL;
cl_kernel kernel = NULL;
size_t kernel_code_size;
int *result;
cl_int ret;
FILE *fp;
cl_uint work_dim;
size_t global_item_size[2];
size_t local_item_size[2];

int const W = 1024;			 
int const H = 1024;			 
int const K = 3;			// filter kernel size
int const Wn = (W + K - 1); // padded image width
int const Hn = (H + K - 1); // padded image height

int point_num = Wn * Hn;
int data_vecs[Wn * Hn];

int filter_coe[K * K] = {-1, 0, 1,
						 -2, 0, 2,
						 -1, 0, 1}; // sobel filter: horizontal gradient
int i, j;

void naive_impl(int *data_in, int *data_out, int *filter_in, int rows, int cols, int K)
{
	for (int ki = 0; ki < rows; ki++)
	{
		for (int kj = 0; kj < cols; kj++)
		{
			int sum = 0;
			for (int y = 0; y < K; y++)
			{
				for (int x = 0; x < K; x++)
				{
					sum = sum + filter_in[y * K + x] * data_in[Wn * (y + ki) + x + kj];
				}
			}
			data_out[ki * cols + kj] = sum;
		}
	}
}

bool check(int *data1, int *data2, int len)
{

	for (i = 0; i < len; i++)
	{
		if (data1[i] != data2[i])
			return false;
	}
	return true;
}

int main(void)
{

	for (i = 0; i < point_num; i++)
	{
		data_vecs[i] = rand() % 20;
	}

	/*
	naive implement sobel operater
	*/
	int *naive_data_out = (int *)malloc(W * H * sizeof(int));

	auto t1 = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 100; i++)
	{
		naive_impl(data_vecs, naive_data_out, filter_coe, H, W, K);
	}
	auto t2 = std::chrono::high_resolution_clock::now();

	auto int_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
	std::cout << "naive_impl cost: " << int_ms.count() << " ms\n";

	/*
	opencl implemnet
	*/

	result = (int *)malloc(W * H * sizeof(int));

	ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
	ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id,
						 &ret_num_devices);
	context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
	command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

	std::ifstream kernel_file("../../sobel.cl");
	std::string cl_str(std::istreambuf_iterator<char>(kernel_file), (std::istreambuf_iterator<char>()));
	char *cl_str_c = (char *)(cl_str.c_str());

	program = clCreateProgramWithSource(context, 1, (const char **)(&cl_str_c),
										(const size_t *)&kernel_code_size, &ret);
	ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
	kernel = clCreateKernel(program, "sobel_vloadn", &ret);
	data_in = clCreateBuffer(context, CL_MEM_READ_WRITE, Wn * Hn * sizeof(int), NULL, &ret);
	data_out = clCreateBuffer(context, CL_MEM_READ_WRITE, W * H * sizeof(int), NULL, &ret);
	filter_in = clCreateBuffer(context, CL_MEM_READ_WRITE, K * K * sizeof(int), NULL, &ret);

	// write image data into data_in buffer
	ret = clEnqueueWriteBuffer(command_queue, data_in, CL_TRUE, 0, Wn * Hn * sizeof(int), data_vecs, 0, NULL, NULL);

	// write filter data into filter_in buffer
	ret = clEnqueueWriteBuffer(command_queue, filter_in, CL_TRUE, 0, K * K * sizeof(int), filter_coe, 0, NULL, NULL);

	// set kernel arguments
	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&data_in);
	ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&filter_in);
	ret = clSetKernelArg(kernel, 2, sizeof(int), (void *)&K);
	ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&data_out);

	work_dim = 2;
	global_item_size[0] = {W};
	global_item_size[1] = {H};
	local_item_size[0] = {16};
	local_item_size[1] = {16};


	auto t11 = std::chrono::high_resolution_clock::now();

	for (int i = 0; i < 100; i++)
	{
		// execute data parallel kernel */
		ret = clEnqueueNDRangeKernel(command_queue, kernel, work_dim, NULL,
									 global_item_size, local_item_size, 0, NULL, NULL);
		// read data_out to host
		ret = clEnqueueReadBuffer(command_queue, data_out, CL_TRUE, 0,
								  W * H * sizeof(int), result, 0, NULL, NULL);
	}

	auto t22 = std::chrono::high_resolution_clock::now();

	// integral duration: requires duration_cast
	auto int_ms1 = std::chrono::duration_cast<std::chrono::milliseconds>(t22 - t11);
	std::cout << "ocl implement cost: " << int_ms1.count() << " ms\n";

	if (check(naive_data_out, result, H * W))
	{
		printf("naive and ocl is equal!\n");
	}
	else
	{
		printf("naive and ocl is not equal!\n");
	}


	ret = clReleaseKernel(kernel);
	ret = clReleaseProgram(program);
	ret = clReleaseMemObject(data_in);
	ret = clReleaseMemObject(data_out);
	ret = clReleaseMemObject(filter_in);
	ret = clReleaseCommandQueue(command_queue);
	ret = clReleaseContext(context);
	free(result);
	free(naive_data_out);

	system("pause");
	return 0;
}

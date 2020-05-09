#include<stdio.h>
#include"CudaInterface.h"

#define PIXELS_RES_PER_THREAD 32
#define THREADS_RES_PER_BLOCK 32

__device__
int computePoint(const int &x, const int &y, const double &dx, const double &dy, const double &zoom, const int &res, const int &max_itr)
{
	const double cx = ((4. * double(x) / double(res)) - 2. + dx) * zoom;
	const double cy = ((4. * double(y) / double(res)) - 2. + dy) * zoom;

	double zx = 0, nzx;
	double zy = 0, nzy;

	int i;
	for(i = 0; i < max_itr; i++)
	{
		if(zx*zx + zy*zy >= 4.) goto break_loop;
		nzx = zx*zx - zy*zy + cx;
		nzy = 2.*zx*zy + cy;
		zx = nzx;
		zy = nzy;
	}
break_loop: {}
	return i;
}

__global__
void computeSet(int *scalar_field, int res, int max_itr, double dx, double dy, double zoom)
{
	const int xPos = PIXELS_RES_PER_THREAD * (blockIdx.x*gridDim.x + threadIdx.x);
	const int yPos = PIXELS_RES_PER_THREAD * (blockIdx.y*gridDim.y + threadIdx.y);
	if(xPos >= res || yPos >= res) return;

	int ptr = yPos*res + xPos;
	
	for(int y = 0; y < res; ++y)
	{
		if(yPos + y >= res) goto b;
		for(int x = 0; x < res; ++x, ++ptr)
		{
			if(xPos + x >= res) goto a;
			*(scalar_field+ptr) = computePoint(x, y, dx, dy, zoom, res, max_itr);
		}
a: {}
   		ptr += xPos;
	}
b: {}
}

int* Mandelbrot::compute() const
{
	// Compute number of blocks and threads per block
	const int block_res = ceil(double(m_res) / double(PIXELS_RES_PER_THREAD) / double(THREADS_RES_PER_BLOCK));
	dim3 blocks(block_res, block_res, 1);
	dim3 threads(PIXELS_RES_PER_THREAD, PIXELS_RES_PER_THREAD, 1);

	printf("-----\n%d\n-----\n", block_res);

	// Allocate memory (host & device)
	int *scalar_field = (int*) malloc(m_res*m_res*sizeof(int));
	int *device_scalar_field; 
	cudaMalloc(&device_scalar_field, m_res*m_res*sizeof(int));
	cudaMemcpy(scalar_field, device_scalar_field, m_res*m_res*sizeof(int), cudaMemcpyHostToDevice);

	// Launch kernel
	computeSet<<<blocks,threads>>>(device_scalar_field, m_res, m_max_itr, m_dx, m_dy, m_zoom);
	cudaDeviceSynchronize();

	printf("-----\n");
	
	// Clear memory
	cudaMemcpy(scalar_field, device_scalar_field, m_res*m_res*sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(device_scalar_field);

	return scalar_field;
}

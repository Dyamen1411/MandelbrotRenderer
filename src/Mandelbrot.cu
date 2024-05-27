#include<stdio.h>
#include<vector>
#include"CudaInterface.h"

#define THREADS_RES_PER_BLOCK 16
#define PIXELS_RES_PER_THREAD 16
#define PIXELS_RES_PER_BLOCK THREADS_RES_PER_BLOCK*PIXELS_RES_PER_THREAD 


__device__
int computePoint(const int &x, const int &y, const double &dx, const double &dy, const double &zoom, const int &res, const int &max_itr)
{
	const double cx = (((4. * double(x) / double(res)) - 2.) / zoom) + dx;
	const double cy = (((4. * double(y) / double(res)) - 2.) / zoom) + dy;

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
	const int xPos = PIXELS_RES_PER_BLOCK*blockIdx.x + PIXELS_RES_PER_THREAD*threadIdx.x;
	const int yPos = PIXELS_RES_PER_BLOCK*blockIdx.y + PIXELS_RES_PER_THREAD*threadIdx.y;

	if(xPos >= res || yPos >= res) return;
	
	int ptr = yPos*res + xPos;
	//printf("BlockId: %3d %3d, ThreadId: %3d %3d, Pos: %6d %6d, Ptr: %10d\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, xPos, yPos, ptr);

	for(int y = yPos; y < yPos+PIXELS_RES_PER_THREAD; ++y)
	{
		if(y >= res) goto b;
		for(int x = xPos; x < xPos+PIXELS_RES_PER_THREAD; ++x, ++ptr)
		{
			if(x >= res) { ptr += xPos; goto a; }
			*(scalar_field+ptr) = computePoint(x, y, dx, dy, zoom, res, max_itr);
		}
   		ptr += res - PIXELS_RES_PER_THREAD;
a: {}
	}
b: {}
}

int compareStr(const char *a, const char *b, int l)
{
	for(int i = 0; i < l; ++i) if(*(a+i) == '\0' || *(b+i) == '\0' || *(a+i) != *(b+i)) return 0;
	return 1;
}

void mandelbrotSettings(std::vector<char*>::iterator &it, const std::vector<char*>::iterator &end, double &dx, double &dy, int &res, int &max_itr, double &zoom)
{
	printf("Reading settings...\n");
	++it;
	for(; it != end; ++it)
	{
		if(**it=='[' || **it=='\0') goto mdlse;

			if(compareStr(*it, "dx", 2)) { dx = atof(*it+3); printf("\tdx = %f\n", dx); }
		else	if(compareStr(*it, "dy", 2)) { dy = atof(*it+3); printf("\tdy = %f\n", dy); }
		else	if(compareStr(*it, "res", 3)) { res = atoi(*it+4); printf("\tres = %d\n", res); }
		else	if(compareStr(*it, "max_itr", 7)) { max_itr = atoi(*it+8); printf("\tmax_itr = %d\n", max_itr); }
		else	if(compareStr(*it, "zoom", 4)) { zoom = atof(*it+5); printf("\tzoom = %f\n", zoom); }
	}
mdlse: {}
}

void mandelbrotRendering(std::vector<char*>::iterator &it, const std::vector<char*>::iterator &end, uint8_t &mode)
{
	printf("rendering\n");
	++it;
	
	for(; it != end; ++it)
	{
		if(**it=='[' || **it=='\0') goto mdlrese;

		if(compareStr(*it, "method", 6))
		{
				if(compareStr(*it+7, "DEFAULT", 7)) { mode = 0; printf("\tmode = %d\n", mode); }
			else	if(compareStr(*it+7, "POTENTIAL", 9)) { mode = 1; printf("\tmode = %d\n", mode); }
		}
	}
mdlrese: {}
}

void mandelbrotRenderingSettings(std::vector<char*>::iterator &it, const std::vector<char*>::iterator &end)
{
	printf("rendering settings\n");
}

void Mandelbrot::init()
{
	printf("Initializing...\n");
	printf("Reading config file\n");

	FILE *fp;
	char *current_buffer, c;
	int ptr = 0;
	std::vector<char*> lines;
	
	// Reading file
	fp = fopen("mdl.cfg", "r");
	if(fp == NULL)
	{
		printf("Could not open config file! Using default parameters...");
		return;
	}

	current_buffer = (char*) malloc(255 * sizeof(char));
	
	// storing line by line
	while((c = fgetc(fp)) != EOF)
	{
		if(c == ';')
		{
			*(current_buffer+ptr) = '\0';
			ptr = 0;
			lines.push_back(current_buffer);
			current_buffer = (char*) malloc(255*sizeof(char));
		}else if(c != '\n')
		{
			*(current_buffer+ptr) = c;
			ptr++;
		}
	}

	fclose(fp);
	
	for(auto it = lines.begin(); it != lines.end(); ++it)
	{
		if(compareStr(*it, "[mandelbrot_settings]", 20)) mandelbrotSettings(it, lines.end(), m_dx, m_dy, m_res, m_max_itr, m_zoom);
		else if(compareStr(*it, "[mandelbrot_rendering]", 22)) mandelbrotRendering(it, lines.end(), m_draw_mode);
		else if(compareStr(*it, "[mandelbrot_rendering_settings]", 30)) mandelbrotRenderingSettings(it, lines.end());
	}

	for(auto it = lines.begin(); it != lines.end(); ++it) free(*it);
	
	printf("-----\n");
}

int* Mandelbrot::compute() const
{
	// Compute number of blocks and threads per block
	const int block_res = ceil(double(m_res) / double(PIXELS_RES_PER_THREAD * THREADS_RES_PER_BLOCK));
	dim3 blocks(block_res, block_res, 1);
	dim3 threads(PIXELS_RES_PER_THREAD, PIXELS_RES_PER_THREAD, 1);

	printf("Running on %d blocks (%d threads).\n", (block_res*block_res), (block_res*block_res*THREADS_RES_PER_BLOCK*THREADS_RES_PER_BLOCK));

	// Allocate memory (host & device)
	int *device_scalar_field, *scalar_field = (int*) malloc(m_res*m_res*sizeof(int));
	cudaMalloc(&device_scalar_field, m_res*m_res*sizeof(int));
	cudaMemcpy(scalar_field, device_scalar_field, m_res*m_res*sizeof(int), cudaMemcpyHostToDevice);

	// Launch kernel
	computeSet<<<blocks,threads>>>(device_scalar_field, m_res, m_max_itr, m_dx, m_dy, m_zoom);
	cudaDeviceSynchronize();

	// Clear memory
	cudaMemcpy(scalar_field, device_scalar_field, m_res*m_res*sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(device_scalar_field);
	
	printf("-----");

	return scalar_field;
}

#include<chrono>
#include<cstdlib>
#include<stdio.h>

#include"CudaInterface.h"

#define RES 100

using namespace std::chrono;

void write(int *scalar_field, const Mandelbrot &m)
{
	FILE *fptr;
	fptr = fopen("out/out.dat", "w");
	
	fwrite(&m.m_draw_mode, sizeof(uint8_t), 1, fptr);
	fwrite(&m.m_max_itr, sizeof(int), 1, fptr);
	fwrite(&m.m_res, sizeof(int), 1, fptr);
	fwrite(scalar_field, sizeof(int), m.m_res*m.m_res, fptr);

	fclose(fptr);
}

int main(int argc, char **argv)
{
	Mandelbrot m;
	m.init();
	
	uint64_t start = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
	int *scalar_field = m.compute();
	uint64_t end = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
	printf("---> Render took %10d ms\n", (end - start));

	write(scalar_field, m);	

	free(scalar_field);

	return 0;
}

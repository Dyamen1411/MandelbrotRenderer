#include<chrono>
#include<cstdlib>
#include<stdio.h>

#include"CudaInterface.h"

#define RES 100

using namespace std::chrono;

void write(int *scalar_field)
{
	FILE *fptr;
	fptr = fopen("out/out.dat", "w");
	
	int res = RES;
	fwrite(&res, sizeof(int), 1, fptr);
	fwrite(scalar_field, sizeof(int), res*res, fptr);

	fclose(fptr);
}

int main(int argc, char **argv)
{
	Mandelbrot m(RES, 100, 0, 0, 1);
	
	uint64_t start = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
	int *scalar_field = m.compute();
	uint64_t end = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
	printf("Render took %10d ms\n", (end - start));

	write(scalar_field);

	free(scalar_field);

	return 0;
}

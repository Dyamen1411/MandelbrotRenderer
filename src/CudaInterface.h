#ifndef _CUDA_INTERFACE_H_
#define _CUDA_INTERFACE_H_

#include <cstdint>

class Mandelbrot
{
public:
	Mandelbrot() :m_draw_mode(0) {}
	~Mandelbrot() {};

public:
	void init();
	int* compute() const;

public:
	// Draw modes: 
	// - 0: DEFAULT
	// - 1: POTENTIAL 
	uint8_t m_draw_mode;
	int m_res;
	int m_max_itr;
	double m_dx, m_dy, m_zoom;
};

#endif /* _CUDA_INTERFACE_H_ */


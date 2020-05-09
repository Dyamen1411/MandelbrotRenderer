#ifndef _CUDA_INTERFACE_H_
#define _CUDA_INTERFACE_H_

class Mandelbrot
{
public:
	Mandelbrot(const int &res, const int &max_itr, const double &dx = 0, const double &dy = 0, const double &zoom = 1) : m_res(res), m_max_itr(max_itr), m_dx(dx), m_dy(dy), m_zoom(zoom) {}
	~Mandelbrot() {};

public:
	int* compute() const;

private:
	int m_res, m_max_itr;
	double m_dx, m_dy, m_zoom;
};

#endif /* _CUDA_INTERFACE_H_ */


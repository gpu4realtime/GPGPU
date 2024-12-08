#pragma once

#include <vector>

namespace mat
{
	template<typename dataT>
	std::vector<dataT> Inversion(const float* d_x, int n);
}

namespace mat_test
{
	void Inversion();
}
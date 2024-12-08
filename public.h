#pragma once

#include <vector>

namespace mat
{
	template<typename dataT>
	std::vector<dataT> Inversion(const float* in, int n);
}

namespace mat_test
{
	void Inversion();
}

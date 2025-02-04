#include "public.h"

#include "helper.h"
#include "helper_cuda.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <cassert>

#define TILE_DIM   32
#define MAX_N_REG  32
#define NUM_OF_WARPS_IN_BLOCK 8

namespace mat_gpu
{
	template<typename dataT>
	struct Data
	{
		dataT x_00_0 = 0;
		dataT x_00_1 = 0;
		dataT x_00_2 = 0;
		dataT x_00_3 = 0;

		dataT x_01_0 = 0;
		dataT x_01_1 = 0;
		dataT x_01_2 = 0;
		dataT x_01_3 = 0;

		dataT x_10_0 = 0;
		dataT x_10_1 = 0;
		dataT x_10_2 = 0;
		dataT x_10_3 = 0;

		dataT x_11_0 = 0;
		dataT x_11_1 = 0;
		dataT x_11_2 = 0;
		dataT x_11_3 = 0;
	};

	template<typename dataT>
	__device__ void compute_echelon_and_row_reduced_echelon_form_generic( Data<dataT>& data
									    , dataT shared_00[TILE_DIM]
									    , dataT shared_01[TILE_DIM]
									    , int i
									    , int j)
	{
		#pragma unroll TILE_DIM
		for (int k = 0; k < TILE_DIM; ++k)
		{
			dataT x_00_p = shared_00[j];
			dataT x_01_p = shared_01[j];

			dataT num = __shfl_sync(0xFFFFFFFF, data.x_00_0, k, TILE_DIM);

			data.x_00_0 = (4 * i <= k || j <= k) ? data.x_00_0 : __fadd_rz(data.x_00_0, -__fmul_rz(num, x_00_p));
			data.x_01_0 = 4 * i <= k ? data.x_01_0 : __fadd_rz(data.x_01_0, -__fmul_rz(num, x_01_p));

			num = __shfl_sync(0xFFFFFFFF, data.x_00_1, k, TILE_DIM);

			data.x_00_1 = (4 * i + 1 <= k || j <= k) ? data.x_00_1 : __fadd_rz(data.x_00_1, -__fmul_rz(num, x_00_p));
			data.x_01_1 = 4 * i + 1 <= k ? data.x_01_1 : __fadd_rz(data.x_01_1, -__fmul_rz(num, x_01_p));

			num = __shfl_sync(0xFFFFFFFF, data.x_00_2, k, TILE_DIM);

			data.x_00_2 = (4 * i + 2 <= k || j <= k) ? data.x_00_2 : __fadd_rz(data.x_00_2, -__fmul_rz(num, x_00_p));
			data.x_01_2 = 4 * i + 2 <= k ? data.x_01_2 : __fadd_rz(data.x_01_2, -__fmul_rz(num, x_01_p));

			num = __shfl_sync(0xFFFFFFFF, data.x_00_3, k, TILE_DIM);

			data.x_00_3 = (4 * i + 3 <= k || j <= k) ? data.x_00_3 : __fadd_rz(data.x_00_3, -__fmul_rz(num, x_00_p));
			data.x_01_3 = 4 * i + 3 <= k ? data.x_01_3 : __fadd_rz(data.x_01_3, -__fmul_rz(num, x_01_p));

			num = __shfl_sync(0xFFFFFFFF, data.x_10_0, k, TILE_DIM);

			data.x_10_0 = j <= k ? data.x_10_0 : __fadd_rz(data.x_10_0, -__fmul_rz(num, x_00_p));
			data.x_11_0 = __fadd_rz(data.x_11_0, -__fmul_rz(num, x_01_p));

			num = __shfl_sync(0xFFFFFFFF, data.x_10_1, k, TILE_DIM);

			data.x_10_1 = j <= k ? data.x_10_1 : __fadd_rz(data.x_10_1, -__fmul_rz(num, x_00_p));
			data.x_11_1 = __fadd_rz(data.x_11_1, -__fmul_rz(num, x_01_p));

			num = __shfl_sync(0xFFFFFFFF, data.x_10_2, k, TILE_DIM);

			data.x_10_2 = j <= k ? data.x_10_2 : __fadd_rz(data.x_10_2, -__fmul_rz(num, x_00_p));
			data.x_11_2 = __fadd_rz(data.x_11_2, -__fmul_rz(num, x_01_p));

			num = __shfl_sync(0xFFFFFFFF, data.x_10_3, k, TILE_DIM);

			data.x_10_3 = j <= k ? data.x_10_3 : __fadd_rz(data.x_10_3, -__fmul_rz(num, x_00_p));
			data.x_11_3 = __fadd_rz(data.x_11_3, -__fmul_rz(num, x_01_p));

			__syncthreads();

			if (k + 1 == 4 * i)
			{
				dataT val = __shfl_sync(0xFFFFFFFF, data.x_00_0, k + 1, TILE_DIM);

				data.x_00_0 = __fdiv_rz(data.x_00_0, val);
				data.x_01_0 = __fdiv_rz(data.x_01_0, val);

				shared_00[j] = data.x_00_0;
				shared_01[j] = data.x_01_0;
			}
			else if (k + 1 == 4 * i + 1)
			{
				dataT val = __shfl_sync(0xFFFFFFFF, data.x_00_1, k + 1, TILE_DIM);

				data.x_00_1 = __fdiv_rz(data.x_00_1, val);
				data.x_01_1 = __fdiv_rz(data.x_01_1, val);

				shared_00[j] = data.x_00_1;
				shared_01[j] = data.x_01_1;
			}
			else if (k + 1 == 4 * i + 2)
			{
				dataT val = __shfl_sync(0xFFFFFFFF, data.x_00_2, k + 1, TILE_DIM);

				data.x_00_2 = __fdiv_rz(data.x_00_2, val);
				data.x_01_2 = __fdiv_rz(data.x_01_2, val);

				shared_00[j] = data.x_00_2;
				shared_01[j] = data.x_01_2;
			}
			else if (k + 1 == 4 * i + 3)
			{
				dataT val = __shfl_sync(0xFFFFFFFF, data.x_00_3, k + 1, TILE_DIM);

				data.x_00_3 = __fdiv_rz(data.x_00_3, val);
				data.x_01_3 = __fdiv_rz(data.x_01_3, val);

				shared_00[j] = data.x_00_3;
				shared_01[j] = data.x_01_3;
			}

			__syncthreads();
		}
	}

	template<typename dataT>
	__device__ void compute_echelon_and_row_reduced_echelon_form_row( Data<dataT>& data
									, dataT shared_00[TILE_DIM]
									, dataT shared_01[TILE_DIM]
									, int i
									, int j)
	{
		#pragma unroll TILE_DIM
		for (int k = 0; k < TILE_DIM; ++k)
		{
			dataT x_00_p = shared_00[j];
			dataT x_01_p = shared_01[j];

			dataT num = __shfl_sync(0xFFFFFFFF, data.x_00_0, k, TILE_DIM);

			data.x_00_0 = (4 * i == k || j <= k) ? data.x_00_0 : __fadd_rz(data.x_00_0, -__fmul_rz(num, x_00_p));
			data.x_01_0 = (4 * i == k) ? data.x_01_0 : __fadd_rz(data.x_01_0, -__fmul_rz(num, x_01_p));

			num = __shfl_sync(0xFFFFFFFF, data.x_00_1, k, TILE_DIM);

			data.x_00_1 = (4 * i + 1 == k || j <= k) ? data.x_00_1 : __fadd_rz(data.x_00_1, -__fmul_rz(num, x_00_p));
			data.x_01_1 = (4 * i + 1 == k) ? data.x_01_1 : __fadd_rz(data.x_01_1, -__fmul_rz(num, x_01_p));

			num = __shfl_sync(0xFFFFFFFF, data.x_00_2, k, TILE_DIM);

			data.x_00_2 = (4 * i + 2 == k || j <= k) ? data.x_00_2 : __fadd_rz(data.x_00_2, -__fmul_rz(num, x_00_p));
			data.x_01_2 = (4 * i + 2 == k) ? data.x_01_2 : __fadd_rz(data.x_01_2, -__fmul_rz(num, x_01_p));

			num = __shfl_sync(0xFFFFFFFF, data.x_00_3, k, TILE_DIM);

			data.x_00_3 = (4 * i + 3 == k || j <= k) ? data.x_00_3 : __fadd_rz(data.x_00_3, -__fmul_rz(num, x_00_p));
			data.x_01_3 = (4 * i + 3 == k) ? data.x_01_3 : __fadd_rz(data.x_01_3, -__fmul_rz(num, x_01_p));

			__syncthreads();

			if (k + 1 == 4 * i)
			{
				dataT val = __shfl_sync(0xFFFFFFFF, data.x_00_0, k + 1, TILE_DIM);

				data.x_00_0 = __fdiv_rz(data.x_00_0, val);
				data.x_01_0 = __fdiv_rz(data.x_01_0, val);

				shared_00[j] = data.x_00_0;
				shared_01[j] = data.x_01_0;
			}
			else if (k + 1 == 4 * i + 1)
			{
				dataT val = __shfl_sync(0xFFFFFFFF, data.x_00_1, k + 1, TILE_DIM);

				data.x_00_1 = __fdiv_rz(data.x_00_1, val);
				data.x_01_1 = __fdiv_rz(data.x_01_1, val);

				shared_00[j] = data.x_00_1;
				shared_01[j] = data.x_01_1;
			}
			else if (k + 1 == 4 * i + 2)
			{
				dataT val = __shfl_sync(0xFFFFFFFF, data.x_00_2, k + 1, TILE_DIM);

				data.x_00_2 = __fdiv_rz(data.x_00_2, val);
				data.x_01_2 = __fdiv_rz(data.x_01_2, val);

				shared_00[j] = data.x_00_2;
				shared_01[j] = data.x_01_2;
			}
			else if (k + 1 == 4 * i + 3)
			{
				dataT val = __shfl_sync(0xFFFFFFFF, data.x_00_3, k + 1, TILE_DIM);

				data.x_00_3 = __fdiv_rz(data.x_00_3, val);
				data.x_01_3 = __fdiv_rz(data.x_01_3, val);

				shared_00[j] = data.x_00_3;
				shared_01[j] = data.x_01_3;
			}

			__syncthreads();
		}
	}

	template<typename dataT>
	__device__ void compute_echelon_and_row_reduced_echelon_form_pivot( Data<dataT>& data
									  , dataT shared_00[TILE_DIM]
									  , dataT shared_01[TILE_DIM]
									  , int i
									  , int j)
	{
		#pragma unroll TILE_DIM
		for (int k = 0; k < TILE_DIM; ++k)
		{
			data.x_00_0 = (4 * i == k) ? data.x_00_0 : __fadd_rz(data.x_00_0, -__fmul_rz(shared_00[j], __shfl_sync(0xFFFFFFFF, data.x_00_0, k, TILE_DIM)));

			data.x_00_1 = (4 * i + 1 == k) ? data.x_00_1 : __fadd_rz(data.x_00_1, -__fmul_rz(shared_00[j], __shfl_sync(0xFFFFFFFF, data.x_00_1, k, TILE_DIM)));

			data.x_00_2 = (4 * i + 2 == k) ? data.x_00_2 : __fadd_rz(data.x_00_2, -__fmul_rz(shared_00[j], __shfl_sync(0xFFFFFFFF, data.x_00_2, k, TILE_DIM)));

			data.x_00_3 = (4 * i + 3 == k) ? data.x_00_3 : __fadd_rz(data.x_00_3, -__fmul_rz(shared_00[j], __shfl_sync(0xFFFFFFFF, data.x_00_3, k, TILE_DIM)));

			__syncthreads();

			if (k + 1 == 4 * i)
			{
				dataT val = __shfl_sync(0xFFFFFFFF, data.x_00_0, k + 1, TILE_DIM);

				data.x_00_0 = __fdiv_rz(data.x_00_0, val);

				shared_00[j] = data.x_00_0;
			}
			else if (k + 1 == 4 * i + 1)
			{
				dataT val = __shfl_sync(0xFFFFFFFF, data.x_00_1, k + 1, TILE_DIM);

				data.x_00_1 = __fdiv_rz(data.x_00_1, val);

				shared_00[j] = data.x_00_1;
			}
			else if (k + 1 == 4 * i + 2)
			{
				dataT val = __shfl_sync(0xFFFFFFFF, data.x_00_2, k + 1, TILE_DIM);

				data.x_00_2 = __fdiv_rz(data.x_00_2, val);

				shared_00[j] = data.x_00_2;
			}
			else if (k + 1 == 4 * i + 3)
			{
				dataT val = __shfl_sync(0xFFFFFFFF, data.x_00_3, k + 1, TILE_DIM);

				data.x_00_3 = __fdiv_rz(data.x_00_3, val);

				shared_00[j] = data.x_00_3;
			}

			__syncthreads();
		}
	}

	template<typename dataT>
	__global__ void __maxnreg__(MAX_N_REG) init_and_compute_echelon( dataT* in
								       , dataT* augmat
								       , dataT* buffer
								       , int n
								       , int nblocks
								       , int iter)
	{
		__shared__ dataT shared_00[TILE_DIM];
		__shared__ dataT shared_01[TILE_DIM];

		int block_col = blockIdx.x / nblocks + 1;
		int block_row = blockIdx.x - nblocks * (blockIdx.x / nblocks);

		Data<dataT> data;

		data.x_00_0 = in[(TILE_DIM * iter + 4 * threadIdx.y) * n + (TILE_DIM * iter + threadIdx.x)];
		data.x_00_1 = in[(TILE_DIM * iter + 4 * threadIdx.y + 1) * n + (TILE_DIM * iter + threadIdx.x)];
		data.x_00_2 = in[(TILE_DIM * iter + 4 * threadIdx.y + 2) * n + (TILE_DIM * iter + threadIdx.x)];
		data.x_00_3 = in[(TILE_DIM * iter + 4 * threadIdx.y + 3) * n + (TILE_DIM * iter + threadIdx.x)];

		if (block_col < nblocks)
		{
			data.x_01_0 = in[(TILE_DIM * iter + 4 * threadIdx.y) * n + (TILE_DIM * block_col + threadIdx.x)];
			data.x_01_1 = in[(TILE_DIM * iter + 4 * threadIdx.y + 1) * n + (TILE_DIM * block_col + threadIdx.x)];
			data.x_01_2 = in[(TILE_DIM * iter + 4 * threadIdx.y + 2) * n + (TILE_DIM * block_col + threadIdx.x)];
			data.x_01_3 = in[(TILE_DIM * iter + 4 * threadIdx.y + 3) * n + (TILE_DIM * block_col + threadIdx.x)];

			data.x_11_0 = in[(TILE_DIM * block_row + 4 * threadIdx.y) * n + (TILE_DIM * block_col + threadIdx.x)];
			data.x_11_1 = in[(TILE_DIM * block_row + 4 * threadIdx.y + 1) * n + (TILE_DIM * block_col + threadIdx.x)];
			data.x_11_2 = in[(TILE_DIM * block_row + 4 * threadIdx.y + 2) * n + (TILE_DIM * block_col + threadIdx.x)];
			data.x_11_3 = in[(TILE_DIM * block_row + 4 * threadIdx.y + 3) * n + (TILE_DIM * block_col + threadIdx.x)];
		}
		else
		{
			data.x_01_0 = 4 * threadIdx.y != threadIdx.x ? 0 : 1;
			data.x_01_1 = 4 * threadIdx.y + 1 != threadIdx.x ? 0 : 1;
			data.x_01_2 = 4 * threadIdx.y + 2 != threadIdx.x ? 0 : 1;
			data.x_01_3 = 4 * threadIdx.y + 3 != threadIdx.x ? 0 : 1;
		}

		data.x_10_0 = in[(TILE_DIM * block_row + 4 * threadIdx.y) * n + (TILE_DIM * iter + threadIdx.x)];
		data.x_10_1 = in[(TILE_DIM * block_row + 4 * threadIdx.y + 1) * n + (TILE_DIM * iter + threadIdx.x)];
		data.x_10_2 = in[(TILE_DIM * block_row + 4 * threadIdx.y + 2) * n + (TILE_DIM * iter + threadIdx.x)];
		data.x_10_3 = in[(TILE_DIM * block_row + 4 * threadIdx.y + 3) * n + (TILE_DIM * iter + threadIdx.x)];

		if (threadIdx.y == 0)
		{
			dataT val = __shfl_sync(0xFFFFFFFF, data.x_00_0, 0, TILE_DIM);

			data.x_00_0 = __fdiv_rz(data.x_00_0, val);
			data.x_01_0 = __fdiv_rz(data.x_01_0, val);

			shared_00[threadIdx.x] = data.x_00_0;
			shared_01[threadIdx.x] = data.x_01_0;
		}

		__syncthreads();

		if (blockIdx.x == nblocks * nblocks)
		{
			compute_echelon_and_row_reduced_echelon_form_pivot(data, shared_00, shared_01, threadIdx.y, threadIdx.x);

			augmat[(4 * threadIdx.y + TILE_DIM * iter) * 2 * n + (TILE_DIM * iter + threadIdx.x)] = data.x_00_0;
			augmat[(4 * threadIdx.y + 1 + TILE_DIM * iter) * 2 * n + (TILE_DIM * iter + threadIdx.x)] = data.x_00_1;
			augmat[(4 * threadIdx.y + 2 + TILE_DIM * iter) * 2 * n + (TILE_DIM * iter + threadIdx.x)] = data.x_00_2;
			augmat[(4 * threadIdx.y + 3 + TILE_DIM * iter) * 2 * n + (TILE_DIM * iter + threadIdx.x)] = data.x_00_3;

			return;
		}

		if (block_row == iter)
		{
			compute_echelon_and_row_reduced_echelon_form_row(data, shared_00, shared_01, threadIdx.y, threadIdx.x);

			if (block_col == 1)
			{
				block_col = 3;

				buffer[(4 * threadIdx.y + TILE_DIM * block_col) * n + (TILE_DIM * block_row + threadIdx.x)] = data.x_01_0;
				buffer[(4 * threadIdx.y + 1 + TILE_DIM * block_col) * n + (TILE_DIM * block_row + threadIdx.x)] = data.x_01_1;
				buffer[(4 * threadIdx.y + 2 + TILE_DIM * block_col) * n + (TILE_DIM * block_row + threadIdx.x)] = data.x_01_2;
				buffer[(4 * threadIdx.y + 3 + TILE_DIM * block_col) * n + (TILE_DIM * block_row + threadIdx.x)] = data.x_01_3;

				return;
			}

			augmat[(4 * threadIdx.y + TILE_DIM * block_row) * 2 * n + (TILE_DIM * block_col + threadIdx.x)] = data.x_01_0;
			augmat[(4 * threadIdx.y + 1 + TILE_DIM * block_row) * 2 * n + (TILE_DIM * block_col + threadIdx.x)] = data.x_01_1;
			augmat[(4 * threadIdx.y + 2 + TILE_DIM * block_row) * 2 * n + (TILE_DIM * block_col + threadIdx.x)] = data.x_01_2;
			augmat[(4 * threadIdx.y + 3 + TILE_DIM * block_row) * 2 * n + (TILE_DIM * block_col + threadIdx.x)] = data.x_01_3;

			return;
		}

		compute_echelon_and_row_reduced_echelon_form_generic(data, shared_00, shared_01, threadIdx.y, threadIdx.x);

		if (block_row == iter + 1)
		{
			block_row = 1;

			buffer[(4 * threadIdx.y + TILE_DIM * block_row) * n + (TILE_DIM * (block_col - 1) + threadIdx.x)] = data.x_11_0;
			buffer[(4 * threadIdx.y + 1 + TILE_DIM * block_row) * n + (TILE_DIM * (block_col - 1) + threadIdx.x)] = data.x_11_1;
			buffer[(4 * threadIdx.y + 2 + TILE_DIM * block_row) * n + (TILE_DIM * (block_col - 1) + threadIdx.x)] = data.x_11_2;
			buffer[(4 * threadIdx.y + 3 + TILE_DIM * block_row) * n + (TILE_DIM * (block_col - 1) + threadIdx.x)] = data.x_11_3;

			return;
		}

		if (block_col == iter + 1)
		{
			block_col = 3;

			buffer[(4 * threadIdx.y + TILE_DIM * block_col) * n + (TILE_DIM * block_row + threadIdx.x)] = data.x_11_0;
			buffer[(4 * threadIdx.y + 1 + TILE_DIM * block_col) * n + (TILE_DIM * block_row + threadIdx.x)] = data.x_11_1;
			buffer[(4 * threadIdx.y + 2 + TILE_DIM * block_col) * n + (TILE_DIM * block_row + threadIdx.x)] = data.x_11_2;
			buffer[(4 * threadIdx.y + 3 + TILE_DIM * block_col) * n + (TILE_DIM * block_row + threadIdx.x)] = data.x_11_3;

			return;
		}

		augmat[(4 * threadIdx.y + TILE_DIM * block_row) * 2 * n + (TILE_DIM * block_col + threadIdx.x)] = data.x_11_0;
		augmat[(4 * threadIdx.y + 1 + TILE_DIM * block_row) * 2 * n + (TILE_DIM * block_col + threadIdx.x)] = data.x_11_1;
		augmat[(4 * threadIdx.y + 2 + TILE_DIM * block_row) * 2 * n + (TILE_DIM * block_col + threadIdx.x)] = data.x_11_2;
		augmat[(4 * threadIdx.y + 3 + TILE_DIM * block_row) * 2 * n + (TILE_DIM * block_col + threadIdx.x)] = data.x_11_3;
	}

	template<typename dataT>
	__global__ void __maxnreg__(MAX_N_REG) compute_echelon( dataT* inv
							      , dataT* augmat
							      , dataT* buffer
							      , int n
							      , int nblocks
							      , int iter)
	{
		__shared__ dataT shared_00[TILE_DIM];
		__shared__ dataT shared_01[TILE_DIM];

		int block_col = blockIdx.x / nblocks + iter + 1;
		int block_row = blockIdx.x - nblocks * (blockIdx.x / nblocks);

		Data<dataT> data;

		data.x_00_0 = buffer[(TILE_DIM * (iter & 1) + 4 * threadIdx.y) * n + threadIdx.x];
		data.x_00_1 = buffer[(TILE_DIM * (iter & 1) + 4 * threadIdx.y + 1) * n + threadIdx.x];
		data.x_00_2 = buffer[(TILE_DIM * (iter & 1) + 4 * threadIdx.y + 2) * n + threadIdx.x];
		data.x_00_3 = buffer[(TILE_DIM * (iter & 1) + 4 * threadIdx.y + 3) * n + threadIdx.x];

		if (block_col < nblocks + iter)
		{
			data.x_01_0 = buffer[(TILE_DIM * (iter & 1) + 4 * threadIdx.y) * n + (TILE_DIM * (block_col - iter) + threadIdx.x)];
			data.x_01_1 = buffer[(TILE_DIM * (iter & 1) + 4 * threadIdx.y + 1) * n + (TILE_DIM * (block_col - iter) + threadIdx.x)];
			data.x_01_2 = buffer[(TILE_DIM * (iter & 1) + 4 * threadIdx.y + 2) * n + (TILE_DIM * (block_col - iter) + threadIdx.x)];
			data.x_01_3 = buffer[(TILE_DIM * (iter & 1) + 4 * threadIdx.y + 3) * n + (TILE_DIM * (block_col - iter) + threadIdx.x)];

			data.x_11_0 = augmat[(TILE_DIM * block_row + 4 * threadIdx.y) * 2 * n + (TILE_DIM * block_col + threadIdx.x)];
			data.x_11_1 = augmat[(TILE_DIM * block_row + 4 * threadIdx.y + 1) * 2 * n + (TILE_DIM * block_col + threadIdx.x)];
			data.x_11_2 = augmat[(TILE_DIM * block_row + 4 * threadIdx.y + 2) * 2 * n + (TILE_DIM * block_col + threadIdx.x)];
			data.x_11_3 = augmat[(TILE_DIM * block_row + 4 * threadIdx.y + 3) * 2 * n + (TILE_DIM * block_col + threadIdx.x)];
		}
		else
		{
			data.x_01_0 = 4 * threadIdx.y != threadIdx.x ? 0 : 1;
			data.x_01_1 = 4 * threadIdx.y + 1 != threadIdx.x ? 0 : 1;
			data.x_01_2 = 4 * threadIdx.y + 2 != threadIdx.x ? 0 : 1;
			data.x_01_3 = 4 * threadIdx.y + 3 != threadIdx.x ? 0 : 1;
		}

		if (block_col != iter || block_row != iter)
		{
			data.x_10_0 = buffer[(TILE_DIM * ((iter & 1) + 2) + 4 * threadIdx.y) * n + (TILE_DIM * block_row + threadIdx.x)];
			data.x_10_1 = buffer[(TILE_DIM * ((iter & 1) + 2) + 4 * threadIdx.y + 1) * n + (TILE_DIM * block_row + threadIdx.x)];
			data.x_10_2 = buffer[(TILE_DIM * ((iter & 1) + 2) + 4 * threadIdx.y + 2) * n + (TILE_DIM * block_row + threadIdx.x)];
			data.x_10_3 = buffer[(TILE_DIM * ((iter & 1) + 2) + 4 * threadIdx.y + 3) * n + (TILE_DIM * block_row + threadIdx.x)];
		}
		else
		{
			data.x_10_0 = data.x_00_0;
			data.x_10_1 = data.x_00_1;
			data.x_10_2 = data.x_00_2;
			data.x_10_3 = data.x_00_3;
		}

		if (threadIdx.y == 0)
		{
			dataT val = __shfl_sync(0xFFFFFFFF, data.x_00_0, 0, TILE_DIM);

			data.x_00_0 = __fdiv_rz(data.x_00_0, val);
			data.x_01_0 = __fdiv_rz(data.x_01_0, val);

			shared_00[threadIdx.x] = data.x_00_0;
			shared_01[threadIdx.x] = data.x_01_0;
		}

		__syncthreads();

		if (blockIdx.x == nblocks * nblocks)
		{
			compute_echelon_and_row_reduced_echelon_form_pivot(data, shared_00, shared_01, threadIdx.y, threadIdx.x);

			if (iter < nblocks - 1)
			{
				augmat[(4 * threadIdx.y + TILE_DIM * iter) * 2 * n + (TILE_DIM * iter + threadIdx.x)] = data.x_00_0;
				augmat[(4 * threadIdx.y + 1 + TILE_DIM * iter) * 2 * n + (TILE_DIM * iter + threadIdx.x)] = data.x_00_1;
				augmat[(4 * threadIdx.y + 2 + TILE_DIM * iter) * 2 * n + (TILE_DIM * iter + threadIdx.x)] = data.x_00_2;
				augmat[(4 * threadIdx.y + 3 + TILE_DIM * iter) * 2 * n + (TILE_DIM * iter + threadIdx.x)] = data.x_00_3;

				return;
			}
			
			block_col -= (iter + 1);

			inv[(4 * threadIdx.y + TILE_DIM * iter) * n + (TILE_DIM * iter + threadIdx.x)] = data.x_00_0;
			inv[(4 * threadIdx.y + 1 + TILE_DIM * iter) * n + (TILE_DIM * iter + threadIdx.x)] = data.x_00_1;
			inv[(4 * threadIdx.y + 2 + TILE_DIM * iter) * n + (TILE_DIM * iter + threadIdx.x)] = data.x_00_2;
			inv[(4 * threadIdx.y + 3 + TILE_DIM * iter) * n + (TILE_DIM * iter + threadIdx.x)] = data.x_00_3;

			return;
		}

		if (block_row == iter)
		{
			compute_echelon_and_row_reduced_echelon_form_row(data, shared_00, shared_01, threadIdx.y, threadIdx.x);

			if (iter < (nblocks - 1) && block_col == iter + 1)
			{
				block_col = ((iter - 1) & 1) + 2;

				buffer[(4 * threadIdx.y + TILE_DIM * block_col) * n + (TILE_DIM * block_row + threadIdx.x)] = data.x_01_0;
				buffer[(4 * threadIdx.y + 1 + TILE_DIM * block_col) * n + (TILE_DIM * block_row + threadIdx.x)] = data.x_01_1;
				buffer[(4 * threadIdx.y + 2 + TILE_DIM * block_col) * n + (TILE_DIM * block_row + threadIdx.x)] = data.x_01_2;
				buffer[(4 * threadIdx.y + 3 + TILE_DIM * block_col) * n + (TILE_DIM * block_row + threadIdx.x)] = data.x_01_3;

				return;
			}

			if (iter < nblocks - 1)
			{
				augmat[(4 * threadIdx.y + TILE_DIM * block_row) * 2 * n + (TILE_DIM * block_col + threadIdx.x)] = data.x_01_0;
				augmat[(4 * threadIdx.y + 1 + TILE_DIM * block_row) * 2 * n + (TILE_DIM * block_col + threadIdx.x)] = data.x_01_1;
				augmat[(4 * threadIdx.y + 2 + TILE_DIM * block_row) * 2 * n + (TILE_DIM * block_col + threadIdx.x)] = data.x_01_2;
				augmat[(4 * threadIdx.y + 3 + TILE_DIM * block_row) * 2 * n + (TILE_DIM * block_col + threadIdx.x)] = data.x_01_3;

				return;
			}

			block_col -= (iter + 1);

			inv[(4 * threadIdx.y + TILE_DIM * block_row) * n + (TILE_DIM * block_col + threadIdx.x)] = data.x_01_0;
			inv[(4 * threadIdx.y + 1 + TILE_DIM * block_row) * n + (TILE_DIM * block_col + threadIdx.x)] = data.x_01_1;
			inv[(4 * threadIdx.y + 2 + TILE_DIM * block_row) * n + (TILE_DIM * block_col + threadIdx.x)] = data.x_01_2;
			inv[(4 * threadIdx.y + 3 + TILE_DIM * block_row) * n + (TILE_DIM * block_col + threadIdx.x)] = data.x_01_3;

			return;
		}

		compute_echelon_and_row_reduced_echelon_form_generic(data, shared_00, shared_01, threadIdx.y, threadIdx.x);

		if (iter < (nblocks - 1) && block_row == iter + 1)
		{
			block_row = ((iter - 1) & 1);

			buffer[(4 * threadIdx.y + TILE_DIM * block_row) * n + (TILE_DIM * (block_col - 1 - iter) + threadIdx.x)] = data.x_11_0;
			buffer[(4 * threadIdx.y + 1 + TILE_DIM * block_row) * n + (TILE_DIM * (block_col - 1 - iter) + threadIdx.x)] = data.x_11_1;
			buffer[(4 * threadIdx.y + 2 + TILE_DIM * block_row) * n + (TILE_DIM * (block_col - 1 - iter) + threadIdx.x)] = data.x_11_2;
			buffer[(4 * threadIdx.y + 3 + TILE_DIM * block_row) * n + (TILE_DIM * (block_col - 1 - iter) + threadIdx.x)] = data.x_11_3;

			return;
		}

		if (iter < (nblocks - 1) && block_col == iter + 1)
		{
			block_col = ((iter - 1) & 1) + 2;

			buffer[(4 * threadIdx.y + TILE_DIM * block_col) * n + (TILE_DIM * block_row + threadIdx.x)] = data.x_11_0;
			buffer[(4 * threadIdx.y + 1 + TILE_DIM * block_col) * n + (TILE_DIM * block_row + threadIdx.x)] = data.x_11_1;
			buffer[(4 * threadIdx.y + 2 + TILE_DIM * block_col) * n + (TILE_DIM * block_row + threadIdx.x)] = data.x_11_2;
			buffer[(4 * threadIdx.y + 3 + TILE_DIM * block_col) * n + (TILE_DIM * block_row + threadIdx.x)] = data.x_11_3;

			return;
		}

		if (iter < nblocks - 1)
		{
			augmat[(4 * threadIdx.y + TILE_DIM * block_row) * 2 * n + (TILE_DIM * block_col + threadIdx.x)] = data.x_11_0;
			augmat[(4 * threadIdx.y + 1 + TILE_DIM * block_row) * 2 * n + (TILE_DIM * block_col + threadIdx.x)] = data.x_11_1;
			augmat[(4 * threadIdx.y + 2 + TILE_DIM * block_row) * 2 * n + (TILE_DIM * block_col + threadIdx.x)] = data.x_11_2;
			augmat[(4 * threadIdx.y + 3 + TILE_DIM * block_row) * 2 * n + (TILE_DIM * block_col + threadIdx.x)] = data.x_11_3;

			return;
		}

		block_col -= (iter + 1);

		inv[(4 * threadIdx.y + TILE_DIM * block_row) * n + (TILE_DIM * block_col + threadIdx.x)] = data.x_11_0;
		inv[(4 * threadIdx.y + 1 + TILE_DIM * block_row) * n + (TILE_DIM * block_col + threadIdx.x)] = data.x_11_1;
		inv[(4 * threadIdx.y + 2 + TILE_DIM * block_row) * n + (TILE_DIM * block_col + threadIdx.x)] = data.x_11_2;
		inv[(4 * threadIdx.y + 3 + TILE_DIM * block_row) * n + (TILE_DIM * block_col + threadIdx.x)] = data.x_11_3;
	}
}

namespace mat
{
	/*
	* Compute inversion of square matrix 
	* 
	* This is GPU implementation of matrix inversion algorithm posted on www.mathworks.com
	* and can be found by link https://www.mathworks.com/matlabcentral/answers/243916-trying-to-write-a-program-that-calculates-the-inverse-of-a-3x3-matrix-my-program-works-for-some-mat#answer_324850
	* 
	* The original algorithm was modified so that the processing was performed in one pass.
	*/
	template<typename dataT>
	std::vector<dataT> Inversion(const float* in, int n)
	{
		dataT* d_data = nullptr;

		checkCudaErrors(cudaMalloc((void**)&d_data, n * n * sizeof(dataT)));

		checkCudaErrors(cudaMemcpy(d_data, in, n * n * sizeof(float), cudaMemcpyHostToDevice));

		dataT* d_augmat = nullptr;

		checkCudaErrors(cudaMalloc((void**)&d_augmat, n * 2 * n * sizeof(dataT)));

		dataT* d_buffer = nullptr;

		checkCudaErrors(cudaMalloc((void**)&d_buffer, 4 * TILE_DIM * n * sizeof(dataT)));

		dim3 block_size(TILE_DIM, NUM_OF_WARPS_IN_BLOCK);

		cudaFuncSetCacheConfig(mat_gpu::init_and_compute_echelon<dataT>, cudaFuncCachePreferL1);

		cudaFuncSetCacheConfig(mat_gpu::compute_echelon<dataT>, cudaFuncCachePreferL1);

		const auto num_of_blocks_in_column = n / TILE_DIM;

		const auto num_of_processing_blocks = num_of_blocks_in_column * num_of_blocks_in_column + 1;

		for (auto iter = 0; iter < num_of_blocks_in_column; ++iter)
		{
			if (iter != 0)
			{
				mat_gpu::compute_echelon<dataT> <<< num_of_processing_blocks, block_size >>>( d_data
													    , d_augmat
													    , d_buffer
													    , n
													    , num_of_blocks_in_column
													    , iter);
			}
			else
			{
				mat_gpu::init_and_compute_echelon<dataT> <<< num_of_processing_blocks, block_size >>>( d_data
														     , d_augmat
														     , d_buffer
														     , n
														     , num_of_blocks_in_column
														     , iter);
			}

			const auto err = cudaGetLastError();

			if (err != cudaSuccess)
			{
				std::cerr << "Failed to launch kernel (error code " << cudaGetErrorString(err) << ")!" << std::endl;

				cudaFree(d_data);

				cudaFree(d_augmat);

				cudaFree(d_buffer);

				return {};
			}

			cudaDeviceSynchronize();
		}

		std::vector<dataT> I(n * n, 0.f);

		checkCudaErrors(cudaMemcpy(I.data(), d_data, n * n * sizeof(dataT), cudaMemcpyDeviceToHost));

		cudaFree(d_data);

		cudaFree(d_augmat);

		cudaFree(d_buffer);

		return I;
	}

	template std::vector<float> Inversion<float>(const float* h_x, int n);
}

namespace mat_test
{
	void Inversion()
	{
		auto x = read_file<float>("test_data/2784_2784_matrix.csv", true);

		auto inv = mat::Inversion<float>(x.first.data(), 2784);

		assert(true == cmp<float>(inv, read_file<float>("test_data/2784_2784_inv_matrix.csv", false).first, 0.00000001f));

		x = read_file<float>("test_data/2048_2048_matrix.csv", true);
		
		inv = mat::Inversion<float>(x.first.data(), 2048);

		assert(true == cmp<float>(inv, read_file<float>("test_data/2048_2048_inv_matrix.csv", false).first, 0.000001f));

		x = read_file<float>("test_data/256_256_matrix.csv", true);
		
		inv = mat::Inversion<float>(x.first.data(), 256);
		
		assert(true == cmp<float>(inv, read_file<float>("test_data/256_256_inv_matrix.csv", false).first, 0.00000001f));

		x = read_file<float>("test_data/64_64_matrix.csv", true);
		
		inv = mat::Inversion<float>(x.first.data(), 64);

		assert(true == cmp<float>(inv, read_file<float>("test_data/64_64_inv_matrix.csv", false).first, 0.00000001f));
	}
}

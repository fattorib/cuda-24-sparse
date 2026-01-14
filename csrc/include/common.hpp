#pragma once
#include <cuda_bf16.h>

#include <algorithm>
#include <bitset>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>

namespace common {

void fill_random(__nv_bfloat16 *A, size_t numel, float scale = 1.0) {
	std::random_device rd{};
	std::mt19937 gen{rd()};
	std::normal_distribution<float> d{0.0, scale};
	for (int i = 0; i < numel; i++) {
		A[i] = __float2bfloat16(d(gen));
	}
}

struct b24 {
	__nv_bfloat16 val;
	size_t index;

	bool operator<(const b24 &other) const {
		return __bfloat162float(val) < __bfloat162float(other.val);
	}
};

// sparsifies an existing bf16 array using 2:4 sparsity keeping the 2 largest values
// in every block of 4 contiguous values
void sparsify24(__nv_bfloat16 *A, __nv_bfloat16 *Asparse, uint16_t *META, size_t m, size_t k) {
	size_t lda = k;
	size_t ldmeta = k / 16;

	size_t bound = std::min(16, int(k));

	for (size_t r = 0; r < m; r++) {
		for (size_t c = 0; c < k; c += 16) {
			uint16_t index = 0;  // 0000000000000000 // this corresponds to 8 x 2-bit indices

			for (size_t c_inner = 0; c_inner < bound; c_inner += 4) {
				size_t i = r * lda + (c + c_inner);

				size_t is = r * (lda / 2) + ((c + c_inner) / 2);

				b24 blocks[4] = {{A[i + 0], i + 0}, {A[i + 1], i + 1}, {A[i + 2], i + 2}, {A[i + 3], i + 3}};
				std::sort(blocks, blocks + 4);

				A[blocks[0].index] = __float2bfloat16(0.0);
				A[blocks[1].index] = __float2bfloat16(0.0);

				uint16_t first, second, pattern;
				if (blocks[2].index < blocks[3].index) {
					Asparse[is + 0] = (blocks[2].val);
					Asparse[is + 1] = (blocks[3].val);

					first = blocks[2].index % 4;
					second = blocks[3].index % 4;
				} else {
					Asparse[is + 0] = (blocks[3].val);
					Asparse[is + 1] = (blocks[2].val);

					first = blocks[3].index % 4;
					second = blocks[2].index % 4;
				}

				pattern = (second << 2) | first;  // ex: 11 01  -> 0000000000001101
				// little endian -> pack like [15:12][11:8][7:4][3:0]
				index = (pattern << c_inner) | index;
			}
			META[r * ldmeta + (c / 16)] = index;
		}
	}
}

// swizzles the Metadata to a format that can be loaded with ldmatrix
void swizzleMeta(const uint16_t *META, uint16_t *METASwizz,
                 int numelMeta, int R) {
	assert(R > 0);
	const int stride = 2 * R;
	const int tileSize = 4 * R;

	assert(numelMeta % tileSize == 0);

	for (int base = 0; base < numelMeta; base += tileSize) {
		for (int r = 0; r < R; ++r) {
			const int even = base + 2 * r;
			const int odd = even + 1;
			const int out = base + 4 * r;

			METASwizz[out + 0] = META[even];
			METASwizz[out + 1] = META[even + stride];
			METASwizz[out + 2] = META[odd];
			METASwizz[out + 3] = META[odd + stride];
		}
	}
}

//  following BLAS naming conventions, this is performs a row major GEMM [m,k] @
//  [k,n] -> [m,n] between bf16 inputs and an fp32 accumulator
void cpu_gemm_tt(const __nv_bfloat16 *A, const __nv_bfloat16 *B,
                 __nv_bfloat16 *C, int m, int n, int k) {
	int lda, ldb, ldc;

	lda = k;
	ldb = n;
	ldc = n;

	for (int r = 0; r < m; r++) {
#pragma omp parallel for
		for (int c = 0; c < n; c++) {
			float tmp = 0.0f;
			for (int inner = 0; inner < k; inner++) {
				tmp += __bfloat162float(A[r * lda + inner]) *
				       __bfloat162float(B[inner * ldb + c]);
			}
			C[r * ldc + c] = __float2bfloat16(tmp);
		}
	}
}

void check_error(const __nv_bfloat16 *arr, const __nv_bfloat16 *ref,
                 int numel) {
	float rel_err;
	float abs_err;

	float max_rel_error = -INFINITY;
	float max_abs_error = -INFINITY;

	float total_diff = 0.0f;
	float diff_norm = 0.0f;
	float norm = 0.0f;

	float a_elem, r_elem, max_a_elem, max_r_elem, max_abs_a_elem,
	    max_abs_r_elem;

	for (int i = 0; i < numel; i++) {
		a_elem = __bfloat162float(arr[i]);
		r_elem = __bfloat162float(ref[i]);

		if (arr[i] != arr[i]) {
			printf("arr: (%7.6f) ref: (%7.6f) \n", a_elem, r_elem);
			throw std::runtime_error(
			    "ERROR: NaN value encountered in output array");
			break;
		}

		if (ref[i] != ref[i]) {
			printf("arr: (%7.6f) ref: (%7.6f) \n", a_elem, r_elem);
			throw std::runtime_error(
			    "ERROR: NaN value encountered in reference array");
		}

		if (r_elem != 0) {
			rel_err = std::abs(a_elem - r_elem) / std::abs(r_elem);
		}

		abs_err = std::abs(a_elem - r_elem);

		total_diff += abs_err;

		max_rel_error = std::fmaxf(max_rel_error, rel_err);
		max_abs_error = std::fmaxf(max_abs_error, abs_err);

		if (max_rel_error == rel_err) {
			max_a_elem = a_elem;
			max_r_elem = r_elem;
		}

		if (max_abs_error == abs_err) {
			max_abs_a_elem = a_elem;
			max_abs_r_elem = r_elem;
		}

		diff_norm += std::pow((a_elem - r_elem), 2.0);
		norm += std::pow((r_elem), 2.0);
	}

	float linalg_rel_error = std::pow(diff_norm, 0.5) / std::pow(norm, 0.5);
	printf("Maximum relative error: (%1.8f)\n", max_rel_error);
	printf("Maximum absolute error: (%1.8f)\n", max_abs_error);
	printf("Linalg relative error: (%1.8f)\n", linalg_rel_error);
	printf("Average abs error: (%1.8f)\n", total_diff / float(numel));
	printf("Offending max relative error (Actual, Expected): (%1.8f, %1.8f)\n",
	       max_a_elem, max_r_elem);
	printf("Offending max absolute error (Actual, Expected): (%1.4f, %1.4f)\n",
	       max_abs_a_elem, max_abs_r_elem);
}

inline __device__ __host__ u_int ceil_div(u_int a, u_int b) { return (a + b - 1) / b; }

template <typename T = __nv_bfloat16>
__global__ void zero_kernel(T *ptr) {
	int tx = threadIdx.x;
	int bx = blockIdx.x;
	int nt = blockDim.x;
	int block_ptr_start = nt * bx;
	int idx = block_ptr_start + tx;

	ptr[idx] = T(0.0);
}

template <typename T = __nv_bfloat16>
void zero_(T *ptr, const int numel) {
	int BLOCKSIZE = 128;
	const dim3 grid(ceil_div(numel, BLOCKSIZE));
	zero_kernel<<<grid, BLOCKSIZE>>>(ptr);
}

// prints a row-major matrix to stdout
void printmat(const __nv_bfloat16 *A, size_t m, size_t n) {
	size_t lda = n;

	for (int r = 0; r < m; r++) {
		std::cout << "[ ";
		for (int c = 0; c < n; c++) {
			float elem = __bfloat162float(A[r * lda + c]);

			std::cout << std::fixed << std::setprecision(4) << std::setw(4) << elem << ", ";
		}
		std::cout << "],\n";
	}
}

// prints a row-major matrix to stdout
void printmeta(const uint16_t *A, size_t m, size_t n) {
	size_t lda = n;

	for (int r = 0; r < m; r++) {
		std::cout << "[ ";
		for (int c = 0; c < n; c++) {
			std::cout << std::fixed << std::setprecision(3) << std::setw(4) << A[r * lda + c] << " ";
		}
		std::cout << "]\n";
	}
}

}  // namespace common

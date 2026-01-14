#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cusparseLt.h>
#include <stdlib.h>

#include <algorithm>
#include <bitset>
#include <cstdint>
#include <iostream>

#include "common.hpp"
#include "cuda_check.hpp"
#include "kernels.hpp"

// type traits for cuSPARSELt
template <typename value_t>
struct cuda_type {};

template <>
struct cuda_type<__nv_bfloat16> {
	static constexpr cudaDataType value = CUDA_R_16BF;
};

template <typename value_t>
struct cusparse_compute_type {};

template <>
struct cusparse_compute_type<float> {
	static constexpr cusparseComputeType value = CUSPARSE_COMPUTE_32F;
};

void benchmark_mma_sp(bf16 *dA, bf16 *dB, bf16 *dC, u_int16_t *dMETA, size_t M, size_t N, size_t K, int iters, int warmup) {
	// following Triton benchmarks -> before each bench iter write 256MB to
	// clear L2
	int *cache_l2;
	size_t numel_l2 = 256e6;
	cudaMalloc(&cache_l2, numel_l2 * sizeof(int));

	float *milliseconds;
	cudaEvent_t *start, *stop;

	start = new cudaEvent_t[iters];
	stop = new cudaEvent_t[iters];
	milliseconds = new float[iters];

	for (int i = 0; i < iters; i++) {
		common::zero_<int>(cache_l2, numel_l2);

		cudaEventCreate(&start[i]);
		cudaEventCreate(&stop[i]);

		cudaEventRecord(start[i]);
		sparse::launch_gemm(dA, dB, dC, dMETA, M, N, K);
		cudaEventRecord(stop[i]);
		cudaEventSynchronize(stop[i]);

		cudaEventElapsedTime(&milliseconds[i], start[i], stop[i]);
	}

	CudaCheckError();

	double total = 0.0;

	for (auto s = warmup; s < iters; s++) {
		total += milliseconds[s];
	}
	double elapsed_time = total * 1e-3;
	double flops = (iters - warmup) * (2 * M * N * K);
	std::cout << "Problem Size: " << M << " x " << N << " x " << K << std::endl;
	std::cout << "Total Elapsed Time: " << elapsed_time << "s" << std::endl;
	std::cout << "TFLOP/s " << (flops * 1e-12) / elapsed_time << std::endl;

	delete[] start;
	delete[] stop;
	delete[] milliseconds;
};

void benchmark_cusparse(bf16 *dA_dense, bf16 *dB, bf16 *dC, size_t M, size_t N, size_t K, int iters, int warmup) {
	// cuSPARSELt configuration
	auto order = CUSPARSE_ORDER_ROW;
	auto opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
	auto opB = CUSPARSE_OPERATION_NON_TRANSPOSE;
	auto type_AB = cuda_type<bf16>::value;
	auto type_C = cuda_type<bf16>::value;
	auto compute_type = cusparse_compute_type<float>::value;
	unsigned alignment = 16;

	auto num_A_rows = M;
	auto num_A_cols = K;
	auto num_B_rows = K;
	auto num_B_cols = N;
	auto num_C_rows = M;
	auto num_C_cols = N;

	auto lda = num_A_cols;
	auto ldb = num_B_cols;
	auto ldc = num_C_cols;

	float alpha = 1.0f;
	float beta = 0.0f;

	// cuSPARSELt handles
	cusparseLtHandle_t handle;
	cusparseLtMatDescriptor_t matA, matB, matC;
	cusparseLtMatmulDescriptor_t matmul;
	cusparseLtMatmulAlgSelection_t alg_sel;
	cusparseLtMatmulPlan_t plan;
	cudaStream_t stream = nullptr;

	CHECK_CUSPARSE(cusparseLtInit(&handle));

	CHECK_CUSPARSE(cusparseLtStructuredDescriptorInit(
	    &handle, &matA, num_A_rows, num_A_cols, lda, alignment,
	    type_AB, order, CUSPARSELT_SPARSITY_50_PERCENT));

	CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(
	    &handle, &matB, num_B_rows, num_B_cols, ldb, alignment,
	    type_AB, order));

	CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(
	    &handle, &matC, num_C_rows, num_C_cols, ldc, alignment,
	    type_C, order));

	CHECK_CUSPARSE(cusparseLtMatmulDescriptorInit(
	    &handle, &matmul, opA, opB, &matA, &matB, &matC, &matC, compute_type));

	CHECK_CUSPARSE(cusparseLtMatmulAlgSelectionInit(
	    &handle, &alg_sel, &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT));

	CHECK_CUSPARSE(cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel));

	CHECK_CUSPARSE(cusparseLtMatmulDescSetAttribute(
	    &handle, &matmul, CUSPARSELT_MATMUL_SPARSE_MAT_POINTER, &dA_dense, sizeof(dA_dense)));

	CHECK_CUSPARSE(cusparseLtSpMMAPrune(&handle, &matmul, dA_dense, dA_dense,
	                                    CUSPARSELT_PRUNE_SPMMA_TILE, stream));

	int *d_valid;
	CHECK_CUDA_CUSPARSE(cudaMalloc((void **)&d_valid, sizeof(int)));
	CHECK_CUSPARSE(cusparseLtSpMMAPruneCheck(&handle, &matmul, dA_dense, d_valid, stream));

	int is_valid;
	CHECK_CUDA_CUSPARSE(cudaMemcpyAsync(&is_valid, d_valid, sizeof(int),
	                                    cudaMemcpyDeviceToHost, stream));
	CHECK_CUDA_CUSPARSE(cudaStreamSynchronize(stream));

	if (is_valid != 0) {
		std::printf(
		    "!!!! The matrix has been pruned in a wrong way. "
		    "cusparseLtMatmul will not provide correct results\n");
		cudaFree(d_valid);
		return;
	}

	size_t compressed_size, compressed_buffer_size;
	CHECK_CUSPARSE(cusparseLtSpMMACompressedSize(&handle, &plan,
	                                             &compressed_size, &compressed_buffer_size));

	bf16 *dA_compressed;
	void *dA_compressedBuffer;
	CHECK_CUDA_CUSPARSE(cudaMalloc((void **)&dA_compressed, compressed_size));
	CHECK_CUDA_CUSPARSE(cudaMalloc((void **)&dA_compressedBuffer, compressed_buffer_size));

	CHECK_CUSPARSE(cusparseLtSpMMACompress(&handle, &plan, dA_dense, dA_compressed,
	                                       dA_compressedBuffer, stream));

	int num_streams = 0;
	cudaStream_t *streams = nullptr;

	CHECK_CUSPARSE(cusparseLtMatmulSearch(&handle, &plan, &alpha,
	                                      dA_compressed, dB, &beta,
	                                      dC, dC, nullptr,
	                                      streams, num_streams));

	size_t workspace_size;
	CHECK_CUSPARSE(cusparseLtMatmulGetWorkspace(&handle, &plan, &workspace_size));

	void *d_workspace;
	CHECK_CUDA_CUSPARSE(cudaMalloc((void **)&d_workspace, workspace_size));

	int *cache_l2;
	size_t numel_l2 = 256e6;
	cudaMalloc(&cache_l2, numel_l2 * sizeof(int));

	float *milliseconds;
	cudaEvent_t *start, *stop;

	start = new cudaEvent_t[iters];
	stop = new cudaEvent_t[iters];
	milliseconds = new float[iters];

	for (int i = 0; i < iters; i++) {
		common::zero_<int>(cache_l2, numel_l2);

		cudaEventCreate(&start[i]);
		cudaEventCreate(&stop[i]);

		cudaEventRecord(start[i]);
		cusparseLtMatmul(&handle, &plan, &alpha, dA_compressed, dB,
		                 &beta, dC, dC, d_workspace, streams, num_streams);
		cudaEventRecord(stop[i]);
		cudaEventSynchronize(stop[i]);

		cudaEventElapsedTime(&milliseconds[i], start[i], stop[i]);
	}

	double total = 0.0;
	for (auto s = warmup; s < iters; s++) {
		total += milliseconds[s];
	}
	double elapsed_time = total * 1e-3;
	double flops = (iters - warmup) * (2.0 * M * N * K);
	std::cout << "Problem Size: " << M << " x " << N << " x " << K << std::endl;
	std::cout << "Total Elapsed Time: " << elapsed_time << "s" << std::endl;
	std::cout << "TFLOP/s " << (flops * 1e-12) / elapsed_time << std::endl;

	delete[] start;
	delete[] stop;
	delete[] milliseconds;

	cudaFree(cache_l2);
	cudaFree(d_valid);
	cudaFree(dA_compressed);
	cudaFree(dA_compressedBuffer);
	cudaFree(d_workspace);

	CHECK_CUSPARSE(cusparseLtMatDescriptorDestroy(&matA));
	CHECK_CUSPARSE(cusparseLtMatDescriptorDestroy(&matB));
	CHECK_CUSPARSE(cusparseLtMatDescriptorDestroy(&matC));
	CHECK_CUSPARSE(cusparseLtMatmulAlgSelectionDestroy(&alg_sel));
	CHECK_CUSPARSE(cusparseLtMatmulPlanDestroy(&plan));
	CHECK_CUSPARSE(cusparseLtDestroy(&handle));
};

int main(int argc, char *argv[]) {
	assert(argc == 7);
	const size_t M = atoi(argv[1]);
	const size_t N = atoi(argv[2]);
	const size_t K = atoi(argv[3]);
	const bool benchmark = atoi(argv[4]);
	const bool check_correctness = atoi(argv[5]);
	const bool benchmark_cusparselt = atoi(argv[6]);

	const int iters = 250;
	const int warmup = 50;

	const size_t numelA = M * K;
	const size_t numelB = K * N;
	const size_t numelC = M * N;

	const size_t numelSp = M * (K / 2);

	const size_t numelMeta = M * (K / 16);

	uint16_t *META, *dMETA, *METASwizz;

	bf16 *A, *Asparse, *B, *C, *C_ref, *dA, *dB, *dC;

	// copies needed for cuSPARSE
	bf16 *A_dense = nullptr;
	bf16 *dA_dense = nullptr;

	A = new bf16[numelA];
	B = new bf16[numelB];
	C = new bf16[numelC];

	Asparse = new bf16[numelSp];

	META = new uint16_t[numelMeta];
	METASwizz = new uint16_t[numelMeta];

	C_ref = new bf16[numelC];

	common::fill_random(A, numelA);
	common::fill_random(B, numelB);

	// save a copy of dense A for cuSPARSELt before sparsification
	if (benchmark_cusparselt) {
		A_dense = new bf16[numelA];
		std::copy(A, A + numelA, A_dense);
	}

	common::sparsify24(A, Asparse, META, M, K);

	common::swizzleMeta(META, METASwizz, numelMeta, (K / 4));

	cudaMalloc(&dA, numelSp * sizeof(bf16));
	cudaMalloc(&dB, numelB * sizeof(bf16));
	cudaMalloc(&dC, numelC * sizeof(bf16));

	cudaMalloc(&dMETA, numelMeta * sizeof(uint16_t));

	cudaMemcpy(dA, Asparse, numelSp * sizeof(bf16), cudaMemcpyHostToDevice);
	cudaMemcpy(dB, B, numelB * sizeof(bf16), cudaMemcpyHostToDevice);
	cudaMemcpy(dMETA, METASwizz, numelMeta * sizeof(uint16_t), cudaMemcpyHostToDevice);

	sparse::launch_gemm(dA, dB, dC, dMETA, M, N, K);

	CudaCheckError();
	cudaMemcpy(C, dC, numelC * sizeof(bf16), cudaMemcpyDeviceToHost);

	// perform a slow CPU GEMM and check output
	if (check_correctness) {
		common::cpu_gemm_tt(A, B, C_ref, M, N, K);
		common::check_error(C, C_ref, numelC);
	}

	if (benchmark) {
		std::cout << "=== Custom Sparse GEMM Benchmark ===" << std::endl;
		benchmark_mma_sp(dA, dB, dC, dMETA, M, N, K, iters, warmup);
	}

	if (benchmark_cusparselt) {
		cudaMalloc(&dA_dense, numelA * sizeof(bf16));
		cudaMemcpy(dA_dense, A_dense, numelA * sizeof(bf16), cudaMemcpyHostToDevice);
		std::cout << "\n=== cuSPARSELt Benchmark ===" << std::endl;
		benchmark_cusparse(dA_dense, dB, dC, M, N, K, iters, warmup);

		cudaFree(dA_dense);
		delete[] A_dense;
	}

	delete[] A;
	delete[] B;
	delete[] C;
	delete[] META;
	delete[] METASwizz;

	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);
	cudaFree(dMETA);

	return 0;
}

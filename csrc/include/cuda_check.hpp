#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cusparseLt.h>

#include <cstdio>
#include <cstdlib>

#define CudaCheckError() __cudaCheckError(__FILE__, __LINE__)
inline void __cudaCheckError(const char *file, const int line) {
	cudaError err = cudaGetLastError();
	if (cudaSuccess != err) {
		fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n", file, line,
		        cudaGetErrorString(err));
		exit(-1);
	}
	err = cudaDeviceSynchronize();
	if (cudaSuccess != err) {
		fprintf(stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
		        file, line, cudaGetErrorString(err));
		exit(-1);
	}
}

#define CHECK_CUSPARSE(func)                                               \
	{                                                                      \
		cusparseStatus_t status = (func);                                  \
		if (status != CUSPARSE_STATUS_SUCCESS) {                           \
			printf("CUSPARSE API failed at line %d with error: %s (%d)\n", \
			       __LINE__, cusparseLtGetErrorString(status), status);    \
			return;                                                        \
		}                                                                  \
	}

#define CHECK_CUDA_CUSPARSE(func)                                      \
	{                                                                  \
		cudaError_t status = (func);                                   \
		if (status != cudaSuccess) {                                   \
			printf("CUDA API failed at line %d with error: %s (%d)\n", \
			       __LINE__, cudaGetErrorString(status), status);      \
			return;                                                    \
		}                                                              \
	}

#pragma once

#include <cuda_bf16.h>

#include <cmath>
#include <cstdint>
namespace sparse {

#define bf16 __nv_bfloat16
#define bf162 __nv_bfloat162

#define NUM_THREADS 128

#define WARP_SIZE 32

#define BM 128
#define BN 128
#define BK 64
#define BKSp 32

#define MMA_PER_M 4
#define MMA_PER_N 8

#define MMA_M 16
#define MMA_N 8
#define MMA_K 32

#define WARP_COLS 2

#define NUMEL_128 8
#define NUMEL_64 4

#define NBUFFER 3

inline __device__ __host__ u_int ceil_div(u_int a, u_int b) { return (a + b - 1) / b; }

// https://leimao.github.io/blog/CuTe-Swizzle/#Examples
constexpr int constexpr_log2(int n) {
	return ((n < 2) ? 0 : 1 + constexpr_log2(n / 2));
}

constexpr u_int base{constexpr_log2(NUMEL_128)};
constexpr u_int bits{constexpr_log2(32 * sizeof(__nv_bfloat16)) - base};

constexpr u_int shiftA{constexpr_log2(BK / 2) - base};
constexpr u_int shiftB{constexpr_log2(BN) - base};
constexpr u_int shiftC{constexpr_log2(BN) - base};
constexpr u_int bitMask = (1 << bits) - 1;
constexpr u_int swizzleMaskA = bitMask << (base + shiftA);
constexpr u_int swizzleMaskB = bitMask << (base + shiftB);
constexpr u_int swizzleMaskC = bitMask << (base + shiftC);

__forceinline__ __device__ void
mma_sp_m16n8k32(const unsigned* A, const unsigned* B, float* C, float* D, const u_int e, const u_int selector) {
	if (selector == 0) {
		asm volatile(
		    "mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10, %11}, "
		    "{%12, %13, %14, %15}, %16, 0x0;\n"
		    : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
		    : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
		      "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
		      "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]), "r"(e));
	} else {
		asm volatile(
		    "mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10, %11}, "
		    "{%12, %13, %14, %15}, %16, 0x1;\n"
		    : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
		    : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
		      "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
		      "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]), "r"(e));
	}
}

__forceinline__ __device__ void
ldmatrix_x4_offset(unsigned* D, u_int addr32) {
	asm volatile(
	    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
	    : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
	    : "r"(addr32));
}

__forceinline__ __device__ void
ldmatrix_x4trans_offset(unsigned* D, u_int addr32) {
	asm volatile(
	    "ldmatrix.sync.aligned.m8n8.trans.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
	    : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
	    : "r"(addr32));
}

template <u_int shift>
__forceinline__ __device__
    u_int
    shiftr(u_int index) {
	return index >> shift;
}

template <u_int shift, u_int mask>
__forceinline__ __device__
    u_int
    swizzle(u_int index) {
	return (index ^ shiftr<shift>(index & mask));
}

__forceinline__ __device__ void cp_async_commit() {
	asm volatile("cp.async.commit_group;\n");
}

__forceinline__ __device__ void cp_async_wait_all() {
	asm volatile("cp.async.wait_all; \n");
}

__forceinline__ __device__ void cp_async_load16_offset(u_int dst_addr32, const uint4* src) {
	asm volatile(
	    "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst_addr32), "l"(src), "n"(16));
}

// waits until _at most_ the last N most recent commit groups are still in flight
template <u_int N>
__forceinline__ __device__ void cp_async_wait_group() {
	asm volatile("cp.async.wait_group %0; \n" ::"n"(N));
}

template <u_int load_per_thread>
__forceinline__ __device__ void loadFromGlobal(
    u_int sharedBase,
    u_int initialSharedIndex,
    const bf16* global,
    u_int globalIncr) {
	u_int sharedAddr = sharedBase + initialSharedIndex * sizeof(bf16);
	const bf16* globalPtr = global;

	constexpr u_int sharedIncr = NUM_THREADS * NUMEL_128 * sizeof(bf16);

	for (u_int load = 0; load < load_per_thread; load++) {
		cp_async_load16_offset(sharedAddr, (const uint4*)globalPtr);
		globalPtr += globalIncr;
		sharedAddr += sharedIncr;
	}
}

template <u_int store_per_thread>
__forceinline__ __device__ void storeToGlobal(bf16* global, const bf16* shared, u_int thread_row, u_int thread_col, u_int ldG, int initialIndex, int globalIncr) {
	int storeOff = 0;
	const bf16* sharedPtr = shared + initialIndex;

	bf16* globalPtr = global;

	constexpr int sharedIncr = NUM_THREADS * NUMEL_128;

	for (u_int store = 0; store < store_per_thread; store++) {
		*((uint4*)(globalPtr)) = *((uint4*)(sharedPtr + storeOff));
		storeOff += sharedIncr;
		globalPtr += globalIncr;
	}
}

template <u_int load_per_thread>
__forceinline__ __device__ void loadMeta(
    u_int sharedBase,
    u_int sharedIndex,
    const uint16_t* global,
    u_int globalIncr, u_int sharedIncr) {
	u_int sharedAddr = sharedBase + sharedIndex * sizeof(uint16_t);
	const uint16_t* globalPtr = global;

	u_int sharedIncrBytes = sharedIncr * sizeof(uint16_t);

	for (u_int load = 0; load < load_per_thread; load++) {
		cp_async_load16_offset(sharedAddr, (const uint4*)globalPtr);
		globalPtr += globalIncr;
		sharedAddr += sharedIncrBytes;
	}
}

__forceinline__ __device__ void loadAtoRegs(
    bf16* __restrict__ aReg,
    u_int AsBase,
    u_int index,
    u_int regStride) {
	u_int addr32 = AsBase + index * sizeof(bf16);
	bf16* aRegStart = &aReg[0];

	for (u_int aRow = 0; aRow < MMA_PER_M; aRow++) {
		unsigned* aPtrld = reinterpret_cast<unsigned*>(aRegStart);
		ldmatrix_x4_offset(aPtrld, addr32);
		addr32 += (MMA_M * BKSp) * sizeof(bf16);
		aRegStart += regStride;
	}
}

__forceinline__ __device__ void loadBtoRegs(
    bf16* __restrict__ bReg,
    u_int BsBase,
    u_int index,
    u_int regStride) {
	u_int addr0 = BsBase + index * sizeof(bf16);
	index ^= 0b00000001000;
	u_int addr1 = BsBase + index * sizeof(bf16);
	index ^= 0b00000011000;
	u_int addr2 = BsBase + index * sizeof(bf16);
	index ^= 0b00000001000;
	u_int addr3 = BsBase + index * sizeof(bf16);
	index ^= 0b00000111000;
	u_int addr4 = BsBase + index * sizeof(bf16);
	index ^= 0b00000001000;
	u_int addr5 = BsBase + index * sizeof(bf16);
	index ^= 0b00000011000;
	u_int addr6 = BsBase + index * sizeof(bf16);
	index ^= 0b00000001000;
	u_int addr7 = BsBase + index * sizeof(bf16);

	ldmatrix_x4trans_offset(reinterpret_cast<unsigned*>(&bReg[0 * regStride]), addr0);
	ldmatrix_x4trans_offset(reinterpret_cast<unsigned*>(&bReg[1 * regStride]), addr1);
	ldmatrix_x4trans_offset(reinterpret_cast<unsigned*>(&bReg[2 * regStride]), addr2);
	ldmatrix_x4trans_offset(reinterpret_cast<unsigned*>(&bReg[3 * regStride]), addr3);
	ldmatrix_x4trans_offset(reinterpret_cast<unsigned*>(&bReg[4 * regStride]), addr4);
	ldmatrix_x4trans_offset(reinterpret_cast<unsigned*>(&bReg[5 * regStride]), addr5);
	ldmatrix_x4trans_offset(reinterpret_cast<unsigned*>(&bReg[6 * regStride]), addr6);
	ldmatrix_x4trans_offset(reinterpret_cast<unsigned*>(&bReg[7 * regStride]), addr7);
}

__forceinline__ __device__ void loadMetatoRegs(
    u_int* __restrict__ mReg,
    u_int MsBase) {
	uint16_t mRegTmp[2 * MMA_PER_M];
	ldmatrix_x4_offset(reinterpret_cast<unsigned*>(&mRegTmp), MsBase);
	mReg[0] = (mRegTmp[1] << 16) | mRegTmp[0];
	mReg[1] = (mRegTmp[3] << 16) | mRegTmp[2];
	mReg[2] = (mRegTmp[5] << 16) | mRegTmp[4];
	mReg[3] = (mRegTmp[7] << 16) | mRegTmp[6];
}

__forceinline__ __device__ void mma_ABsp(float* __restrict__ dReg, const u_int* __restrict__ mReg, const bf16* __restrict__ aReg, const bf16* __restrict__ bReg, const u_int pselect, const u_int ab_stride) {
	float* dRegRow = dReg;
	for (u_int aRow = 0; aRow < MMA_PER_M; aRow++) {
		float* dRegCol = dRegRow;

		unsigned const* aPtr = reinterpret_cast<unsigned const*>(&aReg[ab_stride * aRow]);

#pragma unroll
		for (u_int bCol = 0; bCol < MMA_PER_N; bCol++) {
			unsigned const* bPtr = reinterpret_cast<unsigned const*>(&bReg[ab_stride * bCol]);
			mma_sp_m16n8k32(aPtr, bPtr, dRegCol, dRegCol, mReg[aRow], pselect);
			dRegCol += MMA_PER_M;
		}
		dRegRow += MMA_PER_N * MMA_PER_M;
	}
}

template <u_int GROUP_SIZE = 8>
__launch_bounds__(NUM_THREADS, 1)
    __global__ void sparse_gemm(bf16 const* __restrict__ A,
                                bf16 const* __restrict__ B,
                                bf16* __restrict__ C,
                                uint16_t const* __restrict__ META, int M, int N, int K) {
	constexpr u_int mmaPerK = BK / MMA_K;
	constexpr u_int colsMs = 4 * mmaPerK;

	const u_int ldA = K / 2;
	const u_int ldB = N;
	const u_int ldC = N;
	const u_int ldMETA = (K / 8);

	// TB swizzling for L2 -> 4-5 TFLOPs
	// https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html#l2-cache-optimizations
	u_int linear_idx = blockIdx.x + blockIdx.y * gridDim.x;
	u_int block_per_grp = GROUP_SIZE * gridDim.x;
	u_int grp_id = linear_idx / block_per_grp;
	u_int firstBlockM = grp_id * GROUP_SIZE;
	u_int groupSizeM = min((gridDim.y - firstBlockM), GROUP_SIZE);

	uint row = (firstBlockM + (linear_idx % groupSizeM) % groupSizeM) * BM;
	uint col = ((linear_idx % block_per_grp) / groupSizeM) * BN;

	META += (row / 2) * ldMETA;

	A += row * ldA;
	B += col;

	extern __shared__ bf16 shared[];

	bf16* As = &shared[0];
	bf16* Bs = &shared[NBUFFER * BM * BKSp];
	uint16_t* Ms = reinterpret_cast<uint16_t*>(&shared[(NBUFFER * ((BM * BKSp) + (BK * BN)))]);

	const u_int tx = threadIdx.x;

	const u_int laneId = tx % WARP_SIZE;
	const u_int warpId = tx / WARP_SIZE;

	const u_int loadsPerThreadA = (BKSp * BM) / (NUM_THREADS * NUMEL_128);
	const u_int loadsPerThreadB = (BK * BN) / (NUM_THREADS * NUMEL_128);

	const u_int loadsPerThreadMeta = ((BM / 2) * colsMs) / (NUMEL_64 * NUM_THREADS);

	const u_int threadRowA = tx / (BKSp / NUMEL_128);
	const u_int threadColA = NUMEL_128 * (tx % (BKSp / NUMEL_128));

	const u_int threadRowB = tx / (BN / NUMEL_128);
	const u_int threadColB = NUMEL_128 * (tx % (BN / NUMEL_128));

	const u_int rowsPerTbA = (NUMEL_128 * NUM_THREADS) / BKSp;
	const u_int rowsPerTbB = (NUMEL_128 * NUM_THREADS) / BN;

	const u_int rowsPerTbMeta = (NUMEL_64 * NUM_THREADS) / colsMs;

	const u_int ldMs = colsMs;
	const u_int threadRowMeta = tx / (colsMs / NUMEL_64);
	const u_int threadColMeta = NUMEL_64 * (tx % (colsMs / NUMEL_64));

	bf16 aReg[2][MMA_PER_M][8];
	bf16 bReg[2][MMA_PER_N][8];

	float dReg[MMA_PER_M][MMA_PER_N][4] = {0.};

	u_int mReg[MMA_PER_M] = {};

	u_int warpRowMeta = (warpId / WARP_COLS) * MMA_PER_M * (MMA_M / 2);
	u_int warpRowA = (warpId / WARP_COLS) * MMA_PER_M * MMA_M;
	u_int warpColB = (warpId % WARP_COLS) * MMA_PER_N * MMA_N;

	const u_int stageStrideM = (BM / 2) * colsMs;

	const bool doesGtoS = (threadIdx.x % 2) == 0;

	const u_int indexAG = swizzle<shiftA, swizzleMaskA>(threadRowA * BKSp + threadColA);
	const u_int indexBG = swizzle<shiftB, swizzleMaskB>(threadRowB * BN + threadColB);

	const u_int Astride = (BK / 2);
	const u_int Bstride = BK * ldB;
	const u_int Mstride = 4 * mmaPerK;

	const u_int MsOffset = (warpRowMeta + laneId) * ldMs;
	const u_int AsOffset = (warpRowA + (laneId % MMA_M)) * BKSp + (laneId / MMA_M * NUMEL_128);
	const u_int BsOffset = ((laneId % MMA_K)) * BN + warpColB;

	int globalIncrA = ldA * rowsPerTbA;
	int globalIncrB = ldB * rowsPerTbB;

	int globalIncrM = ldMETA * rowsPerTbMeta;
	constexpr int sharedIncrM = ldMs * rowsPerTbMeta;

	u_int indexB = swizzle<shiftB, swizzleMaskB>(BsOffset);

	u_int AGlobalIndex = threadRowA * ldA + threadColA;
	u_int BGlobalIndex = threadRowB * ldB + threadColB;
	u_int MetaGlobalIndex = threadRowMeta * ldMETA + threadColMeta;
	u_int MetaSharedIndex = threadRowMeta * ldMs + threadColMeta;

	const bf16* ALoadGlobal = A + AGlobalIndex;
	const bf16* BLoadGlobal = B + BGlobalIndex;
	const uint16_t* METALoadGlobal = META + MetaGlobalIndex;

	u_int AsBase = __cvta_generic_to_shared(As);
	u_int BsBase = __cvta_generic_to_shared(Bs);
	u_int MsBase = __cvta_generic_to_shared(Ms);

	constexpr u_int stageStrideB_bytes = BK * BN * sizeof(bf16);
	constexpr u_int stageStrideA_bytes = BM * BKSp * sizeof(bf16);
	constexpr u_int stageStrideM_bytes = (BM / 2) * colsMs * sizeof(uint16_t);

	int kTileCount = K / BK;
	int kTileNext = 0;
	int smemPipeRead = 0;
	int smemPipeWrite = NBUFFER - 1;

#pragma unroll
	for (int kPipe = 0; kPipe < NBUFFER - 1; ++kPipe) {
		u_int AsLoadBase = AsBase + kPipe * stageStrideA_bytes;
		u_int BsLoadBase = BsBase + kPipe * stageStrideB_bytes;
		u_int MsLoadBase = MsBase + kPipe * stageStrideM_bytes;

		loadFromGlobal<loadsPerThreadB>(BsLoadBase, indexBG, BLoadGlobal, globalIncrB);
		loadFromGlobal<loadsPerThreadA>(AsLoadBase, indexAG, ALoadGlobal, globalIncrA);

		if (doesGtoS) {
			loadMeta<loadsPerThreadMeta>(MsLoadBase, MetaSharedIndex, METALoadGlobal, globalIncrM, sharedIncrM);
		}
		cp_async_commit();
		--kTileCount;
		if (kTileCount > 0) {
			++kTileNext;
			ALoadGlobal += Astride;
			BLoadGlobal += Bstride;
			METALoadGlobal += Mstride;
		}
	}
	const u_int indexA0 = swizzle<shiftA, swizzleMaskA>(AsOffset);
	const u_int indexA1 = swizzle<shiftA, swizzleMaskA>(AsOffset + MMA_K / 2);

	cp_async_wait_group<NBUFFER - 2>();
	__syncthreads();

	u_int AsComputeBase = AsBase + smemPipeRead * stageStrideA_bytes;
	u_int BsComputeBase = BsBase + smemPipeRead * stageStrideB_bytes;
	u_int MsComputeAddr = MsBase + smemPipeRead * stageStrideM_bytes + 2 * MsOffset;

	loadMetatoRegs(&mReg[0], MsComputeAddr);

	// load the first k block to regs
	loadAtoRegs(&aReg[0][0][0], AsComputeBase, indexA0, 8);
	loadBtoRegs(&bReg[0][0][0], BsComputeBase, indexB, 8);

#pragma nounroll
	while (kTileCount > -(NBUFFER - 1)) {
		// load the second k block to regs
		loadAtoRegs(&aReg[1][0][0], AsComputeBase, indexA1, 8);
		loadBtoRegs(&bReg[1][0][0], BsComputeBase, indexB + MMA_K * BN, 8);

		u_int AsLoadBase = AsBase + smemPipeWrite * stageStrideA_bytes;
		u_int BsLoadBase = BsBase + smemPipeWrite * stageStrideB_bytes;
		u_int MsLoadBase = MsBase + smemPipeWrite * stageStrideM_bytes;

		loadFromGlobal<loadsPerThreadB>(BsLoadBase, indexBG, BLoadGlobal, globalIncrB);
		loadFromGlobal<loadsPerThreadA>(AsLoadBase, indexAG, ALoadGlobal, globalIncrA);
		if (doesGtoS) {
			loadMeta<loadsPerThreadMeta>(MsLoadBase, MetaSharedIndex, METALoadGlobal, globalIncrM, sharedIncrM);
		}
		cp_async_commit();

		--kTileCount;
		if (kTileCount > 0) {
			ALoadGlobal += Astride;
			BLoadGlobal += Bstride;
			METALoadGlobal += Mstride;
		}

		smemPipeWrite = smemPipeRead;

		mma_ABsp(&dReg[0][0][0], &mReg[0], &aReg[0][0][0], &bReg[0][0][0], 0, 8);
		mma_ABsp(&dReg[0][0][0], &mReg[0], &aReg[1][0][0], &bReg[1][0][0], 1, 8);

		smemPipeRead = (smemPipeRead == NBUFFER - 1) ? 0 : smemPipeRead + 1;

		cp_async_wait_group<NBUFFER - 2>();
		__syncthreads();

		AsComputeBase = AsBase + smemPipeRead * stageStrideA_bytes;
		BsComputeBase = BsBase + smemPipeRead * stageStrideB_bytes;
		MsComputeAddr = MsBase + (smemPipeRead * stageStrideM + MsOffset) * sizeof(uint16_t);

		loadMetatoRegs(&mReg[0], MsComputeAddr);
		loadAtoRegs(&aReg[0][0][0], AsComputeBase, indexA0, 8);
		loadBtoRegs(&bReg[0][0][0], BsComputeBase, indexB, 8);
	}

	cp_async_wait_all();
	__syncthreads();

	bf16* Cs = As;
	const u_int ldCs = BN;

	const u_int warpRow = (warpId / WARP_COLS) * MMA_PER_M * MMA_M;
	const u_int warpCol = (warpId % WARP_COLS) * MMA_PER_N * MMA_N;

	const u_int laneGroup = laneId >> 2;
	const u_int laneInGroup = laneId & 3;
	const u_int threadCol0 = 2 * laneInGroup;

	const u_int warpRowlaneGroup = warpRow + laneGroup;
	const u_int warpColthreadCol0 = warpCol + threadCol0;

	// we pre-compute the xor swizzle to avoid calling `swizzle` in the hot loops
	// this way we only compute a single xor to get the swizzled position
	u_int xors[8] = {
	    0b00000000000,
	    0b00000001000,
	    0b00000011000,
	    0b00000001000,
	    0b00000111000,
	    0b00000001000,
	    0b00000011000,
	    0b00000001000};

	u_int initialIndexC = swizzle<shiftC, swizzleMaskC>(ldCs * warpRowlaneGroup + warpColthreadCol0);

	constexpr u_int storePerThreadC = (BM * BN) / (NUM_THREADS * NUMEL_128);
	constexpr u_int rowsPerTbC = (NUMEL_128 * NUM_THREADS) / BN;

	const u_int threadRowC = tx / (BN / NUMEL_128);
	const u_int threadColC = NUMEL_128 * (tx % (BN / NUMEL_128));

	int globalIncrC = ldC * rowsPerTbC;
	int initialIndex = swizzle<shiftC, swizzleMaskC>(threadRowC * ldCs + threadColC);
	int CGlobalIndex = (threadRowC * ldC + threadColC);

#pragma unroll
	for (u_int cRow = 0; cRow < MMA_PER_M; ++cRow) {
		u_int idx0 = initialIndexC + (storePerThreadC * NUM_THREADS * cRow);

#pragma unroll
		for (u_int cCol = 0; cCol < MMA_PER_N; ++cCol) {
			idx0 ^= xors[cCol];
			u_int idx2 = idx0 + rowsPerTbC * NUM_THREADS;
			*(bf162*)(&Cs[idx0]) = __float22bfloat162_rn(*(float2*)&dReg[cRow][cCol][0]);
			*(bf162*)(&Cs[idx2]) = __float22bfloat162_rn(*(float2*)&dReg[cRow][cCol][2]);
		}
	}

	C += row * ldC + col + CGlobalIndex;

	storeToGlobal<storePerThreadC>(C, Cs, threadRowC, threadColC, ldC, initialIndex, globalIncrC);
}

void launch_gemm(bf16* A, bf16* B, bf16* C, uint16_t* META, int M, int N, int K) {
	dim3 blockDim(NUM_THREADS);
	dim3 gridDim(ceil_div(N, BN), ceil_div(M, BM));

	const u_int GROUP_SIZE = 8;

	constexpr size_t AsBytes = NBUFFER * (BM * (BK / 2)) * sizeof(bf16);
	constexpr size_t BsBytes = NBUFFER * (BK * BN) * sizeof(bf16);
	constexpr size_t MsBytes = NBUFFER * (BM * (2 * (BK / MMA_K))) * sizeof(uint16_t);
	constexpr size_t mem_size = AsBytes + BsBytes + MsBytes;

	cudaDeviceSynchronize();
	cudaFuncSetAttribute(sparse_gemm<GROUP_SIZE>, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);

	sparse_gemm<GROUP_SIZE><<<gridDim, blockDim, mem_size>>>(A, B, C, META, M, N, K);
}
}  // namespace sparse

/*
 ============================================================================
 Name        : Isolated_SW.cu
 Author      : Vuong Pham Duy
 Version     :
 Copyright   : Your copyright notice
 Description : debugging Smith-Waterman Score Matrix Kernel
 ============================================================================
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>
#include <assert.h>
#include <getopt.h>
#include <time.h>

#define WARP 1024
#define LIKELY(x) __builtin_expect((x),1)
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

#define A 1
#define B 4
#define MATH_SIZE 5
#define O_DEL 6
#define E_DEL 1
#define O_INS 6
#define E_INS 1
#define W 100
#define END_BONUS 5
#define ZDROP 100
#define H0 200
#define TLEN 10
#define QLEN 10
#define NUM 100
#define RANGE 4

typedef struct {
	int32_t h, e;
} eh_t;

void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", \
			cudaGetErrorString(code), file, line);
		if (abort)
			exit(code);
	}
}

void bwa_fill_scmat(int a, int b, int8_t mat[25]);
int ksw_extend2(int *cmax_i, int *cmax_j, int *cmax_ie, int *cgscore, int *cmax_off, \
		int qlen, const uint8_t *query, int tlen, const uint8_t *target, int m, const int8_t *mat, \
		int o_del, int e_del, int o_ins, int e_ins, int w, int end_bonus, int zdrop, int h0);

__device__
bool check_active(int32_t h, int32_t e)
{
	if(h != -1 && e != -1) return true;
	else return false;
}
__device__
void reset(int32_t *h, int32_t *e)
{
	*h = -1;
	*e = -1;
}

__device__ int mLock = 0;
extern __shared__ int32_t container[];

__global__
void sw_kernel(int *d_max, int *d_max_j, int *d_max_i, int *d_max_ie, int *d_gscore, int *d_max_off, \
		int w, int oe_ins, int e_ins, int o_del, int e_del, int oe_del, int m, \
		int tlen, int qlen, int passes, int t_lastp, int h0, int zdrop, \
		int32_t *h, int8_t *qp, const uint8_t *target)
{
	__shared__ int break_cnt;
	__shared__ int max;
	__shared__ int max_i;
	__shared__ int max_j;
	__shared__ int max_ie;
	__shared__ int gscore;
	__shared__ int max_off;
	__shared__ int out_h[WARP];
	__shared__ int out_e[WARP];

	bool blocked = true;
	int in_h, in_e;
	int i;
	int active_ts, beg, end;
	int32_t *se, *sh;
	int8_t *sqp;

	/* Initialize */
	if(threadIdx.x == 0) {
		max = h0;
		max_i = -1;
		max_j = -1;
		max_ie = -1;
		gscore = -1;
		max_off = 0;
		break_cnt = 0;
	}

	i = threadIdx.x;
	sh = container;
	se = (int32_t*)&sh[qlen + 1];
	sqp = (int8_t*)&se[qlen + 1];
	for(;;) {
		if(i < qlen + 1) {
			sh[i] = h[i];
			se[i] = 0;
		}
		// qlen > 1, m = 5, qlen * m always bigger than qlen + 1
		if(i < qlen * m) {
			sqp[i] = qp[i];
		}
		else break;
		i += WARP;
	}
	__syncthreads();
	for(int i = 0; i < passes; i++) {
		if(i == passes - 1) {
			if(threadIdx.x >= t_lastp) break;
			else active_ts = t_lastp;
		} else active_ts = WARP;
		reset(&in_h, &in_e);
		reset(&out_h[threadIdx.x], &out_e[threadIdx.x]);
		beg = 0; end = qlen;

		int t, row_i, f = 0, h1, local_m = 0, mj = -1;
		row_i = i * WARP + threadIdx.x;
		int8_t *q = &sqp[target[row_i] * qlen];
		// apply the band and the constraint (if provided)
		if (beg < row_i - w) beg = row_i - w;
		if (end > row_i + w + 1) end = row_i + w + 1;
		if (end > qlen) end = qlen;
		// reset input, output

		if (beg == 0) {
			h1 = h0 - (o_del + e_del * (row_i + 1));
			if (h1 < 0) h1 = 0;
		} else h1 = 0;

		__syncthreads();

		 while(beg < end) {
			if(threadIdx.x == 0) {
				in_h = sh[beg];
				in_e = se[beg];
			} else {
				in_h = out_h[threadIdx.x - 1];
				in_e = out_e[threadIdx.x - 1];
			}
			__syncthreads();
			if(check_active(in_h, in_e)) {
				int local_h;

				out_h[threadIdx.x] = h1;
				if(i != passes - 1) sh[beg] = h1;

				//in_h = in_h? in_h + q[beg] : 0;
				if(in_h) in_h = in_h + q[beg];
				else in_h = 0;

				// local_h = in_h > in_e? in_h : in_e;
				if(in_h > in_e) local_h = in_h;
				else local_h = in_e;

				// local_h = local_h > f? local_h : f;
				if(local_h < f) local_h = f;

				h1 = local_h;

				// mj = local_m > local_h? mj : beg;
				if(local_m <= local_h) mj = beg;
				//local_m = local_m > local_h? local_m : local_h;
				if(local_m < local_h) local_m = local_h;

				t = in_h - oe_del;
				//t = t > 0? t : 0;
				if(t < 0) t = 0;
				in_e -= e_del;
				//in_e = in_e > t? in_e : t;
				if(in_e < t) in_e = t;
				out_e[threadIdx.x] = in_e;
				if(i != passes - 1) se[beg] = in_e;

				t = in_h - oe_ins;
				//t = t > 0? t : 0;
				if(t < 0) t = 0;
				f -= e_ins;
				//f = f > t? f : t;
				if(f < t) f = t;

				reset(&in_h, &in_e);
				beg += 1;
			}
			__syncthreads();
		};
		if(threadIdx.x == active_ts - 1) {
			out_h[threadIdx.x] = h1;
			out_e[threadIdx.x] = 0;
			if(i != passes - 1) {
				sh[end] = h1;
				se[end] = 0;
			}
		}

		blocked = true;
		while(blocked) {
			if(0 == atomicCAS(&mLock, 0, 1)) {
				// critical section
				if(beg == qlen) {
					max_ie = gscore > h1? max_ie : row_i;
					gscore = gscore > h1? gscore : h1;
				}
				atomicExch(&mLock, 0);
				blocked = false;
			}
		}

		if (m == 0) atomicAdd(&break_cnt, 1);
		__syncthreads();
		if (break_cnt > 0) break;

		blocked = true;
		while(blocked) {
			if (break_cnt > 0) break;
			if(0 == atomicCAS(&mLock, 0, 1)) {
				if(local_m > max) {
					max = local_m, max_i = row_i, max_j = mj;
					max_off = max_off > abs(mj - row_i)? max_off : abs(mj - row_i);
				} else if (zdrop > 0) {
					if (i - max_i > mj - max_j) {
						if (max - local_m - ((row_i - max_i) - (mj - max_j)) * e_del > zdrop) break_cnt += 1;
					} else {
						if (max - local_m - ((mj - max_j) - (row_i - max_i)) * e_ins > zdrop) break_cnt += 1;
					}
				}
				atomicExch(&mLock, 0);
				blocked = false;
			}
		}
		if (break_cnt > 0) break;
	}
	__syncthreads();
	if(threadIdx.x == 0) {
		*d_max = max;
		*d_max_i = max_i;
		*d_max_j = max_j;
		*d_max_ie = max_ie;
		*d_gscore = gscore;
		*d_max_off = max_off;
	}
}

int main(int argc, char *argv[])
{
	int tlen = TLEN;
	int qlen = QLEN;
	int num = NUM;
	int i, j, k, l, c;
	int8_t mat[MATH_SIZE * MATH_SIZE];
	uint8_t *query, *target;

	while ((c = getopt(argc, argv, "t:q:n:")) >= 0) {
			if (c == 't') tlen = atoi(optarg);
			else if (c == 'q') qlen = atoi(optarg);
			else if (c == 'n') num = atoi(optarg);
			else return 1;
	}

	query = (uint8_t*)malloc(qlen);
	target = (uint8_t*)malloc(tlen);

	bwa_fill_scmat(A, B, mat);

	int h0 = H0;
	int w = W;

	int32_t *h;
	int8_t *qp; // query profile

	int	oe_del = O_DEL + E_DEL; // opening and ending deletion
	int	oe_ins = O_INS + E_INS; // opening and ending insertion
	int max, max_i, max_j, max_ins, max_del, max_ie, gscore, max_off;
	int passes, t_lastp; // number of passes and number of thread active in the last pass

	srand(time(0));
	for(l = 0; l < num; l++) {
		// random query and qlen
		printf("TEST %d:\nQUERY:\n", l);
		for(c = 0; c < qlen; c++) {
			query[c] = rand() % RANGE;
			printf("%d, ", query[c]);
		}
		printf("\nTARGET:\n");
		for(c = 0; c < tlen; c++) {
			target[c] = rand() % RANGE;
			printf("%d, ", target[c]);
		}

		// allocate memory
		qp = (int8_t*)malloc(qlen * MATH_SIZE);
		h = (int32_t*)calloc(qlen + 1, sizeof(int32_t));

		// generate the query profile
		for (k = i = 0; k < MATH_SIZE; ++k) {
			const int8_t *p = &mat[k * MATH_SIZE];
			for (j = 0; j < qlen; ++j) {
				qp[i++] = p[query[j]];
			}
		}
		// fill the first row
		h[0] = h0; h[1] = h0 > oe_ins? h0 - oe_ins : 0;
		for (j = 2; j <= qlen && h[j-1] > E_INS; ++j) {
			h[j] = h[j - 1] - E_INS;
		}
		// adjust $w if it is too large
		k = MATH_SIZE * MATH_SIZE;
		for (i = 0, max = 0; i < k; ++i) // get the max score
			max = max > mat[i]? max : mat[i];
		max_ins = (int)((double)(qlen * max + END_BONUS - O_INS) / E_INS + 1.);
		max_ins = max_ins > 1? max_ins : 1;
		w = w < max_ins? w : max_ins;
		max_del = (int)((double)(qlen * max + END_BONUS - O_DEL) / E_DEL + 1.);
		max_del = max_del > 1? max_del : 1;
		w = w < max_del? w : max_del;
		max = h0, max_i = max_j = -1; max_ie = -1, gscore = -1; max_off = 0;

		// Initialize
		// memset: max, max_j, max_i, max_ie, gscore, max_off -> GPU
		// kernel parameters:
		// value: w, oe_ins, e_ins, o_del, e_del, oe_del, tlen, qlen, passes, t_lastp, h0, zdrop
		// memcpy: e[...], h[...], qp[...], target[...]
		int *d_max, *d_max_j, *d_max_i, *d_max_ie, *d_gscore, *d_max_off;
		int32_t *d_h;
		int8_t *d_qp;
		uint8_t *d_target;

		passes = (int)((double)tlen / (double)WARP + 1.);
		t_lastp = tlen - (tlen / WARP) * WARP;

		// gpuErrchk(cudaDeviceSetLimit(cudaLimitMallocHeapSize, FIXED_HEAP * ONE_MBYTE));
		// Allocate device memory
		gpuErrchk(cudaMalloc(&d_max, sizeof(int)));
		gpuErrchk(cudaMalloc(&d_max_j, sizeof(int)));
		gpuErrchk(cudaMalloc(&d_max_i, sizeof(int)));
		gpuErrchk(cudaMalloc(&d_max_ie, sizeof(int)));
		gpuErrchk(cudaMalloc(&d_gscore, sizeof(int)));
		gpuErrchk(cudaMalloc(&d_max_off, sizeof(int)));

		gpuErrchk(cudaMalloc(&d_h, sizeof(int32_t) * (qlen + 1)));
		gpuErrchk(cudaMalloc(&d_qp, sizeof(int8_t) * qlen * MATH_SIZE));
		gpuErrchk(cudaMalloc(&d_target, sizeof(uint8_t) * tlen));
		// Transfer data to GPU
		gpuErrchk(cudaMemcpy(d_h, h, sizeof(int32_t) * (qlen + 1), cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(d_qp, qp, sizeof(int8_t) * qlen * MATH_SIZE, cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(d_target, target, sizeof(uint8_t) * tlen, cudaMemcpyHostToDevice));
		// The kernel

		sw_kernel<<<1, WARP, 2 * (qlen + 1) * sizeof(int32_t) + qlen * MATH_SIZE * sizeof(int8_t)>>>\
				(d_max, d_max_j, d_max_i, d_max_ie, d_gscore, d_max_off, \
				w, oe_ins, E_INS, O_DEL, E_DEL, oe_del, MATH_SIZE, \
				tlen, qlen, passes, t_lastp, h0, ZDROP, \
				d_h, d_qp, d_target);

		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());

		// Deallocate host variables
		free(h); free(qp);
		// Get the result back from kernel
		gpuErrchk(cudaMemcpy(&max, d_max, sizeof(int), cudaMemcpyDeviceToHost));
		gpuErrchk(cudaMemcpy(&max_i, d_max_i, sizeof(int), cudaMemcpyDeviceToHost));
		gpuErrchk(cudaMemcpy(&max_j, d_max_j, sizeof(int), cudaMemcpyDeviceToHost));
		gpuErrchk(cudaMemcpy(&max_ie, d_max_ie, sizeof(int), cudaMemcpyDeviceToHost));
		gpuErrchk(cudaMemcpy(&gscore, d_gscore, sizeof(int), cudaMemcpyDeviceToHost));
		gpuErrchk(cudaMemcpy(&max_off, d_max_off, sizeof(int), cudaMemcpyDeviceToHost));

		// Deallocate CUDA variables
		gpuErrchk(cudaFree(d_max_j));
		gpuErrchk(cudaFree(d_max_i));
		gpuErrchk(cudaFree(d_max_ie));
		gpuErrchk(cudaFree(d_gscore));
		gpuErrchk(cudaFree(d_max_off));
		gpuErrchk(cudaFree(d_max));
		gpuErrchk(cudaFree(d_h));
		gpuErrchk(cudaFree(d_qp));
		gpuErrchk(cudaFree(d_target));

		int cmax, cmax_i, cmax_j, cmax_ie, cgscore, cmax_off;
		cmax = ksw_extend2(&cmax_i, &cmax_j, &cmax_ie, &cgscore, &cmax_off, \
				qlen, &query[0], tlen, &target[0], MATH_SIZE, mat, \
				O_DEL, E_DEL, O_INS, E_INS, W, END_BONUS, ZDROP, H0);

		if(cmax == max && cmax_i == max_i && cmax_j == max_j && \
				cmax_ie == max_ie && gscore == cgscore && max_off == cmax_off)
			printf("\nTEST %d PASSED.\n", l);
		else printf("\nTEST %d FAILED.\n", l);
	}

	return 0;
}

/********************
 *** SW extension ***
 ********************/
int ksw_extend2(int *cmax_i, int *cmax_j, int *cmax_ie, int *cgscore, int *cmax_off, \
		int qlen, const uint8_t *query, int tlen, const uint8_t *target, int m, const int8_t *mat, \
		int o_del, int e_del, int o_ins, int e_ins, int w, int end_bonus, int zdrop, int h0)
{
	eh_t *eh; // score array
	int8_t *qp; // query profile
	int i, j, k, \
		oe_del = o_del + e_del, \
		oe_ins = o_ins + e_ins, \
		beg, end, max, max_i, max_j, max_ins, max_del, max_ie, gscore, max_off;
	assert(h0 > 0);
	// allocate memory
	qp = (int8_t*)malloc(qlen * m);
	eh = (eh_t*)calloc(qlen + 1, 8);
	// generate the query profile
	for (k = i = 0; k < m; ++k) {
		const int8_t *p = &mat[k * m];
		for (j = 0; j < qlen; ++j) qp[i++] = p[query[j]];
	}
	// fill the first row
	eh[0].h = h0; eh[1].h = h0 > oe_ins? h0 - oe_ins : 0;
	for (j = 2; j <= qlen && eh[j-1].h > e_ins; ++j)
		eh[j].h = eh[j-1].h - e_ins;
	// adjust $w if it is too large
	k = m * m;
	for (i = 0, max = 0; i < k; ++i) // get the max score
		max = max > mat[i]? max : mat[i];
	max_ins = (int)((double)(qlen * max + end_bonus - o_ins) / e_ins + 1.);
	max_ins = max_ins > 1? max_ins : 1;
	w = w < max_ins? w : max_ins;
	max_del = (int)((double)(qlen * max + end_bonus - o_del) / e_del + 1.);
	max_del = max_del > 1? max_del : 1;
	w = w < max_del? w : max_del;
	max = h0, max_i = max_j = -1; max_ie = -1, gscore = -1;
	max_off = 0;
	beg = 0, end = qlen;
	for (i = 0; LIKELY(i < tlen); ++i) {
		int t, f = 0, h1, m = 0, mj = -1;
		int8_t *q = &qp[target[i] * qlen];
		// apply the band and the constraint (if provided)
		if (beg < i - w) beg = i - w;
		if (end > i + w + 1) end = i + w + 1;
		if (end > qlen) end = qlen;
		// compute the first column
		if (beg == 0) {
			h1 = h0 - (o_del + e_del * (i + 1));
			if (h1 < 0) h1 = 0;
		} else h1 = 0;
		for (j = beg; LIKELY(j < end); ++j) {
			// At the beginning of the loop: eh[j] = { H(i-1,j-1), E(i,j) }, f = F(i,j) and h1 = H(i,j-1)
			// Similar to SSE2-SW, cells are computed in the following order:
			//   H(i,j)   = max{H(i-1,j-1)+S(i,j), E(i,j), F(i,j)}
			//   E(i+1,j) = max{H(i,j)-gapo, E(i,j)} - gape
			//   F(i,j+1) = max{H(i,j)-gapo, F(i,j)} - gape
			eh_t *p = &eh[j];
			int h, M = p->h, e = p->e; // get H(i-1,j-1) and E(i-1,j)
			p->h = h1;          // set H(i,j-1) for the next row
			M = M? M + q[j] : 0;// separating H and M to disallow a cigar like "100M3I3D20M"
			h = M > e? M : e;   // e and f are guaranteed to be non-negative, so h>=0 even if M<0
			h = h > f? h : f;
			h1 = h;             // save H(i,j) to h1 for the next column
			mj = m > h? mj : j; // record the position where max score is achieved
			m = m > h? m : h;   // m is stored at eh[mj+1]
			t = M - oe_del;
			t = t > 0? t : 0;
			e -= e_del;
			e = e > t? e : t;   // computed E(i+1,j)
			p->e = e;           // save E(i+1,j) for the next row
			t = M - oe_ins;
			t = t > 0? t : 0;
			f -= e_ins;
			f = f > t? f : t;   // computed F(i,j+1)
		}
		eh[end].h = h1; eh[end].e = 0;
		if (j == qlen) {
			max_ie = gscore > h1? max_ie : i;
			gscore = gscore > h1? gscore : h1;

		}

		if (m == 0) break;
		if (m > max) {
			max = m, max_i = i, max_j = mj;
			max_off = max_off > abs(mj - i)? max_off : abs(mj - i);
		} else if (zdrop > 0) {
			if (i - max_i > mj - max_j) {
				if (max - m - ((i - max_i) - (mj - max_j)) * e_del > zdrop) break;
			} else {
				if (max - m - ((mj - max_j) - (i - max_i)) * e_ins > zdrop) break;
			}
		}
		// update beg and end for the next round
		for (j = beg; LIKELY(j < end) && eh[j].h == 0 && eh[j].e == 0; ++j);
		beg = j;
		for (j = end; LIKELY(j >= beg) && eh[j].h == 0 && eh[j].e == 0; --j);
		end = j + 2 < qlen? j + 2 : qlen;
		//beg = 0; end = qlen; // uncomment this line for debugging
	}
	free(eh); free(qp);

	*cmax_i = max_i;
	*cmax_j = max_j;
	*cmax_ie = max_ie;
	*cgscore = gscore;
	*cmax_off = max_off;
	return max;
}

/*****************
 * CIGAR related *
 *****************/

void bwa_fill_scmat(int a, int b, int8_t mat[25])
{
	int i, j, k;
	for (i = k = 0; i < 4; ++i) {
		for (j = 0; j < 4; ++j)
			mat[k++] = i == j? a : -b;
		mat[k++] = -1; // ambiguous base
	}
	for (j = 0; j < 5; ++j) mat[k++] = -1;
}

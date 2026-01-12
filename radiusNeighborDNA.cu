#include "cuda_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#define BLOCKSIZE 32
#define MAXNEIGHBOR (1024*256)
#define WIDTH 32
#define LOG2WIDTH 5
#define BASEPERINT 4

typedef unsigned int u32;

__global__ void findNeighborKernel(u32* qry,u32* const mat,
									const u32 ntip,const u32 nbin,const u32 pitch,
                                   	u32* const nNeighbor,u32* const isNeighbor, const u32 radius) {
	const u32 x_local=threadIdx.x;
	const u32 stride=blockDim.x;
	const u32 y_local=threadIdx.y;
	const u32 ybid=blockIdx.y;
	const u32 y_tid=threadIdx.y+blockIdx.y*blockDim.y;
	u32 commonCount=0;
	if(y_tid<ntip){
		for(u32 i=x_local; i<nbin; i+=stride) {
			u32* row_ptr=(u32*)((char*)(mat)+y_tid*pitch);
			u32 value=row_ptr[i]&qry[i];
			for (u32 j = 0; j < 4; ++j) {
				unsigned char nuc = (value >> (j * 8)) & 0x0F;
				if (nuc != 0x00 ) {
					commonCount ++;
				}
			}
		}
	}

    for (u32 offset = 16; offset > 0; offset>>=1) {
        commonCount += __shfl_down_sync(0xFFFFFFFF, commonCount, offset);
    }

	if(x_local==0&&y_tid<ntip){
		if(commonCount>=radius){
			atomicAdd(nNeighbor,1);
//			u32 xx=y_tid>>5;
//			u32 yy=y_tid&31;
			atomicOr(isNeighbor+ybid,1u<<y_local);
		}
	}
}


__host__ void int2idx(u32* const ve,u32 const size,u32* const output,u32* nIdx) {
	u32 k=0;
	for (u32 i = 0; i < size; ++i) {
		u32 value =ve[i];
		u32 base = i * 32;
		while (value != 0) {
			u32 lowest = value & -value;
			int j = 0;
			u32 temp = lowest;
			while (temp > 1) {
				temp >>= 1;
				j++;
			}
			output[k]=base + j;
			k++;
			value &= value - 1;
		}
	}
	*nIdx=k;
}

int main(int argc, char *argv[]) {
	setbuf(stdout, NULL);
	clock_t start = clock();

	if(argc < 3) {
		printf("Error: no sufficient args.\n");
		return 0;
	}

	u32 radius = atoi(argv[2]);

	char input_file[256];
	strcpy(input_file, argv[1]);
	strcat(input_file, ".tbmat");

	u32 dim[2];
	FILE *file = fopen(input_file, "rb");
	if (file == NULL) {
		printf("No bmat input.\n");
		return 0;
	}
	fread(dim, sizeof(u32), 2, file);
	const u32 ntip = dim[0], ncol = dim[1];
	const u32 nsite = ncol * BASEPERINT;

	u32* mat = (u32*)(malloc(sizeof(u32) * ntip * ncol));
	size_t pitch;
	u32* mat_gpu;
	cudaMallocPitch((void**)&mat_gpu, &pitch, ncol * sizeof(u32), ntip);
	fread(mat, sizeof(u32), ntip * ncol, file);
	cudaMemcpy2D(mat_gpu, pitch, 
	             mat, ncol * sizeof(u32),
	             ncol * sizeof(u32), ntip,
	             cudaMemcpyHostToDevice);
	free(mat);
	fclose(file);

	u32* nNeighbor_gpu;
	cudaMalloc((void**)&nNeighbor_gpu, sizeof(u32)*ntip);

	u32* isNeighbor_gpu;
	u32 nbin = (ntip - 1) / WIDTH + 1;
	cudaMalloc((void**)&isNeighbor_gpu, sizeof(u32)*nbin);
	
	u32* nNeighbor;
	cudaHostAlloc((void**)&nNeighbor, sizeof(u32)*ntip, cudaHostAllocDefault);

	u32* isNeighbor;
	cudaHostAlloc((void**)&isNeighbor, sizeof(u32)*nbin, cudaHostAllocDefault);

	cudaMemset(nNeighbor_gpu, 0, sizeof(u32)*ntip);

	u32* idx = (u32*)(malloc(sizeof(u32)*MAXNEIGHBOR));
	u32 nIdx;

	char output_file[256];
	strcpy(output_file, argv[1]);
	strcat(output_file, ".bk");
	FILE *output = fopen(output_file, "wb");

	strcpy(output_file, argv[1]);
	strcat(output_file, ".idx");
	FILE *output_idx = fopen(output_file, "wb");

	cudaStream_t s_run;
	cudaStreamCreate(&s_run);

	const dim3 threads_square(BLOCKSIZE,BLOCKSIZE);
	const dim3 blocks_square(1,(ntip-1)/BLOCKSIZE+1);
	cudaMemset(isNeighbor_gpu, 0, sizeof(u32)*nbin);
	findNeighborKernel<<<blocks_square, threads_square, 0, s_run>>>(
	    mat_gpu,
	    mat_gpu, ntip, ncol,pitch,
	    nNeighbor_gpu, isNeighbor_gpu, radius
	);
	cudaStreamSynchronize(s_run);
	cudaMemcpy(isNeighbor, isNeighbor_gpu, sizeof(u32)*nbin, cudaMemcpyDeviceToHost);

	for(u32 i = 1; i < ntip; i++) {
		if((i & 1023) == 1023) {
			printf("=");
		}
		if((i & 32767) == 32767) {
			printf("\n");
		}
		cudaMemset(isNeighbor_gpu, 0, sizeof(u32)*nbin);
		findNeighborKernel<<<blocks_square, threads_square, 0, s_run>>>(
			(u32*)((char*)(mat_gpu)+i*pitch),
//		    mat_gpu + i*ncol,
		    mat_gpu, ntip, ncol,pitch,
		    nNeighbor_gpu + i, isNeighbor_gpu, radius
		);

		fwrite(isNeighbor, sizeof(u32), nbin, output);
		int2idx(isNeighbor, nbin, idx, &nIdx);
		fwrite(idx, sizeof(u32), nIdx, output_idx);

		cudaStreamSynchronize(s_run);
		cudaMemcpy(isNeighbor, isNeighbor_gpu, sizeof(u32)*nbin, cudaMemcpyDeviceToHost);
	}

	printf("\n");

	fwrite(isNeighbor, sizeof(u32), nbin, output);
	int2idx(isNeighbor, nbin, idx, &nIdx);
	fwrite(idx, sizeof(u32), nIdx, output_idx);

	fclose(output);
	fclose(output_idx);

	cudaStreamDestroy(s_run);

	cudaMemcpy(nNeighbor, nNeighbor_gpu, sizeof(u32)*ntip, cudaMemcpyDeviceToHost);

	strcpy(output_file, argv[1]);
	strcat(output_file, ".nNeighbor");
	FILE *output_nNeighbor = fopen(output_file, "wb");
	dim[0] = 1;
	dim[1] = ntip;
	fwrite(dim, sizeof(u32), 2, output_nNeighbor);
	fwrite(nNeighbor, sizeof(u32), ntip, output_nNeighbor);
	fclose(output_nNeighbor);

	cudaFree(mat_gpu);
	cudaFree(nNeighbor_gpu);
	cudaFree(isNeighbor_gpu);
	cudaFreeHost(nNeighbor);
	cudaFreeHost(isNeighbor);
	free(idx);

	clock_t end = clock();
	double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
	printf("radiusNeighborDNA: %.2fs elapsed for %u tips.\n", time_taken, ntip);

	return 0;
}

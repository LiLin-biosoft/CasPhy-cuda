#include "cuda_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#define BLOCKSIZE 32
#define MAXNEIGHBOR (1024*256)
#define WIDTH 32
#define LOG2WIDTH 5
#define BASEPERINT 4
#define ROWCACHE 16384

typedef unsigned int u32;

__global__ void packMatKernel(u32* const matRaw,
                             const u32 nrow,const u32 nbinRaw,const u32 pitchRaw,
                             u32* const mat,const u32 nbin,const u32 pitch) {
	const u32 xlocal=threadIdx.x;
	const u32 rowId=blockIdx.x;
	const u32 stride=blockDim.x;

	u32* rowRaw_ptr=(u32*)((char*)(matRaw)+rowId*pitchRaw);
	u32* row_ptr=(u32*)((char*)(mat)+rowId*pitch);
	for(u32 i=xlocal; (i>>1)<nbin; i+=stride) {
		u32 value=0;
		if(i<nbinRaw) {
			u32 valueRaw=rowRaw_ptr[i];
			for (u32 j = 0; j < 4; ++j) {
				unsigned char nuc = (valueRaw >> (j * 8)) & 0x0F;
				value|=(nuc<<(j*4));
			}
		}
		u32 mask=__activemask();
		u32 value2=__shfl_down_sync(mask,value,1);
		u32 valuePacked=value|(value2<<16);
		if((xlocal&1)==0) {
			row_ptr[i>>1]=valuePacked;
		}
	}

}

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
			for (u32 j = 0; j < 8; ++j) {
				unsigned char nuc = (value >> (j * 4)) & 0x0F;
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

//	u32 dim[2];
//	FILE *file = fopen(input_file, "rb");
//	if (file == NULL) {
//		printf("No bmat input.\n");
//		return 0;
//	}
//	fread(dim, sizeof(u32), 2, file);
//	const u32 ntip = dim[0], ncol = dim[1];
//	const u32 nsite = ncol * BASEPERINT;
//
//	u32* mat = (u32*)(malloc(sizeof(u32) * ntip * ncol));
//	size_t pitch;
//	u32* mat_gpu;
//	cudaMallocPitch((void**)&mat_gpu, &pitch, ncol * sizeof(u32), ntip);
//	fread(mat, sizeof(u32), ntip * ncol, file);
//	cudaMemcpy2D(mat_gpu, pitch, 
//	             mat, ncol * sizeof(u32),
//	             ncol * sizeof(u32), ntip,
//	             cudaMemcpyHostToDevice);
//	free(mat);
//	fclose(file);

////////////////////////

	u32 dim[2];
	FILE *file = fopen(input_file, "rb");
	if (file == NULL) {
		printf("No bmat input.\n");
		return 0;
	}
	fread(dim, sizeof(u32), 2, file);
	const u32 ntip = dim[0], nbinRaw = dim[1];

	u32* matRaw = (u32*)(malloc(sizeof(u32)*nbinRaw*ROWCACHE));
	u32* matRaw_gpu;
	size_t pitchRaw;
	cudaMallocPitch((void**)&matRaw_gpu, &pitchRaw, nbinRaw*sizeof(u32),ROWCACHE);
	u32* mat_gpu;
	u32 ncol=(nbinRaw-1)/2+1;
	size_t pitch;
	cudaMallocPitch((void**)&mat_gpu, &pitch, ncol*sizeof(u32),ntip);
	
	for(u32 i=0;i<ntip;i+=ROWCACHE){
		u32 nCache=ROWCACHE;
		if(ntip-i<nCache){
			nCache=ntip-i;
		}
		fread(matRaw, sizeof(u32), nbinRaw*nCache, file);
		cudaMemcpy2D(matRaw_gpu, pitchRaw, 
		             matRaw, nbinRaw * sizeof(u32),
		             nbinRaw * sizeof(u32), nCache,
		             cudaMemcpyHostToDevice);
		packMatKernel<<<nCache,32>>>(matRaw_gpu,nCache,nbinRaw,pitchRaw,
		                        (u32*)((char*)(mat_gpu)+i*pitch),ncol,pitch);
	}
	fclose(file);
	cudaFree(matRaw_gpu);
	free(matRaw);

//	u32* mat=(u32*)(malloc(sizeof(u32)*ntip*ncol));
//	cudaMemcpy2D(mat, ncol*sizeof(u32), 
//	    mat_gpu, pitch,
//	    ncol * sizeof(u32), ntip,
//	    cudaMemcpyDeviceToHost);

//	char output_file[256];
//	strcpy(output_file,argv[1]);
//	strcat(output_file,".packed");
//	FILE *output=fopen(output_file,"wb");
//	dim[0]=ntip,dim[1]=ncol;
//	fwrite(dim,sizeof(u32),2,output);
//	fwrite(mat,sizeof(u32),ntip*ncol,output);
//	fclose(output);


///////////////////////// 

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

	u32** idxCache=(u32**)(malloc(sizeof(u32*)*ntip));
	for(u32 i=0;i<ntip;i++){
		idxCache[i]=(u32*)(malloc(sizeof(u32)*MAXNEIGHBOR));
	}
	u32* nIdxCache=(u32*)(malloc(sizeof(u32)*ntip));
	memset(nIdxCache, 0, sizeof(u32)*ntip);

	for(u32 i = 1; i < ntip; i++) {
		if((i & 1023) == 1023) {
			printf("=");
		}
		if((i & 32767) == 32767) {
			printf("\n");
		}
		
		u32 binOffset=i>>5;
		u32 rowOffset=binOffset<<5;
		
		cudaMemset(isNeighbor_gpu, 0, sizeof(u32)*nbin);
		findNeighborKernel<<<blocks_square, threads_square, 0, s_run>>>(
			(u32*)((char*)(mat_gpu)+i*pitch),
		    (u32*)((char*)(mat_gpu)+rowOffset*pitch), ntip-rowOffset, ncol,pitch,
		    nNeighbor_gpu + i, isNeighbor_gpu+binOffset, radius
		);

		binOffset=(i-1)>>5;
		rowOffset=binOffset<<5;
		u32 selfIdx=i-1;
		for(u32 j=0;j<nIdxCache[selfIdx];j++){
			u32 tmpIdx=idxCache[selfIdx][j];
			u32 xx=tmpIdx>>5;
			u32 yy=tmpIdx&31;
			isNeighbor[xx]|=1<<yy;
		}
		fwrite(isNeighbor, sizeof(u32), nbin, output);
		int2idx(isNeighbor+binOffset, nbin-binOffset, idx, &nIdx);
		for(u32 j=0;j<nIdx;j++){
			u32 neighborIdx=idx[j]+rowOffset;
			if((selfIdx>>5)<(neighborIdx>>5)){
				idxCache[neighborIdx][nIdxCache[neighborIdx]]=selfIdx;
				nIdxCache[neighborIdx]++;	
			}
			idxCache[selfIdx][nIdxCache[selfIdx]]=neighborIdx;
			nIdxCache[selfIdx]++;
		}
		fwrite(idxCache[selfIdx], sizeof(u32), nIdxCache[selfIdx], output_idx);
		free(idxCache[selfIdx]);

		cudaStreamSynchronize(s_run);
		cudaMemcpy(isNeighbor, isNeighbor_gpu, sizeof(u32)*nbin, cudaMemcpyDeviceToHost);
	}

	printf("\n");

	u32 binOffset=(ntip-1)>>5;
	u32 rowOffset=binOffset<<5;
	u32 selfIdx=ntip-1;
	for(u32 j=0;j<nIdxCache[selfIdx];j++){
		u32 tmpIdx=idxCache[selfIdx][j];
		u32 xx=tmpIdx>>5;
		u32 yy=tmpIdx&31;
		isNeighbor[xx]|=1<<yy;
	}
	fwrite(isNeighbor, sizeof(u32), nbin, output);
	int2idx(isNeighbor+binOffset, nbin-binOffset, idx, &nIdx);
	for(u32 j=0;j<nIdx;j++){
		u32 neighborIdx=idx[j]+rowOffset;
		idxCache[selfIdx][nIdxCache[selfIdx]]=neighborIdx;
		nIdxCache[selfIdx]++;
	}
	fwrite(idxCache[selfIdx], sizeof(u32), nIdxCache[selfIdx], output_idx);
	free(idxCache[selfIdx]);

	free(idxCache);
	fclose(output);
	fclose(output_idx);

	cudaStreamDestroy(s_run);

//	cudaMemcpy(nNeighbor, nNeighbor_gpu, sizeof(u32)*ntip, cudaMemcpyDeviceToHost);

	strcpy(output_file, argv[1]);
	strcat(output_file, ".nNeighbor");
	FILE *output_nNeighbor = fopen(output_file, "wb");
	dim[0] = 1;
	dim[1] = ntip;
	fwrite(dim, sizeof(u32), 2, output_nNeighbor);
	fwrite(nIdxCache, sizeof(u32), ntip, output_nNeighbor);
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

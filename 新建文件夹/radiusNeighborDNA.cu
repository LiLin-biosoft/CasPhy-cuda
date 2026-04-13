#include "cuda_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#define BLOCKSIZE 32
#define MAXNEIGHBOR (1024*256)
#define NRESERVE 1024
#define WIDTH 32
#define LOG2WIDTH 5
#define BASEPERINT 4
#define ROWCACHE 8192

#define CUDACALL(call) {                                    \
    cudaError_t err = call;                                         \
    if (err != cudaSuccess) {                                       \
        printf("CUDA error at %s:%d code=%d(%s) \"%s\"\n",          \
               __FILE__, __LINE__, err, cudaGetErrorString(err), #call); \
        exit(1);                                                    \
    }                                                               \
}

typedef unsigned int u32;

//radiusNeighborDNAmulti

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
                                   	u32* const isNeighbor, const u32 radius,
									const u32 bidOffset) {
	const u32 x_local=threadIdx.x;
	const u32 stride=blockDim.x;
	const u32 y_local=threadIdx.y;
	const u32 ybid=blockIdx.y+bidOffset;
	const u32 y_tid=threadIdx.y+ybid*blockDim.y;
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

	int gpuCount;
	CUDACALL(cudaGetDeviceCount(&gpuCount));
	printf("%u devices detected\n",gpuCount);

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
	const u32 ntip = dim[0], nbinRaw = dim[1];

	u32 ncol=(nbinRaw-1)/2+1;
	size_t* pitch=(size_t*)(malloc(sizeof(size_t)*gpuCount));
	u32** mat_gpu=(u32**)(malloc(sizeof(u32*)*gpuCount));
	for(u32 gpuIdx=0;gpuIdx<gpuCount;++gpuIdx){
		CUDACALL(cudaSetDevice(gpuIdx));
		CUDACALL(cudaMallocPitch((void**)&(mat_gpu[gpuIdx]), &(pitch[gpuIdx]), ncol*sizeof(u32),ntip));
	}
	
	CUDACALL(cudaSetDevice(0));
	u32* matRaw = (u32*)(malloc(sizeof(u32)*nbinRaw*ROWCACHE));
	u32* matRaw_gpu;
	size_t pitchRaw;
	CUDACALL(cudaMallocPitch((void**)&matRaw_gpu, &pitchRaw, nbinRaw*sizeof(u32),ROWCACHE));
	
	for(u32 i=0;i<ntip;i+=ROWCACHE){
		u32 nCache=ROWCACHE;
		if(ntip-i<nCache){
			nCache=ntip-i;
		}
		fread(matRaw, sizeof(u32), nbinRaw*nCache, file);
		CUDACALL(cudaMemcpy2D(matRaw_gpu, pitchRaw, 
		             matRaw, nbinRaw * sizeof(u32),
		             nbinRaw * sizeof(u32), nCache,
		             cudaMemcpyHostToDevice));
		packMatKernel<<<nCache,32>>>(matRaw_gpu,nCache,nbinRaw,pitchRaw,
		                        (u32*)((char*)(mat_gpu[0])+i*pitch[0]),ncol,pitch[0]);
	}
	fclose(file);
	cudaFree(matRaw_gpu);
	free(matRaw);

	u32 nbin = (ntip - 1) / WIDTH + 1;
	u32** isNeighbor_gpu=(u32**)(malloc(sizeof(u32*)*gpuCount));
	for(u32 i=0;i<gpuCount;++i){
		CUDACALL(cudaSetDevice(i));
		CUDACALL(cudaMalloc((void**)&(isNeighbor_gpu[i]), sizeof(u32)*nbin));
	}

	u32* isNeighbor;
	CUDACALL(cudaHostAlloc((void**)&isNeighbor, sizeof(u32)*nbin, cudaHostAllocDefault));

	u32* idx = (u32*)(malloc(sizeof(u32)*MAXNEIGHBOR));
	u32 nIdx;

	char output_file[256];
	strcpy(output_file, argv[1]);
	strcat(output_file, ".idx");
	FILE *output_idx = fopen(output_file, "wb");

	cudaStream_t* s_run=(cudaStream_t*)(malloc(sizeof(cudaStream_t)*gpuCount));
	for(u32 i=0;i<gpuCount;++i){
		CUDACALL(cudaSetDevice(i));
		CUDACALL(cudaStreamCreate(&(s_run[i])));
	}
	if(gpuCount>1){
		u32* mat=(u32*)(malloc(sizeof(u32)*ncol*ROWCACHE));
		for(u32 i=0;i<ntip;i+=ROWCACHE){
			u32 nCache=ROWCACHE;
			if(ntip-i<nCache){
				nCache=ntip-i;
			}
			CUDACALL(cudaSetDevice(0));
			CUDACALL(cudaMemcpy2DAsync(
				mat,sizeof(u32)*ncol,
				(u32*)((char*)mat_gpu[0]+i*pitch[0]),pitch[0],
				sizeof(u32)*ncol,nCache,
				cudaMemcpyDeviceToHost,s_run[0]
			));
			CUDACALL(cudaStreamSynchronize(s_run[0]));
			for(u32 gpuIdx=1;gpuIdx<gpuCount;++gpuIdx){
				CUDACALL(cudaSetDevice(gpuIdx));
				CUDACALL(cudaMemcpy2DAsync(
					(u32*)((char*)mat_gpu[gpuIdx]+i*pitch[gpuIdx]),pitch[gpuIdx],
					mat,sizeof(u32)*ncol,
					sizeof(u32)*ncol,nCache,
					cudaMemcpyHostToDevice,s_run[gpuIdx]
				))		
			}
			for(u32 gpuIdx=1;gpuIdx<gpuCount;++gpuIdx){
				CUDACALL(cudaSetDevice(gpuIdx));
				CUDACALL(cudaStreamSynchronize(s_run[gpuIdx]));
			}
		}
		free(mat);
	}


	u32 nblock=(ntip-1)/BLOCKSIZE+1;
	u32* nblockGPU=(u32*)(malloc(sizeof(u32)*(gpuCount)));
	u32 nblockDivide=(nblock-1)/gpuCount+1;
	u32* bidOffset=(u32*)(malloc(sizeof(u32)*(gpuCount+1)));
	bidOffset[0]=0;
	for(u32 gpuIdx=0;gpuIdx<gpuCount;++gpuIdx){
		u32 tmp=bidOffset[gpuIdx]+nblockDivide;
		if(tmp<nblock){
			bidOffset[gpuIdx+1]=tmp;
			nblockGPU[gpuIdx]=nblockDivide;
		}else{
			bidOffset[gpuIdx+1]=nblock;
			nblockGPU[gpuIdx]=nblock-bidOffset[gpuIdx];
		}
	}

	const dim3 threads_square(BLOCKSIZE,BLOCKSIZE);
	
	for(u32 gpuIdx=0;gpuIdx<gpuCount;++gpuIdx){
		CUDACALL(cudaSetDevice(gpuIdx));
		const dim3 blocks_square(1,nblockGPU[gpuIdx]);
		CUDACALL(cudaMemsetAsync(isNeighbor_gpu[gpuIdx], 0, sizeof(u32)*nbin,s_run[gpuIdx]));
		findNeighborKernel<<<blocks_square, threads_square, 0, s_run[gpuIdx]>>>(
			mat_gpu[gpuIdx],
			mat_gpu[gpuIdx], ntip, ncol,pitch[gpuIdx],
			isNeighbor_gpu[gpuIdx], radius,bidOffset[gpuIdx]
		);
		CUDACALL(cudaMemcpyAsync(
			isNeighbor+bidOffset[gpuIdx], isNeighbor_gpu[gpuIdx]+bidOffset[gpuIdx],
			sizeof(u32)*nblockGPU[gpuIdx], cudaMemcpyDeviceToHost,s_run[gpuIdx]
		));
	}

	std::vector<std::vector<u32>> idxCache(ntip);
	for (u32 i = 0; i < ntip; ++i){
		idxCache[i].reserve(NRESERVE);
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

		nblock=(ntip-rowOffset-1)/BLOCKSIZE+1;
		nblockDivide=(nblock-1)/gpuCount+1;
		bidOffset[0]=0;
		for(u32 gpuIdx=0;gpuIdx<gpuCount;++gpuIdx){
			u32 tmp=bidOffset[gpuIdx]+nblockDivide;
			if(tmp<nblock){
				bidOffset[gpuIdx+1]=tmp;
				nblockGPU[gpuIdx]=nblockDivide;
			}else{
				bidOffset[gpuIdx+1]=nblock;
				nblockGPU[gpuIdx]=nblock-bidOffset[gpuIdx];
			}
		}
		
		for(u32 gpuIdx=0;gpuIdx<gpuCount;++gpuIdx){
			CUDACALL(cudaSetDevice(gpuIdx));
			CUDACALL(cudaStreamSynchronize(s_run[gpuIdx]));
			const dim3 blocks_square(1,nblockGPU[gpuIdx]);
			CUDACALL(cudaMemsetAsync(isNeighbor_gpu[gpuIdx], 0, sizeof(u32)*nbin,s_run[gpuIdx]));
			findNeighborKernel<<<blocks_square, threads_square, 0, s_run[gpuIdx]>>>(
				(u32*)((char*)(mat_gpu[gpuIdx])+i*pitch[gpuIdx]),
				(u32*)((char*)(mat_gpu[gpuIdx])+rowOffset*pitch[gpuIdx]), ntip-rowOffset, ncol,pitch[gpuIdx],
				isNeighbor_gpu[gpuIdx]+binOffset, radius,bidOffset[gpuIdx]
			);
		}
		
		u32 binOffset_previous=(i-1)>>5;
		u32 rowOffset_previous=binOffset_previous<<5;
		u32 selfIdx=i-1;

		int2idx(isNeighbor+binOffset_previous, nbin-binOffset_previous, idx, &nIdx);
		for(u32 j=0;j<nIdx;j++){
			u32 neighborIdx=idx[j]+rowOffset_previous;
			if((selfIdx>>5)<(neighborIdx>>5)){
				idxCache[neighborIdx].push_back(selfIdx);
				nIdxCache[neighborIdx]++;	
			}
			idxCache[selfIdx].push_back(neighborIdx);
			nIdxCache[selfIdx]++;
		}
		fwrite(idxCache[selfIdx].data(), sizeof(u32), nIdxCache[selfIdx], output_idx);
		idxCache[selfIdx].clear();
		idxCache[selfIdx].shrink_to_fit();
		for(u32 gpuIdx=0;gpuIdx<gpuCount;++gpuIdx){
			CUDACALL(cudaSetDevice(gpuIdx));
			CUDACALL(cudaStreamSynchronize(s_run[gpuIdx]));
			CUDACALL(cudaMemcpyAsync(
				isNeighbor+binOffset+bidOffset[gpuIdx], isNeighbor_gpu[gpuIdx]+binOffset+bidOffset[gpuIdx],
				sizeof(u32)*nblockGPU[gpuIdx], cudaMemcpyDeviceToHost,s_run[gpuIdx]
			));			
		}
	}

	printf("\n");

	u32 binOffset=(ntip-1)>>5;
	u32 rowOffset=binOffset<<5;
	u32 selfIdx=ntip-1;

	for(u32 gpuIdx=0;gpuIdx<gpuCount;++gpuIdx){
		CUDACALL(cudaSetDevice(gpuIdx));
		CUDACALL(cudaStreamSynchronize(s_run[gpuIdx]));
	}
	int2idx(isNeighbor+binOffset, nbin-binOffset, idx, &nIdx);
	for(u32 j=0;j<nIdx;j++){
		u32 neighborIdx=idx[j]+rowOffset;
		idxCache[selfIdx].push_back(neighborIdx);
		nIdxCache[selfIdx]++;
	}
	fwrite(idxCache[selfIdx].data(), sizeof(u32), nIdxCache[selfIdx], output_idx);
	idxCache[selfIdx].clear();
	idxCache[selfIdx].shrink_to_fit();	
	idxCache.clear();
	idxCache.shrink_to_fit();
	fclose(output_idx);

	for(u32 gpuIdx=0;gpuIdx<gpuCount;++gpuIdx){
		CUDACALL(cudaStreamDestroy(s_run[gpuIdx]));		
	}

	strcpy(output_file, argv[1]);
	strcat(output_file, ".nNeighbor");
	FILE *output_nNeighbor = fopen(output_file, "wb");
	dim[0] = 1;
	dim[1] = ntip;
	fwrite(dim, sizeof(u32), 2, output_nNeighbor);
	fwrite(nIdxCache, sizeof(u32), ntip, output_nNeighbor);
	fclose(output_nNeighbor);

	cudaFree(mat_gpu);
	for(u32 i=0;i<gpuCount;++i){
		CUDACALL(cudaFree(isNeighbor_gpu[i]));
		CUDACALL(cudaFree(mat_gpu[i]));
	}
	free(isNeighbor_gpu);
	free(mat_gpu);
	CUDACALL(cudaFreeHost(isNeighbor));
	free(idx);

	clock_t end = clock();
	double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
	printf("radiusNeighborDNA: %.2fs elapsed for %u tips.\n", time_taken, ntip);

	return 0;
}

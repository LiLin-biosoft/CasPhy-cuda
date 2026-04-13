#include "cuda_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#define BLOCKSIZE 128
#define MAXMUTATE (1024*256)
#define MAXNEIGHBOR (1024*512)
#define NRESERVE 1024
#define WIDTH 32
#define LOG2WIDTH 5

#define CUDACALL(call) {                                    \
    cudaError_t err = call;                                         \
    if (err != cudaSuccess) {                                       \
        printf("CUDA error at %s:%d code=%d(%s) \"%s\"\n",          \
               __FILE__, __LINE__, err, cudaGetErrorString(err), #call); \
        exit(1);                                                    \
    }                                                               \
}

typedef unsigned int u32;

//radiusNeighborMulti

__global__ void findNeighborKernel(u32* const mat,u32 const nrow,u32 const ntip,
							u32* const isNeighbor, float radius,
							u32 const nMutate,u32* const mutateIdx,
							float* const weightRun,const u32 binOffset){
	const u32 tid=threadIdx.x+blockIdx.x*blockDim.x+binOffset*WIDTH;
	float commonCount=0;
	if(tid<ntip){
		for(u32 i=0;i<nMutate;i++){
			u32 tmpIdx=mutateIdx[i];
			u32 xx=tmpIdx>>LOG2WIDTH;
			u32 yy=tmpIdx&(WIDTH-1);
			u32 tmp=mat[tid+xx*ntip];
			if(((tmp>>yy)&1)==1){
				commonCount+=weightRun[i];
			}
		}
	}
    u32 active = __activemask();
    u32 mask = __ballot_sync(active, commonCount >= radius);
	if((tid&31)==0){
		if ((tid >> 5) < (ntip + 31) / 32) {
			isNeighbor[tid>>5]=mask;
		}
	}
}

__global__ void int2idxKernel(u32* const mat,const u32 nbin,
				u32* idx, u32* nIdx,u32 binOffset){
	
	__shared__ u32 countLocal[BLOCKSIZE];
	__shared__ u32 offsetLocal[BLOCKSIZE+1];
	u32 idxLocal[32];
	const u32 tid=threadIdx.x;
	if(tid==0){
		offsetLocal[BLOCKSIZE]=0;
	}

	for(u32 i=0;i<nbin;i+=BLOCKSIZE){
		u32 nCount=0;
		u32 valIdx=i+tid+binOffset;
		if(valIdx<nbin){
			u32 val=mat[valIdx];
			while (val != 0) {
				int j = __ffs(val) - 1;
				idxLocal[nCount] =j+valIdx*32;
				nCount++;
				val &= val - 1;
			}
			
		}
		countLocal[tid]=nCount;
		__syncthreads();
		
		for(u32 j=1;j<BLOCKSIZE;j<<=1){
			if((tid&j)>0){
				u32 mask=~(j-1);
				countLocal[tid]+=countLocal[(tid&mask)-1];
			}
			__syncthreads();
		}

		if(tid==0){
			offsetLocal[0]=offsetLocal[BLOCKSIZE];
		}
		__syncthreads();
		offsetLocal[tid+1]=offsetLocal[0]+countLocal[tid];
		__syncthreads();
		for(u32 j=0;j<nCount;++j){
			idx[offsetLocal[tid]+j]=idxLocal[j];
		}
	}
	__syncthreads();
	if(tid==0){
		nIdx[0]=offsetLocal[BLOCKSIZE];
	}
}

__host__ void int2idx(u32* const ve,u32 const size,u32* const output,u32* nIdx){
	u32 k=0;
    for (u32 i = 0; i < size; i++) {
        u32 val = ve[i];
//        while (val) {
//        	unsigned long j;
//            _BitScanForward(&j,val);
////          u32 j=__builtin_ctz(val);
//            output[k]=i*32+j;
//            k++;
//            u32 t=val&-val; // obtain first 1-bit
//            val ^=t; // clean processed bit
//        }
		for(u32 j=0;j<WIDTH;j++){
			if(((val>>j)&1)==1){
				output[k]=i*WIDTH+j;
				k++;
			}
		}
    }
    *nIdx=k;
}

__host__ void int2idx2(u32* const ve,u32 const size,u32* const output,u32* nIdx) {
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

__host__ void blockDivide(const u32 nblock,const u32 gpuCount,u32* nblockGPU, u32* bidOffset){
	u32 nblockDivide=(nblock-1)/gpuCount+1;
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
}

int main(int argc, char *argv[]){
	setbuf(stdout, NULL); 
	clock_t start = clock();
	double time_cpu1=0;
	double time_cpu2=0;

	if(argc<3){
		printf("Error: no sufficient args.\n");
		return 0;
	} 

	int gpuCount;
	CUDACALL(cudaGetDeviceCount(&gpuCount));
	printf("%u devices detected\n",gpuCount);

	float radius=atoi(argv[2]);

	char input_file[256];
	strcpy(input_file,argv[1]);
	strcat(input_file,".bmat");
    u32 dim[2]; 
    FILE *file=fopen(input_file, "rb");
    if (file == NULL) {
    	printf("No bmat input.\n");
    	return 0;
  	} 
    fread(dim, sizeof(u32), 2, file);
    const u32 nrow=dim[0],ntip=dim[1];	
    const u32 nsite=nrow*WIDTH;
	u32 nbin=(ntip-1)/(WIDTH)+1;
	u32* mat=(u32*)(malloc(sizeof(u32)*nrow*ntip));
	u32* oneLine;
	CUDACALL(cudaHostAlloc(
		(void**)&oneLine,sizeof(u32)*ntip,
		cudaHostAllocDefault
	));

	cudaStream_t* s_run=(cudaStream_t*)(malloc(sizeof(cudaStream_t)*gpuCount));
	for(u32 gpuIdx=0;gpuIdx<gpuCount;++gpuIdx){
		CUDACALL(cudaSetDevice(gpuIdx));
		CUDACALL(cudaStreamCreate(&(s_run[gpuIdx])));
	}

	u32** mat_gpu=(u32**)(malloc(sizeof(u32*)*gpuCount));
	for(u32 gpuIdx=0;gpuIdx<gpuCount;++gpuIdx){
		CUDACALL(cudaSetDevice(gpuIdx));
		CUDACALL(cudaMalloc((void**)&(mat_gpu[gpuIdx]),sizeof(u32)*nrow*ntip));
	}
	
	for(u32 i=0;i<nrow;i++){
		fread(oneLine, sizeof(u32), ntip, file);
		for(u32 gpuIdx=0;gpuIdx<gpuCount;++gpuIdx){
			CUDACALL(cudaSetDevice(gpuIdx));
			CUDACALL(cudaMemcpyAsync(
				mat_gpu[gpuIdx]+i*ntip,oneLine, sizeof(u32)*ntip,
				cudaMemcpyHostToDevice,s_run[gpuIdx]
			));
		}
		memcpy(mat+i*ntip,oneLine,sizeof(u32)*ntip);
		for(u32 gpuIdx=0;gpuIdx<gpuCount;++gpuIdx){
			CUDACALL(cudaSetDevice(gpuIdx));
			CUDACALL(cudaStreamSynchronize(s_run[gpuIdx]));
		}
	}
	fclose(file);
	CUDACALL(cudaFreeHost(oneLine));
	
	strcpy(input_file,argv[1]);
	strcat(input_file,".w");
    file=fopen(input_file, "rb");
    if (file == NULL) {
    	printf("No weight input.\n");
    	return 0;
  	} 
  	
  	double* weightTmp=(double*)(malloc(sizeof(double)*nsite));
	fread(weightTmp, sizeof(double), nsite, file);  
	fclose(file);	
	float* weight=(float*)(malloc(sizeof(float)*nsite));
	for(u32 i=0;i<nsite;i++){
		weight[i]=(float)(weightTmp[i]);
	}
	free(weightTmp);

	u32** mutateIdx=(u32**)(malloc(sizeof(u32*)*gpuCount));
	float** weightRun=(float**)(malloc(sizeof(float*)*gpuCount));
	u32** idx=(u32**)(malloc(sizeof(u32*)*gpuCount));
	u32* nIdx=(u32*)(malloc(sizeof(u32)*gpuCount));
	CUDACALL(cudaHostAlloc((void**)&nIdx,sizeof(u32)*gpuCount,cudaHostAllocDefault));

	u32** isNeighbor_gpu=(u32**)(malloc(sizeof(u32*)*gpuCount));
	u32** idx_gpu=(u32**)(malloc(sizeof(u32*)*gpuCount));
	u32** nIdx_gpu=(u32**)(malloc(sizeof(u32*)*gpuCount));
	u32** mutateIdx_gpu=(u32**)(malloc(sizeof(u32*)*gpuCount));
	float** weightRun_gpu=(float**)(malloc(sizeof(float*)*gpuCount));
	for(u32 gpuIdx=0;gpuIdx<gpuCount;++gpuIdx){
		CUDACALL(cudaSetDevice(gpuIdx));
		CUDACALL(cudaHostAlloc((void**)&(mutateIdx[gpuIdx]),sizeof(u32)*MAXMUTATE,cudaHostAllocDefault));
		CUDACALL(cudaHostAlloc((void**)&(weightRun[gpuIdx]),sizeof(float)*MAXMUTATE,cudaHostAllocDefault));
		CUDACALL(cudaHostAlloc((void**)&(idx[gpuIdx]),sizeof(u32)*MAXNEIGHBOR,cudaHostAllocDefault));
		CUDACALL(cudaMalloc((void**)&(isNeighbor_gpu[gpuIdx]),sizeof(u32)*nbin));
		CUDACALL(cudaMalloc((void**)&(idx_gpu[gpuIdx]),sizeof(u32)*MAXNEIGHBOR));
		CUDACALL(cudaMalloc((void**)&(nIdx_gpu[gpuIdx]),sizeof(u32)));
		CUDACALL(cudaMalloc((void**)&(mutateIdx_gpu[gpuIdx]),sizeof(u32)*MAXMUTATE));
		CUDACALL(cudaMalloc((void**)&(weightRun_gpu[gpuIdx]),sizeof(float)*MAXMUTATE));
	}	

	u32 gpuIdx=0;
	CUDACALL(cudaSetDevice(gpuIdx));
//	first scan for non-zero sites and then calculate the distance
	u32 nMutate=0;
	for(u32 j=0;j<nrow;j++){
		u32 tmp=mat[j*ntip];
		for(u32 k=0;k<WIDTH;k++){
			if(((tmp>>k)&1)==1){
				mutateIdx[gpuIdx][nMutate]=j*WIDTH+k;
				weightRun[gpuIdx][nMutate]=weight[j*WIDTH+k];
				nMutate++;
			}
		}
	}
	
	CUDACALL(cudaMemcpyAsync(
		mutateIdx_gpu[gpuIdx],mutateIdx[gpuIdx],
		sizeof(u32)*nMutate,cudaMemcpyHostToDevice,s_run[gpuIdx]
	));
	CUDACALL(cudaMemcpyAsync(
		weightRun_gpu[gpuIdx],weightRun[gpuIdx],
		sizeof(float)*nMutate,cudaMemcpyHostToDevice,s_run[gpuIdx]
	));
	CUDACALL(cudaMemsetAsync(isNeighbor_gpu[gpuIdx], 0, sizeof(u32)*nbin,s_run[gpuIdx]));	
	findNeighborKernel<<<(ntip-1)/BLOCKSIZE+1,BLOCKSIZE,0,s_run[gpuIdx]>>>(
		mat_gpu[gpuIdx],nrow,ntip,
		isNeighbor_gpu[gpuIdx],radius,
		nMutate,mutateIdx_gpu[gpuIdx],
		weightRun_gpu[gpuIdx],0
	);
	int2idxKernel<<<1,BLOCKSIZE,0,s_run[gpuIdx]>>>(
		isNeighbor_gpu[gpuIdx],
		nbin,
		idx_gpu[gpuIdx],
		nIdx_gpu[gpuIdx],
		0
	);
	CUDACALL(cudaMemcpyAsync(
		&(nIdx[gpuIdx]),nIdx_gpu[gpuIdx],
		sizeof(u32), cudaMemcpyDeviceToHost,s_run[gpuIdx]
	));
	CUDACALL(cudaMemcpyAsync(
		idx[gpuIdx],idx_gpu[gpuIdx],
		sizeof(u32)*MAXNEIGHBOR,
		cudaMemcpyDeviceToHost,
		s_run[gpuIdx]
	));
	
	char output_file[256];
	strcpy(output_file,argv[1]);
	strcat(output_file,".idx");
	FILE *output_idx=fopen(output_file,"wb");	

	std::vector<std::vector<u32>> idxCache(ntip);
	for (u32 i = 0; i < ntip; ++i){
		idxCache[i].reserve(NRESERVE);
	}	
	
	u32* nIdxCache=(u32*)(malloc(sizeof(u32)*ntip));
	memset(nIdxCache, 0, sizeof(u32)*ntip);
	
	u32 previous_gpuIdx=gpuCount-1;
	for(u32 i=1;i<ntip;i++){
		if((i&1023)==1023){
	      printf("=");
	    }
	    if((i&32767)==32767){
	      printf("\n");
	    }

		gpuIdx=(gpuIdx+1)%gpuCount;
		CUDACALL(cudaSetDevice(gpuIdx));
		CUDACALL(cudaStreamSynchronize(s_run[gpuIdx]));
		clock_t t0=clock();
		nMutate=0;
		for(u32 j=0;j<nrow;j++){
			u32 tmp=mat[i+j*ntip];
			for(u32 k=0;k<WIDTH;k++){
				if(((tmp>>k)&1)==1){
					mutateIdx[gpuIdx][nMutate]=j*WIDTH+k;
					weightRun[gpuIdx][nMutate]=weight[j*WIDTH+k];
					nMutate++;
				}
			}
		}
		clock_t t1=clock();
		time_cpu1+=((double) (t1 - t0)) / CLOCKS_PER_SEC;

		u32 binOffset=i>>5;
		u32 rowOffset=binOffset<<5;		

		CUDACALL(cudaMemcpyAsync(
			mutateIdx_gpu[gpuIdx],mutateIdx[gpuIdx],
			sizeof(u32)*nMutate,cudaMemcpyHostToDevice,s_run[gpuIdx]
		));
		CUDACALL(cudaMemcpyAsync(
			weightRun_gpu[gpuIdx],weightRun[gpuIdx],
			sizeof(float)*nMutate,cudaMemcpyHostToDevice,s_run[gpuIdx]
		));
		CUDACALL(cudaMemsetAsync(isNeighbor_gpu[gpuIdx], 0, sizeof(u32)*nbin,s_run[gpuIdx]));
		findNeighborKernel<<<(ntip-rowOffset-1)/BLOCKSIZE+1,BLOCKSIZE,0,s_run[gpuIdx]>>>(
			mat_gpu[gpuIdx],nrow,ntip,
			isNeighbor_gpu[gpuIdx],radius,
			nMutate,mutateIdx_gpu[gpuIdx],
			weightRun_gpu[gpuIdx],binOffset
		);
		int2idxKernel<<<1,BLOCKSIZE,0,s_run[gpuIdx]>>>(
			isNeighbor_gpu[gpuIdx],
			nbin,
			idx_gpu[gpuIdx],
			nIdx_gpu[gpuIdx],
			binOffset
		);
		
		previous_gpuIdx=(previous_gpuIdx+1)%gpuCount;
		u32 selfIdx=i-1;
		CUDACALL(cudaSetDevice(previous_gpuIdx));
		CUDACALL(cudaStreamSynchronize(s_run[previous_gpuIdx]));
		t0=clock();
		for(u32 j=0;j<nIdx[previous_gpuIdx];j++){
			u32 neighborIdx=idx[previous_gpuIdx][j];
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
		t1=clock();
		time_cpu2+=((double) (t1 - t0)) / CLOCKS_PER_SEC;

		CUDACALL(cudaMemcpyAsync(
			&(nIdx[gpuIdx]),nIdx_gpu[gpuIdx],
			sizeof(u32),cudaMemcpyDeviceToHost,s_run[gpuIdx]
		));
		CUDACALL(cudaMemcpyAsync(
			idx[gpuIdx],idx_gpu[gpuIdx],
			sizeof(u32)*MAXNEIGHBOR, cudaMemcpyDeviceToHost,s_run[gpuIdx]
		));
	}
	printf("\n");

	previous_gpuIdx=(previous_gpuIdx+1)%gpuCount;
	u32 selfIdx=ntip-1;
	CUDACALL(cudaSetDevice(previous_gpuIdx));
	CUDACALL(cudaStreamSynchronize(s_run[previous_gpuIdx]));
	for(u32 j=0;j<nIdx[previous_gpuIdx];j++){
		u32 neighborIdx=idx[previous_gpuIdx][j];
		idxCache[selfIdx].push_back(neighborIdx);
		nIdxCache[selfIdx]++;
	}
	fwrite(idxCache[selfIdx].data(), sizeof(u32), nIdxCache[selfIdx], output_idx);
	idxCache[selfIdx].clear();
	idxCache[selfIdx].shrink_to_fit();	
	idxCache.clear();
	idxCache.shrink_to_fit();
	
	fclose(output_idx);
	
	strcpy(output_file,argv[1]);
	strcat(output_file,".nNeighbor");
	FILE *output_nNeighbor=fopen(output_file,"wb");
	dim[0]=1,dim[1]=ntip;
	fwrite(dim,sizeof(u32),2,output_nNeighbor);
	fwrite(nIdxCache,sizeof(u32),ntip,output_nNeighbor);	
	fclose(output_nNeighbor);
	
	free(mat);
	for(u32 gpuIdx=0;gpuIdx<gpuCount;++gpuIdx){
		CUDACALL(cudaStreamDestroy(s_run[gpuIdx]));
		CUDACALL(cudaFreeHost(idx[gpuIdx]));
		CUDACALL(cudaFreeHost(mutateIdx[gpuIdx]));
		CUDACALL(cudaFreeHost(weightRun[gpuIdx]));
		CUDACALL(cudaFree(mat_gpu[gpuIdx]));
		CUDACALL(cudaFree(isNeighbor_gpu[gpuIdx]));
		CUDACALL(cudaFree(idx_gpu[gpuIdx]));
		CUDACALL(cudaFree(nIdx_gpu[gpuIdx]));
		CUDACALL(cudaFree(mutateIdx_gpu[gpuIdx]));
		CUDACALL(cudaFree(weightRun_gpu[gpuIdx]));
	}
	free(s_run);
	CUDACALL(cudaFreeHost(nIdx));
	free(idx);
	free(mutateIdx);
	free(weightRun);
	free(mat_gpu);
	free(isNeighbor_gpu);
	free(idx_gpu);
	free(nIdx_gpu);
	free(mutateIdx_gpu);
	free(weightRun_gpu);
	
	clock_t end=clock();
	double time_taken;
	time_taken = ((double) (end - start)) / CLOCKS_PER_SEC;
	printf("radiusNeighbor: %.2fs elapsed for %u tips.\n",time_taken,ntip);
	printf("CPUtime: %.2fs,%.2fs elapsed.\n",time_cpu1,time_cpu2);
    return 0;
}

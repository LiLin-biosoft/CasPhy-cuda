#include "cuda_runtime.h"
#include <stdio.h>
#include <stdlib.h>

#define BLOCKSIZE 128
#define CACHESIZE 128
#define CROSSSIZE 8
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

__global__ void inferEdgeStateKernel(u32* const edge,
	u32* const nodeState,
	u32* const edgeState,
	const u32 nbin){

	u32 tid=threadIdx.x;
	u32 bid=blockIdx.x;
	u32 stride=blockDim.x;
	u32 ansIdx=edge[bid*2]-1;
	u32 desIdx=edge[bid*2+1]-1;
	
	for(u32 i=tid; i<nbin; i+=stride) {
		u32 value1=nodeState[ansIdx*nbin+i];
		u32 value2=nodeState[desIdx*nbin+i];
		edgeState[desIdx*nbin+i]=(value1|value2);
	}
}

__global__ void findRoot2Kernel(
		u32* const mat_x,const u32 nrow_x,
		u32* const mat_y,const u32 nrow_y,
		const u32 ncol,
		u32* const ymin){

	const u32 xtid=threadIdx.x+blockIdx.x*blockDim.x;
	const u32 ytid=threadIdx.y+blockIdx.y*blockDim.y;

	if(xtid<nrow_x&&ytid<nrow_y){
		u32 count=0;
		for(u32 i=0;i<ncol;i++){
			u32 value=mat_x[xtid*ncol+i]&mat_y[ytid*ncol+i];
		    for(int k=0;k<4;k++){
		        unsigned char nuc=(value>>(k*8))&0x0F;
		        if(nuc!=0x00){
		        	count++;
		        }
		    }
		}
		atomicMax(&(ymin[ytid]),count);			
	}
}

__global__ void findRootKernel(
		u32* const mat_x,const u32 nrow_x, 
		u32* const mat_y,const u32 nrow_y,
		const u32 ncol,
		u32* const ymin,u32* const yminIdx){
	__shared__ u32 threadSum[CROSSSIZE*CROSSSIZE];
	__shared__ u32 threadSumIdx[CROSSSIZE*CROSSSIZE];
	__shared__ u32 xCache[CROSSSIZE*CROSSSIZE];
	__shared__ u32 yCache[CROSSSIZE*CROSSSIZE];
	
	const u32 xlocal=threadIdx.x;
	const u32 ylocal=threadIdx.y;
	const u32 xtid=threadIdx.x+blockIdx.x*blockDim.x;
	const u32 ytid=threadIdx.y+blockIdx.y*blockDim.y;
	const u32 xtidread=threadIdx.y+blockIdx.x*blockDim.x;
	
	threadSum[ylocal*CROSSSIZE+xlocal]=0;
	threadSumIdx[ylocal*CROSSSIZE+xlocal]=xtid;
	
	__syncthreads();
	
	u32 count=0;
	for(u32 i=0;i<ncol;i+=CROSSSIZE){
		if(ytid<nrow_y&&i+xlocal<ncol){
			yCache[ylocal*CROSSSIZE+xlocal]=mat_y[ytid*ncol+i+xlocal];
		}
		if(xtidread<nrow_x&&i+xlocal<ncol){
			xCache[ylocal*CROSSSIZE+xlocal]=mat_x[xtidread*ncol+i+xlocal];
		}
		__syncthreads();

		for(u32 j=0;j<CROSSSIZE&&i+j<ncol;j++){
			if(xtid<nrow_x&&ytid<nrow_y){
				u32 value=xCache[xlocal*CROSSSIZE+j]&yCache[ylocal*CROSSSIZE+j];
		        for(int k=0;k<4;k++){
		          unsigned char nuc=(value>>(k*8))&0x0F;
		          if(nuc!=0x00){
		            count++;
		          }
		        }
			}			
		}
		__syncthreads();
	}
	if(xtid<nrow_x&&ytid<nrow_y){
		threadSum[ylocal*CROSSSIZE+xlocal]=count;
	}
	__syncthreads();
	
    for(u32 s=blockDim.x/2;s>0;s>>=1){
        if(xlocal<s){
            if(threadSum[ylocal*CROSSSIZE+xlocal]<threadSum[ylocal*CROSSSIZE+xlocal+s]){
				threadSum[ylocal*CROSSSIZE+xlocal]=threadSum[ylocal*CROSSSIZE+xlocal+s];
				threadSumIdx[ylocal*CROSSSIZE+xlocal]=threadSumIdx[ylocal*CROSSSIZE+xlocal+s];
            }
        }
        __syncthreads();
    }
    if(xlocal==0&&ytid<nrow_y){
    	ymin[ytid*gridDim.x+blockIdx.x]=threadSum[ylocal*CROSSSIZE];
    	yminIdx[ytid*gridDim.x+blockIdx.x]=threadSumIdx[ylocal*CROSSSIZE];
	}
}

__global__ void rowMaxKernel(u32* const data,u32* const dataIdx,
							const u32 nrow,const u32 ncol,
							u32* const value,u32* const idx){
	const u32 xlocal=threadIdx.x;
	const u32 bid=blockIdx.x;
	const u32 stride=blockDim.x;
	u32 localMax=0;
	u32 localMaxIdx=0;
	for(u32 i=xlocal;i<ncol;i+=stride){
		u32 tmp=data[bid*ncol+i];
		if(localMax<tmp){
			localMax=tmp;
			localMaxIdx=dataIdx[bid*ncol+i];
		}
	}
	
    for (u32 offset = 16; offset > 0; offset>>=1) {
    	unsigned mask = __activemask();
        u32 tmp=__shfl_down_sync(mask, localMax, offset);
        u32 tmpIdx=__shfl_down_sync(mask, localMaxIdx, offset);
        if(localMax<tmp){
        	localMax=tmp;
        	localMaxIdx=tmpIdx;
		}
    }
    if(xlocal==0){
    	value[bid]=localMax;
    	idx[bid]=localMaxIdx;
	}
}

int main(int argc, char *argv[]) {
	setbuf(stdout, NULL);
	clock_t start = clock();

	char input_file[256];
	strcpy(input_file,argv[2]);
	strcat(input_file,".nodeState");
	u32 dim[2];
	FILE *file=fopen(input_file, "rb");
	if (file == NULL) {
		printf("No nodeState input.\n");
		return 0;
	}
	fread(dim, sizeof(u32), 2, file);
	const u32 nnode=dim[0];
	const u32 nbin=dim[1];

	u32* nodeState=(u32*)(malloc(sizeof(u32)*nbin*nnode));
	fread(nodeState,sizeof(u32),nbin*nnode,file);
	fclose(file);
	u32* nodeState_gpu;
	CUDACALL(cudaMalloc((void**)&nodeState_gpu,sizeof(u32)*nbin*nnode));
	CUDACALL(cudaMemcpy(nodeState_gpu,nodeState,sizeof(u32)*nbin*nnode,cudaMemcpyHostToDevice));
	free(nodeState);

	strcpy(input_file,argv[2]);
	strcat(input_file,".outmatIdx");
	file=fopen(input_file, "rb");
	if (file == NULL) {
		printf("No outmatIdx input.\n");
		return 0;
	}
	fread(dim, sizeof(u32), 2, file);
	const u32 nSamples=dim[1];
	u32* outmatIdx=(u32*)(malloc(sizeof(u32)*nSamples));
	fread(outmatIdx,sizeof(u32),nSamples,file);
	fclose(file);
	u32* outmat=(u32*)(malloc(sizeof(u32)*nbin*nSamples));
	
	strcpy(input_file,argv[1]);
	file=fopen(input_file, "rb");
	if (file == NULL) {
		printf("No mat input.\n");
		return 0;
	}
	fread(dim, sizeof(u32), 2, file);
	fseek(file,sizeof(u32)*nbin*outmatIdx[0],SEEK_CUR);
	fread(outmat,sizeof(u32),nbin,file);
	for(u32 i=1;i<nSamples;i++){
		fseek(file,sizeof(u32)*nbin*(outmatIdx[i]-outmatIdx[i-1]-1),SEEK_CUR);
		fread(outmat+i*nbin,sizeof(u32),nbin,file);
	}
	fclose(file);
	u32* outmat_gpu;
	CUDACALL(cudaMalloc((void**)&outmat_gpu,sizeof(u32)*nbin*nSamples));
	CUDACALL(cudaMemcpy(outmat_gpu,outmat,sizeof(u32)*nbin*nSamples,cudaMemcpyHostToDevice));
	free(outmat);

	strcpy(input_file,argv[2]);
	strcat(input_file,".edge");
	file=fopen(input_file, "rb");
	if (file == NULL) {
		printf("No edge input.\n");
		return 0;
	}
	fread(dim, sizeof(u32), 2, file);
	const u32 nedge=dim[0];
	u32* edge=(u32*)(malloc(sizeof(u32)*nedge*2));
	fread(edge, sizeof(u32), nedge*2, file);
	fclose(file);
	u32* edge_gpu;
	CUDACALL(cudaMalloc((void**)&edge_gpu,sizeof(u32)*nedge*2));
	CUDACALL(cudaMemcpy(edge_gpu,edge,sizeof(u32)*nedge*2,cudaMemcpyHostToDevice));
	free(edge);

	u32* edgeState_gpu;
	CUDACALL(cudaMalloc((void**)&edgeState_gpu,sizeof(u32)*nbin*nnode));
	cudaMemset(edgeState_gpu, 0, sizeof(u32)*nbin*nnode);
	inferEdgeStateKernel<<<nedge,32>>>(edge_gpu,nodeState_gpu,edgeState_gpu,nbin);
	
	
	u32 nblock_x=(nnode-1)/CROSSSIZE+1;
	u32 nblock_y=(nSamples-1)/CROSSSIZE+1;
	u32* rootFound_gpu;
	CUDACALL(cudaMalloc((void**)&rootFound_gpu,sizeof(u32)*nSamples*nblock_x));
	u32* nShared_gpu;
	CUDACALL(cudaMalloc((void**)&nShared_gpu,sizeof(u32)*nSamples*nblock_x));
	CUDACALL(cudaMemset(nShared_gpu,0,sizeof(u32)*nSamples));
	
	const dim3 blocks_square(nblock_x,nblock_y);
	const dim3 threads_square(CROSSSIZE,CROSSSIZE);	
	findRootKernel<<<blocks_square,threads_square>>>(
		edgeState_gpu,nnode,
		outmat_gpu,nSamples,
		nbin,
		nShared_gpu,rootFound_gpu);
	
//	u32* rootFound=(u32*)(malloc(sizeof(u32)*nSamples*nblock_x));
//	CUDACALL(cudaMemcpy(rootFound,rootFound_gpu,sizeof(u32)*nSamples*nblock_x,cudaMemcpyDeviceToHost));
//	u32* nShared=(u32*)(malloc(sizeof(u32)*nSamples*nblock_x));
//	CUDACALL(cudaMemcpy(nShared,nShared_gpu,sizeof(u32)*nSamples*nblock_x,cudaMemcpyDeviceToHost));

//	for(u32 i=0;i<16;i++){
//		u32 maxValue=0;
//		u32 maxIdx=0;
//		for(u32 j=0;j<nblock_x;j++){
//			u32 tmp=nShared[i*nblock_x+j];
//			u32 tmpIdx=rootFound[i*nblock_x+j];
//			if(maxValue<tmp){
//				maxValue=tmp;
//				maxIdx=tmpIdx;
//			}
//		}
//		printf("%u,%u\n",maxIdx,maxValue);
//	}

	u32* rootGlobal_gpu;
	CUDACALL(cudaMalloc((void**)&rootGlobal_gpu,sizeof(u32)*nSamples));
	u32* nSharedGlobal_gpu;
	CUDACALL(cudaMalloc((void**)&nSharedGlobal_gpu,sizeof(u32)*nSamples));
	rowMaxKernel<<<nSamples,32>>>(nShared_gpu,rootFound_gpu,
		nSamples,nblock_x,
		nSharedGlobal_gpu,rootGlobal_gpu);
		
	u32* rootGlobal=(u32*)(malloc(sizeof(u32)*nSamples));
	u32* nSharedGlobal=(u32*)(malloc(sizeof(u32)*nSamples));
	CUDACALL(cudaMemcpy(rootGlobal,rootGlobal_gpu,sizeof(u32)*nSamples,cudaMemcpyDeviceToHost));
	CUDACALL(cudaMemcpy(nSharedGlobal,nSharedGlobal_gpu,sizeof(u32)*nSamples,cudaMemcpyDeviceToHost));
	
//	for(u32 i=0;i<16;i++){
//		printf("%u,%u\n",rootGlobal[i],nSharedGlobal[i]);
//	}
	
	char output_file[256];
	strcpy(output_file,argv[2]);
	strcat(output_file,".rootVote2");
	FILE* output=fopen(output_file,"wb");
	
	dim[0]=2,dim[1]=nSamples;
	fwrite(dim,sizeof(u32),2,output);
	fwrite(rootGlobal,sizeof(u32),nSamples,output);
	fwrite(nSharedGlobal,sizeof(u32),nSamples,output);
	fclose(output);

	CUDACALL(cudaFree(nodeState_gpu));
	CUDACALL(cudaFree(edgeState_gpu));
	CUDACALL(cudaFree(outmat_gpu));
	CUDACALL(cudaFree(edge_gpu));
	CUDACALL(cudaFree(rootFound_gpu));
	CUDACALL(cudaFree(nShared_gpu));
	CUDACALL(cudaFree(rootGlobal_gpu));
	CUDACALL(cudaFree(nSharedGlobal_gpu));
	free(rootGlobal);
	free(nSharedGlobal);

	clock_t end=clock();
	double time_taken;
	time_taken = ((double) (end - start)) / CLOCKS_PER_SEC;
	printf("voteRoot: %.2fs elapsed for %u nodes and %u outgroups.\n",time_taken,nnode,nSamples);

	return 0;
}

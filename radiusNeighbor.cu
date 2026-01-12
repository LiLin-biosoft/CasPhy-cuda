#include "cuda_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#define BLOCKSIZE 128
#define MAXMUTATE (1024*256)
#define MAXNEIGHBOR (1024*256)
#define WIDTH 32
#define LOG2WIDTH 5

typedef unsigned int u32;

__global__ void findNeighborKernel(u32* const mat,u32 const nrow,u32 const ncol,
							u32* const nNeighbor,u32* const isNeighbor, float radius,
							u32 const nMutate,u32* const mutateIdx,float* const weightRun){
	const u32 thread_xidx=threadIdx.x+blockIdx.x*blockDim.x;
	float commonCount=0;
	if(thread_xidx<ncol){
		for(u32 i=0;i<nMutate;i++){
			u32 tmpIdx=mutateIdx[i];
			u32 xx=tmpIdx>>LOG2WIDTH;
			u32 yy=tmpIdx&(WIDTH-1);
			u32 tmp=mat[thread_xidx+xx*ncol];
			if(((tmp>>yy)&1)==1){
				commonCount+=weightRun[i];
			}
		}
	}
    u32 active = __activemask();
    u32 mask = __ballot_sync(active, commonCount >= radius);
	if((thread_xidx&31)==0){
		atomicAdd(nNeighbor,__popc(mask));
		if ((thread_xidx >> 5) < (ncol + 31) / 32) {
			isNeighbor[thread_xidx>>5]=mask;
		}
	}
}

__global__ void fillvalue(u32* const data,u32 const size,u32 const value){
	u32 tid=blockIdx.x*blockDim.x+threadIdx.x;
	u32 stride=blockDim.x*gridDim.x;
    for(u32 i=tid;i<size;i+=stride){
		data[i]=value;
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


int main(int argc, char *argv[]){
	setbuf(stdout, NULL); 
	clock_t start = clock();

	if(argc<3){
		printf("Error: no sufficient args.\n");
		return 0;
	} 

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
    const u32 nrow=dim[0],ncol=dim[1];	
    const u32 nsite=nrow*WIDTH;
//	u32* mat;
//	cudaHostAlloc((void**)&mat,sizeof(u32)*nrow*ncol,cudaHostAllocDefault);	// fail on EYPC 9554
	u32* mat=(u32*)(malloc(sizeof(u32)*nrow*ncol));

	u32* mat_gpu;
	cudaMalloc((void**)&mat_gpu,sizeof(u32)*nrow*ncol);
	for(u32 i=0;i<nrow;i++){
		fread(mat+i*ncol, sizeof(u32), ncol, file);
		cudaMemcpy(mat_gpu+i*ncol,mat+i*ncol, sizeof(u32)*ncol, cudaMemcpyHostToDevice);
	}
	fclose(file);
	
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
	
	u32* nNeighbor_gpu;
	cudaMalloc((void**)&nNeighbor_gpu,sizeof(u32)*(ncol));
	u32* isNeighbor_gpu;
	u32 nbin=(ncol-1)/(WIDTH)+1;
	cudaMalloc((void**)&isNeighbor_gpu,sizeof(u32)*nbin);
	u32* nNeighbor;
	cudaHostAlloc((void**)&nNeighbor,sizeof(u32)*(ncol),cudaHostAllocDefault);
	u32* isNeighbor;
	cudaHostAlloc((void**)&isNeighbor,sizeof(u32)*nbin,cudaHostAllocDefault);
	u32* mutateIdx;
	cudaHostAlloc((void**)&mutateIdx,sizeof(u32)*MAXMUTATE,cudaHostAllocDefault);
	u32* mutateIdx_gpu;
	cudaMalloc((void**)&mutateIdx_gpu,sizeof(u32)*MAXMUTATE);
	float* weightRun;
	cudaHostAlloc((void**)&weightRun,sizeof(float)*MAXMUTATE,cudaHostAllocDefault);
	float* weightRun_gpu;
	cudaMalloc((void**)&weightRun_gpu,sizeof(float)*MAXMUTATE);

//	first scan for non-zero sites and then calculate the distance
	u32 nMutate=0;
	for(u32 j=0;j<nrow;j++){
		u32 tmp=mat[j*ncol];
		for(u32 k=0;k<WIDTH;k++){
			if(((tmp>>k)&1)==1){
				mutateIdx[nMutate]=j*WIDTH+k;
				weightRun[nMutate]=weight[j*WIDTH+k];
				nMutate++;
			}
		}
	}
	fillvalue<<<2,BLOCKSIZE>>>(nNeighbor_gpu,ncol,0);
	u32* idx=(u32*)(malloc(sizeof(u32)*MAXNEIGHBOR));
	u32 nIdx;

	char output_file[256];
	strcpy(output_file,argv[1]);
	strcat(output_file,".bk");
	FILE *output=fopen(output_file,"wb");
	
	strcpy(output_file,argv[1]);
	strcat(output_file,".idx");
	FILE *output_idx=fopen(output_file,"wb");	
	
	cudaStream_t s_upload, s_download,s_run;
	cudaStreamCreate(&s_upload);
	cudaStreamCreate(&s_download);
	cudaStreamCreate(&s_run);
		
	for(u32 i=0;i<ncol;i++){
		if((i&1023)==1023){
	      printf("=");
	    }
	    if((i&32767)==32767){
	      printf("\n");
	    }
		cudaMemcpyAsync(mutateIdx_gpu,mutateIdx,sizeof(u32)*nMutate,cudaMemcpyHostToDevice,s_upload);
		cudaMemcpyAsync(weightRun_gpu,weightRun,sizeof(float)*nMutate,cudaMemcpyHostToDevice,s_upload);
		
		cudaStreamSynchronize(s_upload);
		cudaStreamSynchronize(s_download);
		findNeighborKernel<<<(ncol-1)/BLOCKSIZE+1,BLOCKSIZE,0,s_run>>>(
			mat_gpu,nrow,ncol,
			nNeighbor_gpu+i,isNeighbor_gpu,radius,
			nMutate,mutateIdx_gpu,weightRun_gpu
		);
		if(i>0){
			fwrite(isNeighbor,sizeof(u32),nbin,output);	
			int2idx(isNeighbor,nbin,idx,&nIdx);
			fwrite(idx,sizeof(u32),nIdx,output_idx);		
		}
		if(i+1<ncol){
			nMutate=0;
			for(u32 j=0;j<nrow;j++){
				u32 tmp=mat[i+1+j*ncol];
				for(u32 k=0;k<WIDTH;k++){
					if(((tmp>>k)&1)==1){
						mutateIdx[nMutate]=j*WIDTH+k;
						weightRun[nMutate]=weight[j*WIDTH+k];
						nMutate++;
					}
				}
			}
		}
		cudaStreamSynchronize(s_run);
		cudaMemcpyAsync(isNeighbor,isNeighbor_gpu, sizeof(u32)*nbin, cudaMemcpyDeviceToHost,s_download);
	}
	printf("\n");
	cudaStreamSynchronize(s_download);
	fwrite(isNeighbor,sizeof(u32),nbin,output);
	int2idx(isNeighbor,nbin,idx,&nIdx);
	fwrite(idx,sizeof(u32),nIdx,output_idx);	
	fclose(output);
	fclose(output_idx);

	cudaStreamDestroy(s_upload);
	cudaStreamDestroy(s_download);
	cudaStreamDestroy(s_run);
	
	cudaMemcpy(nNeighbor,nNeighbor_gpu, sizeof(u32)*ncol, cudaMemcpyDeviceToHost);
	strcpy(output_file,argv[1]);
	strcat(output_file,".nNeighbor");
	FILE *output_nNeighbor=fopen(output_file,"wb");
	dim[0]=1,dim[1]=ncol;
	fwrite(dim,sizeof(u32),2,output_nNeighbor);
	fwrite(nNeighbor,sizeof(u32),ncol,output_nNeighbor);	
	fclose(output_nNeighbor);
	
//	cudaFreeHost(mat);
	free(mat);
	cudaFree(mat_gpu);
	cudaFree(nNeighbor_gpu);
	cudaFree(isNeighbor_gpu);
	cudaFreeHost(nNeighbor);
	cudaFreeHost(isNeighbor);
	cudaFreeHost(mutateIdx);
	cudaFree(mutateIdx_gpu);
	cudaFreeHost(weightRun);
	cudaFree(weightRun_gpu);
	
	
	clock_t end=clock();
	double time_taken;
	time_taken = ((double) (end - start)) / CLOCKS_PER_SEC;
	printf("radiusNeighbor: %.2fs elapsed for %u tips.\n",time_taken,ncol);
    return 0;
}

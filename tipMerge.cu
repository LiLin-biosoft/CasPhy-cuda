#include "cuda_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#define XSIZE 32
#define YSIZE 4
#define MAXNEIGHBOR (1024*1024)
#define WIDTH 32
#define LOG2WIDTH 5
#define CACHEMAX (1024*8) // 2048 MB

#define CUDACALL(call) {                                    \
    cudaError_t err = call;                                         \
    if (err != cudaSuccess) {                                       \
        printf("CUDA error at %s:%d code=%d(%s) \"%s\"\n",          \
               __FILE__, __LINE__, err, cudaGetErrorString(err), #call); \
        exit(1);                                                    \
    }                                                               \
}

typedef unsigned int u32;

__global__ void sharedNeighborKernel(u32* const mat,u32 offset,u32* const rowIndices,u32 nrow,u32 ncol,
									u32* const nShared,u32 nIndices,u32* const idxNeighbor){

	__shared__ u32 threadSum[YSIZE*XSIZE];
	
    const u32 xtid=blockIdx.x*blockDim.x+threadIdx.x;
    const u32 local_xtid=threadIdx.x;
    const u32 local_ytid=threadIdx.y;
    const u32 stride=blockDim.y;
    
    u32 local_sum=0;
	if(xtid<nrow){
		u32 rowIdx=rowIndices[xtid]-offset;
		for(u32 i=local_ytid;i<nIndices;i+=stride){
			u32 tmp=idxNeighbor[i];
			u32 xx=tmp>>LOG2WIDTH;
			u32 yy=tmp&(WIDTH-1);
			if(((mat[rowIdx*ncol+xx]>>yy)&1)==1){
				local_sum++;
			}
		}
	}
	threadSum[local_xtid*YSIZE+local_ytid]=local_sum;
    __syncthreads();

    for(u32 s=blockDim.y/2;s>0;s>>=1){
        if(local_ytid<s){
			threadSum[local_xtid*YSIZE+local_ytid]+=threadSum[local_xtid*YSIZE+local_ytid+s];
        }
        __syncthreads();
    }
    if(local_ytid==0){
    	nShared[xtid]=threadSum[local_xtid*YSIZE];
	}
}

__host__ int transitionRoot(int* const transition,int idx){
  if(idx==-1){
    return(-1);
  }else{
    int root=idx;
    while(transition[root]!=root&&transition[root]!=-1){
      root=transition[root];
    }
    return(root);
  }
}

int main(int argc, char *argv[]){
	setbuf(stdout, NULL); 
	clock_t start = clock();

	const double pShared=atof(argv[2]);

	char input_file[256];
	strcpy(input_file,argv[1]);
	strcat(input_file,".nNeighbor");
	FILE *file = fopen(input_file, "rb");
    if (file == NULL) {
    	printf("No nNeighbor input.\n");
    	return 0;
  	}
  	u32 dim[2];
    fread(dim, sizeof(u32), 2, file);
    const u32 ntip=dim[1];
    const u32 nbin=(ntip-1)/WIDTH+1;
    u32* nNeighbor=(u32*)(malloc(sizeof(u32)*ntip));
	fread(nNeighbor, sizeof(u32), ntip, file);
	fclose(file);

	strcpy(input_file,argv[1]);
	strcat(input_file,".idx");
    file = fopen(input_file, "rb");
    if (file == NULL) {
    	printf("No indices input.\n");
    	return 0;
  	} 
    u32** neighborIndices=(u32**)(malloc(sizeof(u32*)*ntip));
	u32 maxNeighbor=0;
    for(u32 i=0;i<ntip;i++){
		if(nNeighbor[i]>MAXNEIGHBOR){
			printf("find too many neighbors, increase radius to reduce No. neighbors.\n");
			return 0;
		}
		if(nNeighbor[i]>maxNeighbor){
			maxNeighbor=nNeighbor[i];
		}
    	neighborIndices[i]=(u32*)(malloc(sizeof(u32)*nNeighbor[i]));
    	fread(neighborIndices[i], sizeof(u32), nNeighbor[i], file);
	}
	fclose(file);
	
	u32* searchOffset=(u32*)(malloc(sizeof(u32)*ntip));
	for(u32 i=0;i<ntip;i++){
		u32 j=0;
		for(;j<nNeighbor[i]&&neighborIndices[i][j]<=i;j++){
			;
		}
		searchOffset[i]=j;
	}
	

	strcpy(input_file,argv[1]);
	strcat(input_file,".bk");
    file = fopen(input_file, "rb");
    if (file == NULL) {
    	printf("No bk input.\n");
    	return 0;
  	}

	u32 lineSizeMB=((sizeof(u32)*nbin)>>20)+1;
	u32 cacheSize=CACHEMAX/lineSizeMB;
	u32* cache;
	CUDACALL(cudaHostAlloc((void**)&cache,sizeof(u32)*nbin*cacheSize,cudaHostAllocDefault));	
	u32* cache_gpu;
	CUDACALL(cudaMalloc((void**)&cache_gpu,sizeof(u32)*nbin*cacheSize));
	u32* nShared=(u32*)(malloc(sizeof(u32)*maxNeighbor));
	u32* nShared_gpu;
	CUDACALL(cudaMalloc((void**)&nShared_gpu,sizeof(u32)*maxNeighbor));
	u32* idxRun=(u32*)(malloc(sizeof(32)*maxNeighbor));
	u32* idxRun_gpu;
	CUDACALL(cudaMalloc((void**)&idxRun_gpu,sizeof(u32)*maxNeighbor));
	u32* idxNeighbor;
	CUDACALL(cudaMalloc((void**)&idxNeighbor,sizeof(u32)*maxNeighbor));
	
	int* member=(int*)(malloc(sizeof(int)*ntip));
	int* transition=(int*)(malloc(sizeof(int)*ntip));
	for(u32 i=0;i<ntip;i++){
		member[i]=-1;
		transition[i]=-1;
	}
	u32 nCategory=0;
	for(u32 i=0;i<ntip;i+=cacheSize){
	      	printf("=");
	    	if(((i/cacheSize)&31)==31){
	      		printf("\n");
	    	}
		u32 nCache=cacheSize;
		if(i+nCache>ntip){
			nCache=ntip-i;
		}
		fread(cache,sizeof(u32),nbin*nCache,file);
		CUDACALL(cudaMemcpy(cache_gpu,cache,sizeof(u32)*nbin*nCache,cudaMemcpyHostToDevice));
		for(u32 j=0;j<i+nCache;j++){
			int root1=member[j];
			if(root1!=-1){
				root1=transitionRoot(transition,root1);
			}
			u32 nIdxRun=0;
			for(u32 k=searchOffset[j];k<nNeighbor[j];k++){
				u32 tmpIdx=neighborIndices[j][k];
				if(tmpIdx>=i+nCache){
					searchOffset[j]=k;
					break;
				}
				if(tmpIdx>=i){
					if(root1==-1||root1!=transitionRoot(transition,tmpIdx)){
						idxRun[nIdxRun]=neighborIndices[j][k];
						nIdxRun++;						
					}
				}
			}
			if(nIdxRun>0){
				CUDACALL(cudaMemcpy(idxRun_gpu,idxRun,sizeof(u32)*nIdxRun,cudaMemcpyHostToDevice));
				CUDACALL(cudaMemcpy(idxNeighbor,neighborIndices[j],sizeof(u32)*nNeighbor[j],cudaMemcpyHostToDevice));
				u32 nblock=(nIdxRun-1)/XSIZE+1;
				const dim3 threads_square(XSIZE,YSIZE);	
				sharedNeighborKernel<<<nblock,threads_square>>>(
					cache_gpu,i,idxRun_gpu,nIdxRun,nbin,nShared_gpu,nNeighbor[j],idxNeighbor
				);
				CUDACALL(cudaMemcpy(nShared,nShared_gpu,sizeof(u32)*nIdxRun,cudaMemcpyDeviceToHost));
				for(u32 k=0;k<nIdxRun;k++){
					u32 tmpIdx=idxRun[k];
					u32 tmpShared=nShared[k];
					if(tmpShared>nNeighbor[j]*pShared||tmpShared>nNeighbor[tmpIdx]*pShared){
						int root2=transitionRoot(transition,member[tmpIdx]);
						if(root1==-1){
							if(root2==-1){
								root1=nCategory;
								transition[nCategory]=nCategory;
								member[j]=nCategory;
								member[tmpIdx]=nCategory;
								nCategory++;								
							}else{
								root1=root2;
								member[j]=root2;
							}
						}else{
							if(root2==-1){
								member[tmpIdx]=root1;
							}else{
								transition[root1]=root2;
								member[j]=root2;
								root1=root2;
							}
						}
					}
				}
			}
		}
	}
	printf("\n");
	fclose(file);
    for(u32 i=0;i<ntip;i++){
    	free(neighborIndices[i]);
	}	
	free(neighborIndices);

	char output_file[256];
	strcpy(output_file,argv[1]);
	strcat(output_file,".member");
	FILE *output = fopen(output_file, "w");
	for(u32 i=0;i<ntip;i++){
		if(member[i]==-1){
			member[i]=nCategory;
			nCategory++;
		}else{
			member[i]=transitionRoot(transition,member[i]);
		}
		fprintf(output,"%d\n",member[i]);
	}
	fclose(output);
	
	clock_t end=clock();
	double time_taken;
	time_taken = ((double) (end - start)) / CLOCKS_PER_SEC;
	printf("tipMerge: %.2fs elapsed for %u tips.\n",time_taken,ntip);
    return 0;
}

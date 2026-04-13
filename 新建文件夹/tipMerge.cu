#include "cuda_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <stdint.h>

#define MAXNEIGHBOR (1024*1024)
#define MAXINDICES UINT64_MAX
#define CACHEMAX (1024*128)

#define CUDACALL(call) {                                    \
    cudaError_t err = call;                                         \
    if (err != cudaSuccess) {                                       \
        printf("CUDA error at %s:%d code=%d(%s) \"%s\"\n",          \
               __FILE__, __LINE__, err, cudaGetErrorString(err), #call); \
        exit(1);                                                    \
    }                                                               \
}

typedef unsigned int u32;
typedef unsigned long int u64;

//tipMergeIdxmulti

__global__ void sharedIdxKernel(
				u32* refIndices,
				const u32 nIdxRef,
				u32* const neighborIndices,
				const u64 globalOffset,
				u64* const idxOffset,
				u32* const nNeighbor,
				u32* const idxRun,
				u32* const nShared,
				const u32 bidOffset
			){
				
	__shared__ u32 refCache[32];
	__shared__ u32 qryCache[32];
	
	u32 bid=blockIdx.x+bidOffset;
	u32 tid=threadIdx.x;
	u32 qryRunIdx=idxRun[bid];
	u32 nIdxQry=nNeighbor[qryRunIdx];
	u32* qryIndices=neighborIndices+idxOffset[qryRunIdx]-globalOffset;

	u32 nProcessedRef=0;
	u32 nProcessedQry=0;
	u32 rMin=0;
	u32 rMax=0;
	u32 qMin=0;
	u32 qMax=0;
	u32 threadFound=0;

	refCache[tid]=refIndices[tid+nProcessedRef];
	// __syncthreads();
	nProcessedRef+=32;
	rMin=refCache[0];
	rMax=refCache[min(31,nIdxRef-nProcessedRef+31)];
	
	qryCache[tid]=qryIndices[tid+nProcessedQry];
	// __syncthreads();
	nProcessedQry+=32;
	qMin=qryCache[0];
	qMax=qryCache[min(31,nIdxQry-nProcessedQry+31)];

	if(rMin<=qMax&&qMin<=rMax){
		u32 valueRef=refCache[tid];
		bool inRange=(valueRef>=qMin&&valueRef<=qMax);
		unsigned int mask =__ballot_sync(0xFFFFFFFF,inRange);
		int lowest_bit=__ffs(mask)-1;
		int highest_bit=31-__clz(mask);
		u32 rBound=min(highest_bit,nIdxRef-nProcessedRef+31);
			
		if(nProcessedQry-32+tid<nIdxQry){
			u32 valueQry=qryCache[tid];
			if(valueQry<=rMax){
				for(u32 i=lowest_bit;i<=rBound;++i){
					u32 valueRef=refCache[i];
					if(valueQry==valueRef){
						++threadFound;
					}
				}					
			}
		}
	}
	while(!(nProcessedRef>=nIdxRef&&rMax<=qMax)&&!(nProcessedQry>=nIdxQry&&qMax<=rMax)){
		if(rMax>qMax){
			qryCache[tid]=qryIndices[tid+nProcessedQry];
			nProcessedQry+=32;
			// __syncthreads();
			qMin=qryCache[0];
			qMax=qryCache[min(31,nIdxQry-nProcessedQry+31)];
		}else{
			refCache[tid]=refIndices[tid+nProcessedRef];
			nProcessedRef+=32;
			// __syncthreads();
			rMin=refCache[0];
			rMax=refCache[min(31,nIdxRef-nProcessedRef+31)];
		}		
		if(rMin<=qMax&&qMin<=rMax){
			u32 valueRef=refCache[tid];
			bool inRange=(valueRef>=qMin&&valueRef<=qMax);
			unsigned int mask =__ballot_sync(0xFFFFFFFF,inRange);
			int lowest_bit=__ffs(mask)-1;
			int highest_bit=31-__clz(mask);
			u32 rBound=min(highest_bit,nIdxRef-nProcessedRef+31);
			
			if(nProcessedQry-32+tid<nIdxQry){
				u32 valueQry=qryCache[tid];
				if(valueQry<=rMax){
					for(u32 i=lowest_bit;i<=rBound;++i){
						u32 valueRef=refCache[i];
						if(valueQry==valueRef){
							++threadFound;
						}
					}					
				}
			}
		}
	}
	
	unsigned mask = 0xffffffff;
	for (int offset = 16; offset > 0; offset >>= 1) {
        threadFound += __shfl_down_sync(mask, threadFound, offset);
    }
	if(tid==0){
		nShared[bid]=threadFound;
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
	// int gpuCount=2;
	int gpuCount;
	CUDACALL(cudaGetDeviceCount(&gpuCount));
	printf("%u devices detected\n",gpuCount);
	
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
	u32* nNeighbor;
	CUDACALL(cudaHostAlloc((void**)&nNeighbor,sizeof(u32)*ntip,cudaHostAllocDefault));
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
	u64* idxOffset;
	CUDACALL(cudaHostAlloc((void**)&idxOffset,sizeof(u64)*(ntip+1),cudaHostAllocDefault));
	idxOffset[0]=0UL;
	
    for(u32 i=0;i<ntip;i++){
		if(nNeighbor[i]>MAXNEIGHBOR){
			printf("find too many neighbors, increase radius to reduce No. neighbors.\n");
			return 0;
		}
		if(nNeighbor[i]>maxNeighbor){
			maxNeighbor=nNeighbor[i];
		}
		u64 nAllocate=(nNeighbor[i]+31)/32*32;
		if(idxOffset[i]+nAllocate>MAXINDICES){
			printf("No. indices reach limit, increase radius to reduce No. neighbors.\n");
			return 0;
		}else{
			idxOffset[i+1]=idxOffset[i]+nAllocate;
		}
    	neighborIndices[i]=(u32*)(malloc(sizeof(u32)*nNeighbor[i]));
    	fread(neighborIndices[i], sizeof(u32), nNeighbor[i], file);
	}
	fclose(file);

	u32* neighborIndices_tmp;
	CUDACALL(cudaHostAlloc(
		(void**)&neighborIndices_tmp,
		sizeof(u32)*maxNeighbor,
		cudaHostAllocDefault
	));

	u32* searchOffset=(u32*)(malloc(sizeof(u32)*ntip));
	for(u32 i=0;i<ntip;i++){
		u32 j=0;
		for(;j<nNeighbor[i]&&neighborIndices[i][j]<=i;j++){
			;
		}
		searchOffset[i]=j;
	}

	u32 cacheSize=0;
	for(u32 i=0;i<ntip;i+=CACHEMAX){
		u32 tmp=0;
		if(i+CACHEMAX<ntip){
			tmp=idxOffset[i+CACHEMAX]-idxOffset[i];
		}else{
			tmp=idxOffset[ntip]-idxOffset[i];
		}
		if(tmp>cacheSize){
			cacheSize=tmp;
		}
	}
	printf("Cache size in GPU: %uMB\n",cacheSize>>18);

	u32* nShared;
	CUDACALL(cudaHostAlloc(
		(void**)&nShared,
		sizeof(u32)*maxNeighbor,
		cudaHostAllocDefault
	));
	u32* idxRun;
	CUDACALL(cudaHostAlloc(
		(void**)&idxRun,
		sizeof(u32)*maxNeighbor,
		cudaHostAllocDefault
	));

	cudaStream_t* s_run=(cudaStream_t*)(malloc(sizeof(cudaStream_t)*gpuCount));
	u32** neighborIndices_gpu=(u32**)(malloc(sizeof(u32*)*gpuCount));
	u32** refIndices_gpu=(u32**)(malloc(sizeof(u32*)*gpuCount));
	u64** idxOffset_gpu=(u64**)(malloc(sizeof(u64*)*gpuCount));
	u32** nNeighbor_gpu=(u32**)(malloc(sizeof(u32*)*gpuCount));
	u32** nShared_gpu=(u32**)(malloc(sizeof(u32*)*gpuCount));
	u32** idxRun_gpu=(u32**)(malloc(sizeof(u32*)*gpuCount));
	for(u32 gpuIdx=0;gpuIdx<gpuCount;++gpuIdx){
		CUDACALL(cudaSetDevice(gpuIdx));
		CUDACALL(cudaStreamCreate(&(s_run[gpuIdx])));
		CUDACALL(cudaMalloc((void**)&(neighborIndices_gpu[gpuIdx]),sizeof(u32)*cacheSize));
		CUDACALL(cudaMalloc((void**)&(refIndices_gpu[gpuIdx]),sizeof(u32)*maxNeighbor));
		CUDACALL(cudaMalloc((void**)&(idxOffset_gpu[gpuIdx]),sizeof(u64)*(ntip+1)));
		CUDACALL(cudaMemcpyAsync(
			idxOffset_gpu[gpuIdx],idxOffset,sizeof(u64)*(ntip+1),
			cudaMemcpyHostToDevice,s_run[gpuIdx]
		));
		CUDACALL(cudaMalloc((void**)&(nNeighbor_gpu[gpuIdx]),sizeof(u32)*ntip));
		CUDACALL(cudaMemcpyAsync(
			nNeighbor_gpu[gpuIdx],nNeighbor,sizeof(u32)*ntip,
			cudaMemcpyHostToDevice,s_run[gpuIdx]
		));
		CUDACALL(cudaMalloc((void**)&(nShared_gpu[gpuIdx]),sizeof(u32)*maxNeighbor));
		CUDACALL(cudaMalloc((void**)&(idxRun_gpu[gpuIdx]),sizeof(u32)*maxNeighbor));
	}

	int* member=(int*)(malloc(sizeof(int)*ntip));
	int* transition=(int*)(malloc(sizeof(int)*ntip));
	for(u32 i=0;i<ntip;i++){
		member[i]=-1;
		transition[i]=-1;
	}
	u32 nCategory=0;
	
	double gpu_taken=0;
	u32* nblockGPU=(u32*)(malloc(sizeof(u32)*(gpuCount)));
	u32* bidOffset=(u32*)(malloc(sizeof(u32)*(gpuCount+1)));
	
	for(u32 i=0;i<ntip;i+=CACHEMAX){
	      	printf("=");
	    	if(((i/CACHEMAX)&31)==31){
	      		printf("\n");
	    	}
		u32 nCache=CACHEMAX;
		if(i+nCache>ntip){
			nCache=ntip-i;
		}
		
		u64 globalOffset=idxOffset[i];
		for(u32 j=i;j<i+nCache;++j){
			memcpy(
				neighborIndices_tmp,
				neighborIndices[j],
				sizeof(u32)*nNeighbor[j]
			);
			for(u32 gpuIdx=0;gpuIdx<gpuCount;++gpuIdx){
				CUDACALL(cudaSetDevice(gpuIdx));
				CUDACALL(cudaMemcpyAsync(
					neighborIndices_gpu[gpuIdx]+idxOffset[j]-globalOffset,
					neighborIndices_tmp,
					sizeof(u32)*nNeighbor[j],
					cudaMemcpyHostToDevice,s_run[gpuIdx]
				));
			}
			for(u32 gpuIdx=0;gpuIdx<gpuCount;++gpuIdx){
				CUDACALL(cudaSetDevice(gpuIdx));
				CUDACALL(cudaStreamSynchronize(s_run[gpuIdx]));
			}
		}
		
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
				u32 nblockDivide=(nIdxRun-1)/gpuCount+1;
				bidOffset[0]=0;
				for(u32 gpuIdx=0;gpuIdx<gpuCount;++gpuIdx){
					u32 tmp=bidOffset[gpuIdx]+nblockDivide;
					if(tmp<nIdxRun){
						bidOffset[gpuIdx+1]=tmp;
						nblockGPU[gpuIdx]=nblockDivide;
					}else{
						bidOffset[gpuIdx+1]=nIdxRun;
						nblockGPU[gpuIdx]=nIdxRun-bidOffset[gpuIdx];
					}
				}
				
				clock_t gpu_start=clock();
				memcpy(neighborIndices_tmp,neighborIndices[j],sizeof(u32)*nNeighbor[j]);
				for(u32 gpuIdx=0;gpuIdx<gpuCount;++gpuIdx){
					CUDACALL(cudaSetDevice(gpuIdx));
					CUDACALL(cudaMemcpyAsync(
						idxRun_gpu[gpuIdx],idxRun,sizeof(u32)*nIdxRun,
						cudaMemcpyHostToDevice,s_run[gpuIdx]
					));
					CUDACALL(cudaMemcpyAsync(
						refIndices_gpu[gpuIdx],neighborIndices_tmp,sizeof(u32)*nNeighbor[j],
						cudaMemcpyHostToDevice,s_run[gpuIdx]
					));
					sharedIdxKernel<<<nblockGPU[gpuIdx],32,0,s_run[gpuIdx]>>>(
						refIndices_gpu[gpuIdx],nNeighbor[j],
						neighborIndices_gpu[gpuIdx],globalOffset,
						idxOffset_gpu[gpuIdx],nNeighbor_gpu[gpuIdx],
						idxRun_gpu[gpuIdx],nShared_gpu[gpuIdx],
						bidOffset[gpuIdx]
					);
					CUDACALL(cudaMemcpyAsync(
						nShared+bidOffset[gpuIdx],
						nShared_gpu[gpuIdx]+bidOffset[gpuIdx],
						sizeof(u32)*nblockGPU[gpuIdx],
						cudaMemcpyDeviceToHost,s_run[gpuIdx]
					));
				}
				for(u32 gpuIdx=0;gpuIdx<gpuCount;++gpuIdx){
					CUDACALL(cudaSetDevice(gpuIdx));
					CUDACALL(cudaStreamSynchronize(s_run[gpuIdx]));
				}
				clock_t gpu_end=clock();
				gpu_taken += ((double) (gpu_end - gpu_start)) / CLOCKS_PER_SEC;
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
    for(u32 i=0;i<ntip;i++){
    	free(neighborIndices[i]);
	}
	free(neighborIndices);
	CUDACALL(cudaFreeHost(neighborIndices_tmp));
	for(u32 gpuIdx=0;gpuIdx<gpuCount;++gpuIdx){
		CUDACALL(cudaSetDevice(gpuIdx));
		CUDACALL(cudaFree(refIndices_gpu[gpuIdx]));
		CUDACALL(cudaFree(neighborIndices_gpu[gpuIdx]));
		CUDACALL(cudaFree(idxOffset_gpu[gpuIdx]));
		CUDACALL(cudaFree(nNeighbor_gpu[gpuIdx]));
		CUDACALL(cudaFree(idxRun_gpu[gpuIdx]));
		CUDACALL(cudaFree(nShared_gpu[gpuIdx]));
		CUDACALL(cudaStreamDestroy(s_run[gpuIdx]));
	}
	
	free(s_run);
	free(refIndices_gpu);
	free(neighborIndices_gpu);
	free(idxOffset_gpu);
	free(nNeighbor_gpu);
	free(idxRun_gpu);
	free(nShared_gpu);

	CUDACALL(cudaFreeHost(nNeighbor));	
	CUDACALL(cudaFreeHost(idxOffset));
	CUDACALL(cudaFreeHost(idxRun));
	CUDACALL(cudaFreeHost(nShared));

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
	printf("tipMerge: %.2fs elapsed in GPU.\n",gpu_taken);
    return 0;
}

#include "cuda_runtime.h"
#include <stdio.h>
#include <stdlib.h>

#define BLOCKSIZE 128
#define MAXMUTATE (1024*8)
#define WIDTH 32
#define LOG2WIDTH 5

typedef unsigned int u32;

__constant__ u32 mutateIdxConst[MAXMUTATE];
__constant__ float weightConst[MAXMUTATE];

__global__ void fillFloat(float* const data,u32 const size,float const value){
	u32 tid=blockIdx.x*blockDim.x+threadIdx.x;
	u32 stride=blockDim.x*gridDim.x;
    for(u32 i=tid;i<size;i+=stride){
		data[i]=value;
    }
}

__global__ void fillFloatCol(float* const data,u32 const size,float const value){
	u32 tid=blockIdx.x*blockDim.x+threadIdx.x;
	u32 stride=blockDim.x*gridDim.x;
    for(u32 i=tid;i<size;i+=stride){
		data[i*size]=value;
    }
}

__global__ void calculateSharedKernel(u32 nMutate,
							u32* const cacheMat,u32 const nbin, 
							u32* const rowIndices,u32 nIndices,
							float* const cacheShared){
	const u32 tid=threadIdx.x+blockIdx.x*blockDim.x;
	if(tid<nIndices){
		const u32 rowIdx=rowIndices[tid];
		float commonCount=0;
		for(u32 i=0;i<nMutate;i++){
			u32 tmpIdx=mutateIdxConst[i];
			u32 xx=tmpIdx>>LOG2WIDTH;
			u32 yy=tmpIdx&(WIDTH-1);
			u32 tmp=cacheMat[rowIdx*nbin+xx];
			if(((tmp>>yy)&1)==1){
				commonCount+=weightConst[i];
			}
		}
		cacheShared[rowIdx]=commonCount;
	}
}

__global__ void updateAncestralKernel(
					u32* const seq1,u32* const bistate1,
					u32* const seq2,u32* const bistate2,
					const u32 size){

	u32 tid = blockIdx.x * blockDim.x + threadIdx.x;
	u32 stride = blockDim.x * gridDim.x;
	for (u32 i = tid; i < size; i += stride) {
	    u32 s1 = seq1[i];
	    u32 s2 = seq2[i];
	    u32 bs1 = bistate1[i];
	    u32 bs2 = bistate2[i];
	    const int diff = s1 ^ s2; //s1 and s2 is different
	    const int conflict = diff & (bs1 | bs2); //different and one seq is bistate => res should be 0
	    
	    seq1[i] = (s1 | s2) & ~conflict;
	    bistate1[i] = (diff & ~(bs1 ^ bs2)) | (bs1 & bs2); //res is bistate if bs1=0, bs2=0,diff=1 or bs1=1, bs2=1
	}
}

__global__ void updateAncestralKernel2(
					u32* const seq1,u32* const bistate1,
					u32* const seq2,u32* const bistate2,
					const u32 size){

	u32 tid = blockIdx.x * blockDim.x + threadIdx.x;
	u32 stride = blockDim.x * gridDim.x;
	for (u32 i = tid; i < size; i += stride) {
	    u32 s1 = seq1[i];
	    u32 s2 = seq2[i];
	    u32 bs1 = bistate1[i];
	    u32 bs2 = bistate2[i];
	    
	    seq1[i] = (s1&s2)|(bs1&s2)|(bs2&s1);
	    bistate1[i] = (s1^s2)&(~(bs1|bs2));
	}
}

__global__ void nodeStateOutputKernel(u32* const seq,u32* const bistate,u32 const size,u32* const output){
	u32 tid = blockIdx.x * blockDim.x + threadIdx.x;
	u32 stride = blockDim.x * gridDim.x;
	for (u32 i = tid; i < size; i += stride) {
		output[i]=seq[i]&(~bistate[i]);
	}
}

__global__ void findMaxFloatKernel(const float* array,u32* const arrayIndices,u32 size,
							float* max_values,u32* max_indices){
    __shared__ float threadMax[BLOCKSIZE];
    __shared__ u32 threadMaxIdx[BLOCKSIZE];
    
    u32 tid=threadIdx.x;
    u32 global_id=blockIdx.x*blockDim.x+threadIdx.x;
    u32 stride=blockDim.x*gridDim.x;
    
    float local_max=-1;
    u32 local_index=0;
    
    // each thread process several elements
    for(u32 i=global_id;i<size;i+=stride){
    	float value=array[i];
    	u32 idx=arrayIndices[i];
       if(value>local_max||value==local_max&&idx<local_index){
            local_max=value;
            local_index=idx;
        }
    }
    
    // each thread save local max to shared memory
    threadMax[tid]=local_max;
    threadMaxIdx[tid]=local_index;
    __syncthreads();
    
    // find local max in each block
    for(u32 s=blockDim.x/2;s>0;s>>=1){
        if(tid<s){
            if(threadMax[tid+s]>threadMax[tid]||(threadMax[tid+s]==threadMax[tid]&&threadMaxIdx[tid+s]<threadMaxIdx[tid])) {
				threadMax[tid]=threadMax[tid+s];
                threadMaxIdx[tid]=threadMaxIdx[tid+s];
            }
        }
        __syncthreads();
    }
    
    // save block local max to output
    if(tid==0){
        max_values[blockIdx.x]=threadMax[0];
        max_indices[blockIdx.x]=threadMaxIdx[0];
    }
}

__global__ void findValueKernel(u32* const edge,u32 nedge,
							u32 value, u32* idx){
    __shared__ u32 threadFound[BLOCKSIZE];
    
    u32 tid=threadIdx.x;
    u32 stride=blockDim.x;
    
    u32 lineIdx=nedge+1;
    
    // each thread process several elements
    for(u32 i=tid;i<nedge;i+=stride){
		if(edge[i*2]==value&&i<=lineIdx){
			lineIdx=i;
		}
    }
    
    // each thread save local minIdx to shared memory
    threadFound[tid]=lineIdx;
    __syncthreads();
    
    // find local minIdx in each block
    for(u32 s=blockDim.x/2;s>0;s>>=1){
        if(tid<s){
            if(threadFound[tid]>threadFound[tid+s]) {
				threadFound[tid]=threadFound[tid+s];
            }
        }
        __syncthreads();
    }
    
    // save global minIdx to output
    if(tid==0){
        idx[0]=threadFound[0];
        edge[idx[0]*2]=(nedge+1);
    }
}

__global__ void findValueKernel2(u32* const edge,u32 nedge,
							u32 value, u32* idx){
    u32 tid=threadIdx.x;
    u32 stride=blockDim.x;
    for(u32 i=tid;i<nedge;i+=stride){
		if(edge[i*2+1]==value){
			idx[0]=i;
		}
    }
}


//each node calculate shared mutations with previous nodes
//place cacheShared in host mem

int main(int argc, char *argv[]){
	setbuf(stdout, NULL); 
	clock_t start = clock();
	
	int ancestralType=atoi(argv[3]);
	FILE *file;
	char input_file[256];
	strcpy(input_file,argv[1]);
	strcat(input_file,".tbmat"); //read mat,each row is one tip
    u32 dim[2]; 
    file = fopen(input_file, "rb");
    if (file == NULL) {
    	printf("No tbmat input.\n");
    	return 0;
  	} 
    fread(dim, sizeof(u32), 2, file);
    const u32 ntip=dim[0],nbin=dim[1];	
    const u32 nnode=ntip*2-1;
    const u32 nedge=ntip*2-2;
    const u32 nsite=nbin*WIDTH;
//    printf("ntip: %u, nsite: %u\n",ntip,nsite);
	u32* oneLine;
	cudaHostAlloc((void**)&oneLine,sizeof(u32)*nbin,cudaHostAllocDefault);	
	
	u32* cacheMatHost;
	cudaHostAlloc((void**)&cacheMatHost,sizeof(u32)*nbin*ntip,cudaHostAllocDefault);
	u32* cacheMat;
	cudaMalloc((void**)&cacheMat,sizeof(u32)*ntip*nbin);
	u32* cacheBistate;
	cudaMalloc((void**)&cacheBistate,sizeof(u32)*ntip*nbin);
	cudaMemset(cacheBistate, 0, sizeof(u32)*ntip*nbin);
	u32* cacheTmp;
	cudaMalloc((void**)&cacheTmp,sizeof(u32)*nbin);

	fread(cacheMatHost, sizeof(u32), nbin*ntip, file);
	cudaMemcpy(cacheMat,cacheMatHost, sizeof(u32)*nbin*ntip, cudaMemcpyHostToDevice);
	cudaFreeHost(cacheMatHost);
	fclose(file);
	
    file=fopen(argv[2], "rb"); //read weight, double vector
    if (file == NULL) {
    	printf("No weight input.\n");
    	return 0;
  	} 
  	
	// transform weight from double to float
  	double* weightTmp=(double*)(malloc(sizeof(double)*nsite));
	fread(weightTmp, sizeof(double),nsite, file);  
	fclose(file);	
	float* weight=(float*)(malloc(sizeof(float)*nsite));
	for(u32 i=0;i<nsite;i++){
		weight[i]=(float)(weightTmp[i]);
	}
	free(weightTmp);

	float** cacheShared=(float**)(malloc(sizeof(float*)*ntip));
	for(u32 i=0;i<ntip;i++){
		cacheShared[i]=(float*)(malloc(sizeof(float)*ntip));
	}
//	float* cacheShared=(float*)(malloc(sizeof(float)*ntip*ntip));
	float* cacheShared_gpu;
	cudaMalloc((void**)&cacheShared_gpu,sizeof(float)*ntip);
	u32* mutateIdx;
	cudaHostAlloc((void**)&mutateIdx,sizeof(u32)*MAXMUTATE,cudaHostAllocDefault);
	float* weightRun;
	cudaHostAlloc((void**)&weightRun,sizeof(float)*MAXMUTATE,cudaHostAllocDefault);
	float* localMax;
	cudaMalloc((void**)&localMax,sizeof(float)*ntip);
	float* localMaxHost;
	cudaHostAlloc((void**)&localMaxHost,sizeof(float)*ntip,cudaHostAllocDefault);
	u32* localMaxIdx=(u32*)(malloc(sizeof(u32)*ntip));

	u32* nodeState;
	cudaHostAlloc((void**)&nodeState,sizeof(u32)*nbin*nnode,cudaHostAllocDefault);
	u32* node2cache=(u32*)(malloc(sizeof(u32)*nnode));
	u32* cache2node;
	cudaMalloc((void**)&cache2node,sizeof(u32)*ntip);
	u32* validCacheIndices=(u32*)(malloc(sizeof(u32)*ntip));
	u32 nValidCache=ntip;
	// initialize links from node to cache
	node2cache[0]=0;
	validCacheIndices[0]=0;
	for(u32 i=1;i<ntip;i++){
		node2cache[i]=i;
		node2cache[i-1+ntip]=ntip; //=ntip means invalid
		validCacheIndices[i]=i;
	}
	// initialize links from cache to node
	cudaMemcpy(cache2node,node2cache,sizeof(u32)*ntip,cudaMemcpyHostToDevice);
	
	u32 maxBlockN=(ntip-1)/BLOCKSIZE+1;
//	printf("maxBlockN: %u\n",maxBlockN);
	float* maxValues_gpu;
	cudaMalloc((void**)&maxValues_gpu,sizeof(float)*maxBlockN);
	u32* maxIndices_gpu;
	cudaMalloc((void**)&maxIndices_gpu,sizeof(u32)*maxBlockN);
	float* maxValues;
	cudaHostAlloc((void**)&maxValues,sizeof(float)*maxBlockN,cudaHostAllocDefault);	
	u32* maxIndices;	
	cudaHostAlloc((void**)&maxIndices,sizeof(u32)*maxBlockN,cudaHostAllocDefault);
	u32* rowIndices_gpu;
	cudaMalloc((void**)&rowIndices_gpu,sizeof(u32)*(ntip-1));

	float maxShared=-1;
	for(u32 i=0;i<ntip-1;i++){
		// copy the mutation profile in cache i from GPU to host and scan for mutation
		cudaMemcpy(oneLine,cacheMat+i*nbin,sizeof(u32)*nbin,cudaMemcpyDeviceToHost);
		u32 nMutate=0;
		for(u32 j=0;j<nbin;j++){
			u32 tmp=oneLine[j];
			for(u32 k=0;k<WIDTH;k++){
				if(((tmp>>k)&1)==1){
					mutateIdx[nMutate]=j*WIDTH+k;
					weightRun[nMutate]=weight[j*WIDTH+k];
					nMutate++;
				}
			}
		}
		// copy the indices of mutated sites and weight to GPU
		cudaMemcpyToSymbol(mutateIdxConst,mutateIdx,sizeof(u32)*nMutate);
		cudaMemcpyToSymbol(weightConst,weightRun,sizeof(float)*nMutate);
		// copy the indices of rows for calculation to GPU
		u32 nIndices=ntip-i-1;
		u32 BlockN=(nIndices-1)/BLOCKSIZE+1;
		cudaMemcpy(rowIndices_gpu,node2cache+i+1,sizeof(u32)*nIndices,cudaMemcpyHostToDevice);
		// calcuate shared
		fillFloat<<<1,BLOCKSIZE>>>(cacheShared_gpu,i+1,-1);
		calculateSharedKernel<<<BlockN,BLOCKSIZE>>>(
			nMutate,cacheMat,nbin,
			rowIndices_gpu,nIndices,
			cacheShared_gpu
		);
		cudaMemcpy(cacheShared[i],cacheShared_gpu,sizeof(float)*ntip,cudaMemcpyDeviceToHost);
		
		// find block local max
		findMaxFloatKernel<<<BlockN,BLOCKSIZE>>>(
			cacheShared_gpu+i+1,cache2node+i+1,nIndices,
			maxValues_gpu,maxIndices_gpu
		);
		cudaMemcpy(maxValues,maxValues_gpu,sizeof(float)*BlockN,cudaMemcpyDeviceToHost);
		cudaMemcpy(maxIndices,maxIndices_gpu,sizeof(u32)*BlockN,cudaMemcpyDeviceToHost);
		// find row local max
		maxShared=-1;
		u32 maxIdx=0;
		for(u32 j=0;j<BlockN;j++){
			if(maxValues[j]>maxShared||maxValues[j]==maxShared&&maxIndices[j]<maxIdx){
				maxShared=maxValues[j];
				maxIdx=maxIndices[j];
			}
		}
//		printf("%u: %.2f,%u\n",i,maxShared,maxIdx);
//		 write row local max to GPU
		cudaMemcpy(localMax+i,&maxShared,sizeof(float),cudaMemcpyHostToDevice);
		localMaxIdx[i]=maxIdx;
	}
	// write the first row local max to GPU (invalid)
	for(u32 i=0;i<ntip;i++){
		cacheShared[ntip-1][i]=-1;
	}
	maxShared=-1;
	cudaMemcpy(localMax+ntip-1,&maxShared,sizeof(float),cudaMemcpyHostToDevice);

	// check cacheMat	
//	for(u32 i=0;i<5;i++){
//		for(u32 j=0;j<5;j++){
//			printf("%.2f,",cacheShared[j+i*ntip]);
//		}
//		printf("\n");
//	}

	// initialize edge and node index
	u32* edge=(u32*)(malloc(sizeof(u32)*nedge*2));
	u32 nodeCount=ntip;
	
	// start join
	while(nodeCount<nnode){
//		if(((nodeCount-ntip)&1023)==1023){
//      		printf("=");
//    	}
//	    if(((nodeCount-ntip)&32767)==32767){
//	      printf("\n");
//	    }		
		u32 joinIdx1=0,joinIdx2=0;
		u32 cacheIdx1=0,cacheIdx2=0;
		do{  //find global maximum
			findMaxFloatKernel<<<(ntip-1)/BLOCKSIZE+1,BLOCKSIZE>>>(
				localMax,cache2node,ntip,maxValues_gpu,maxIndices_gpu
			);
			cudaMemcpy(maxValues,maxValues_gpu,maxBlockN*sizeof(float),cudaMemcpyDeviceToHost);
			cudaMemcpy(maxIndices,maxIndices_gpu,maxBlockN*sizeof(u32),cudaMemcpyDeviceToHost);
			maxShared=-1;
			for(u32 j=0;j<maxBlockN;j++){
				if(maxValues[j]>maxShared||maxValues[j]==maxShared&&maxIndices[j]<joinIdx1){
					maxShared=maxValues[j];
					joinIdx1=maxIndices[j]; //get the cacheIdx
				}
			}
			cacheIdx1=node2cache[joinIdx1];
			joinIdx2=localMaxIdx[cacheIdx1];
			cacheIdx2=node2cache[joinIdx2];
			
		
			if(cacheIdx2>=ntip){ // if joinIdx2 is not in cache (has been used), update joinIdx1's local maximum
				// global max found but joinIdx2 is invalid
//				printf("## joinIdx1: %u(cacheIdx: %u), joinIdx2: %u(cacheIdx: invalid), shared: %.2f\n",
//					joinIdx1,cacheIdx1,joinIdx2,maxShared);
//				cudaMemcpy(rowIndices_gpu,validCacheIndices,sizeof(u32)*nValidCache,cudaMemcpyHostToDevice);
				u32 BlockN=(ntip-1)/BLOCKSIZE+1;
				cudaMemcpy(cacheShared_gpu,cacheShared[cacheIdx1],sizeof(float)*ntip,cudaMemcpyHostToDevice); 
				findMaxFloatKernel<<<BlockN,BLOCKSIZE>>>(
					cacheShared_gpu,cache2node,ntip,
					maxValues_gpu,maxIndices_gpu
				); 
				cudaMemcpy(maxValues,maxValues_gpu,BlockN*sizeof(float),cudaMemcpyDeviceToHost);
				cudaMemcpy(maxIndices,maxIndices_gpu,BlockN*sizeof(u32),cudaMemcpyDeviceToHost);
				maxShared=-1;
				u32 maxIdx=0;
				for(u32 j=0;j<BlockN;j++){
					if(maxValues[j]>maxShared||maxValues[j]==maxShared&&maxIndices[j]<maxIdx){
						maxShared=maxValues[j];
						maxIdx=maxIndices[j];
					}
				}
				cudaMemcpy(localMax+cacheIdx1,&maxShared,sizeof(float),cudaMemcpyHostToDevice);
				localMaxIdx[cacheIdx1]=maxIdx;
//				printf("## update rowMax of %u(cacheIdx: %u) -> node %u(cacheIdx: %u), shared: %.2f\n",
//					joinIdx1,cacheIdx1,maxIdx,node2cache[maxIdx],maxShared);
			}
		}while(cacheIdx2>=ntip); // Loop until joinIdx2 is not used
		
		// global max found
//		printf("joinIdx1: %u(cacheIdx: %u), joinIdx2: %u(cacheIdx: %u), shared: %.2f\n",
//			joinIdx1,cacheIdx1,joinIdx2,cacheIdx2,maxShared);
		
		// write state of join node to edge,then update ancestral
		cacheIdx2=node2cache[joinIdx2];
		nodeStateOutputKernel<<<2,BLOCKSIZE>>>(
			cacheMat+nbin*cacheIdx1,cacheBistate+nbin*cacheIdx1,nbin,cacheTmp
		);
		cudaMemcpy(nodeState+(nodeCount-ntip)*2*nbin,cacheTmp,nbin*sizeof(u32),cudaMemcpyDeviceToHost);
		nodeStateOutputKernel<<<2,BLOCKSIZE>>>(
			cacheMat+nbin*cacheIdx2,cacheBistate+nbin*cacheIdx2,nbin,cacheTmp
		);		
		cudaMemcpy(nodeState+((nodeCount-ntip)*2+1)*nbin,cacheTmp,nbin*sizeof(u32),cudaMemcpyDeviceToHost);
		if(ancestralType==0){
			updateAncestralKernel<<<2,BLOCKSIZE>>>(
				cacheMat+nbin*cacheIdx1,cacheBistate+nbin*cacheIdx1,
				cacheMat+nbin*cacheIdx2,cacheBistate+nbin*cacheIdx2,
				nbin
			);			
		}else{
			updateAncestralKernel2<<<2,BLOCKSIZE>>>(
				cacheMat+nbin*cacheIdx1,cacheBistate+nbin*cacheIdx1,
				cacheMat+nbin*cacheIdx2,cacheBistate+nbin*cacheIdx2,
				nbin
			);
		}

		
		// write new branches to edge
		edge[(nodeCount-ntip)*4]=nodeCount;
		edge[(nodeCount-ntip)*4+1]=joinIdx1;
		edge[(nodeCount-ntip)*4+2]=nodeCount;
		edge[(nodeCount-ntip)*4+3]=joinIdx2;	
	
		// update index vector
		node2cache[joinIdx1]=ntip; //mark joinIdx1 as used
		node2cache[joinIdx2]=ntip; //mark joinIdx2 as used
		node2cache[nodeCount]=cacheIdx1; // link new node to cacheIdx1
		cudaMemcpy(cache2node+cacheIdx1,&nodeCount,sizeof(u32),cudaMemcpyHostToDevice);//link cacheIdx1 to new node
		cudaMemcpy(cache2node+cacheIdx2,&nnode,sizeof(u32),cudaMemcpyHostToDevice);//mark cacheIdx2 as invalid
		
		// mark cacheIdx1 and cacheIdx2 in localMax as invalid
		maxShared=-1;
		
		// update validCacheIndices
		nValidCache--;
		u32 ptr;
		for(ptr=0;validCacheIndices[ptr]<cacheIdx2;ptr++){
			;
		}
		for(;ptr<nValidCache;ptr++){
			validCacheIndices[ptr]=validCacheIndices[ptr+1];
		}
		for(u32 i=0;i<nValidCache;i++){
			u32 iTmp=validCacheIndices[i];
			cacheShared[iTmp][cacheIdx2]=-1;
		}
		free(cacheShared[cacheIdx2]);
		cacheShared[cacheIdx2]=NULL;

		// copy the mutation profile of ancestral from GPU to host and scan for mutation
		cudaMemcpy(oneLine,cacheMat+nbin*cacheIdx1,sizeof(u32)*nbin,cudaMemcpyDeviceToHost);
		u32 nMutate=0;
		for(u32 j=0;j<nbin;j++){
			u32 tmp=oneLine[j];
			for(u32 k=0;k<WIDTH;k++){
				if(((tmp>>k)&1)==1){
					mutateIdx[nMutate]=j*WIDTH+k;
					weightRun[nMutate]=weight[j*WIDTH+k];
					nMutate++;
				}
			}
			
		}
		// copy the indices of mutated sites and weight to GPU
		cudaMemcpyToSymbol(mutateIdxConst,mutateIdx,sizeof(u32)*nMutate);
		cudaMemcpyToSymbol(weightConst,weightRun,sizeof(u32)*nMutate);
		// copy the indices of rows for calculation to GPU
		u32 BlockN=(nValidCache-1)/BLOCKSIZE+1;
		cudaMemcpy(rowIndices_gpu,validCacheIndices,sizeof(u32)*nValidCache,cudaMemcpyHostToDevice);
		// calcuate shared
		fillFloat<<<1,BLOCKSIZE>>>(cacheShared_gpu,ntip,-1);
		calculateSharedKernel<<<BlockN,BLOCKSIZE>>>(
			nMutate,cacheMat,nbin,
			rowIndices_gpu,nValidCache,
			cacheShared_gpu
		); 
		cudaMemcpy(cacheShared[cacheIdx1],cacheShared_gpu,sizeof(float)*ntip,cudaMemcpyDeviceToHost);
		cudaMemcpy(localMaxHost,localMax,sizeof(float)*ntip,cudaMemcpyDeviceToHost);
		for(u32 i=0;i<nValidCache;i++){
			u32 iTmp=validCacheIndices[i];
			cacheShared[iTmp][cacheIdx1]=cacheShared[cacheIdx1][iTmp];
			cacheShared[cacheIdx1][iTmp]=-1;
			if(cacheShared[iTmp][cacheIdx1]>localMaxHost[iTmp]){
				localMaxHost[iTmp]=cacheShared[iTmp][cacheIdx1];
				localMaxIdx[iTmp]=nodeCount;
			}
		}
		cacheShared[cacheIdx1][cacheIdx1]=-1;
		localMaxHost[cacheIdx1]=-1;
		localMaxHost[cacheIdx2]=-1;
		cudaMemcpy(localMax,localMaxHost,sizeof(float)*ntip,cudaMemcpyHostToDevice);

		nodeCount++;
	}
	free(cacheShared[node2cache[nodeCount-1]]);
	cacheShared[node2cache[nodeCount-1]]=NULL;

	u32 cacheIdx1=node2cache[nodeCount-1];
	nodeStateOutputKernel<<<2,BLOCKSIZE>>>(
		cacheMat+nbin*cacheIdx1,cacheBistate+nbin*cacheIdx1,nbin,cacheTmp
	);
	cudaMemcpy(nodeState+(nnode-1)*nbin,cacheTmp,nbin*sizeof(u32),cudaMemcpyDeviceToHost);
//	memcpy(nodeState+(nnode-1)*nbin,oneLine,sizeof(u32)*nbin);
	free(cacheShared);
	cudaFree(cacheShared_gpu);
	
	cudaFreeHost(oneLine);
	cudaFree(cacheMat);
	cudaFree(cacheBistate);
	cudaFree(cacheTmp);
	free(weight);
	cudaFreeHost(weightRun);
	cudaFreeHost(mutateIdx);	
	cudaFree(localMax);
	cudaFreeHost(localMaxHost);
	free(localMaxIdx);
	free(node2cache);
	cudaFree(cache2node);
	free(validCacheIndices);
	cudaFree(maxValues_gpu);
	cudaFree(maxIndices_gpu);
	cudaFreeHost(maxValues);
	cudaFreeHost(maxIndices);
	cudaFree(rowIndices_gpu);

/////////////////////////////////////////////////////////////

	u32* edge_gpu;
	cudaMalloc((void**)&edge_gpu,sizeof(u32)*nedge*2);
	cudaMemcpy(edge_gpu,edge, sizeof(u32)*nedge*2, cudaMemcpyHostToDevice);
	
	const u32 root=nedge; //ntip=(nedge+2)/2
	u32* nodeRelabel=(u32*)(malloc(sizeof(u32)*nnode));
	for(u32 i=0;i<ntip;i++){
		nodeRelabel[i]=i;
	}
	for(u32 i=ntip;i<nnode;i++){
		nodeRelabel[i]=nnode;
	}
	nodeCount=ntip;
	u32 idx=0;
	u32* idx_gpu;
	cudaMalloc((void**)&idx_gpu,sizeof(u32));	
	u32* ances_stack=(u32*)(malloc(sizeof(u32)*(nedge+1)));
	ances_stack[0]=root;
	int stack_pointer=0;

	FILE *outputEdge;
	char output_file[256];
	strcpy(output_file,argv[1]);
	strcat(output_file,".edge2");
	outputEdge = fopen(output_file, "wb");
	dim[0]=nedge;
	dim[1]=2;
	fwrite(dim, sizeof(u32), 2, outputEdge);

	FILE *outputNode;
	strcpy(output_file,argv[1]);
	strcat(output_file,".nodeState2");
	outputNode = fopen(output_file, "wb");
	dim[0]=ntip-1;
	dim[1]=nbin;
	fwrite(dim, sizeof(u32), 2, outputNode);
	fwrite(nodeState+(nnode-1)*nbin,sizeof(u32),nbin,outputNode);

	while(stack_pointer>(-1)){
		findValueKernel<<<1,BLOCKSIZE>>>(edge_gpu,nedge,ances_stack[stack_pointer],idx_gpu); 
		cudaMemcpy(&idx,idx_gpu,sizeof(u32),cudaMemcpyDeviceToHost);
		if(idx>=nedge){
			stack_pointer--;
		}else{
			stack_pointer++;
			ances_stack[stack_pointer]=edge[idx*2+1];
			u32 tmp[2];
			tmp[0]=edge[idx*2];tmp[1]=edge[idx*2+1];
			
			if(nodeRelabel[tmp[0]]==nnode){
				nodeRelabel[tmp[0]]=nodeCount;
				tmp[0]=nodeCount;
				nodeCount++;
			}else{
				tmp[0]=nodeRelabel[tmp[0]];
			}
			if(nodeRelabel[tmp[1]]==nnode){
				findValueKernel2<<<1,BLOCKSIZE>>>(edge_gpu,nedge,tmp[1],idx_gpu); 
				cudaMemcpy(&idx,idx_gpu,sizeof(u32),cudaMemcpyDeviceToHost);
//				printf("%u,%u\n",tmp[1],idx+1);
				fwrite(nodeState+idx*nbin,sizeof(u32),nbin,outputNode);
				
				nodeRelabel[tmp[1]]=nodeCount;
				tmp[1]=nodeCount;
				nodeCount++;
			}else{
				tmp[1]=nodeRelabel[tmp[1]];
			}
			fwrite(tmp,sizeof(u32),2,outputEdge);
		}
	}
//	printf("\n");
	fclose(outputEdge);
	fclose(outputNode);
	
	cudaFree(edge_gpu);
	free(edge);
	cudaFreeHost(nodeState);
	free(nodeRelabel);
	cudaFree(idx_gpu);
	free(ances_stack);


	clock_t end=clock();
	double time_taken;
	time_taken = ((double) (end - start)) / CLOCKS_PER_SEC;
	printf("joinEdgeBit: %.2fs elapsed for %u tips.\n",time_taken,ntip);
	return 0; 
}

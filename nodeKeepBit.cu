#include "cuda_runtime.h"
#include <stdio.h>
#include <stdlib.h>

#define BLOCKSIZE 128
#define CACHESIZE 128
#define MAXMUTATE (1024*1024)
#define WIDTH 32
#define LOG2WIDTH 5

typedef unsigned int u32;

__global__ void mutationNestedKernel(u32* mutationIdx,u32 nMutate,u32* mat,u32 nrow,u32 ncol,u32* nFound) {

	u32 tid=threadIdx.x+blockIdx.x*blockDim.x;
	u32 localSum=0;

	if(tid<nrow) {
		for(u32 i=0; i<nMutate; i++) {
			u32 tmpIdx=mutationIdx[i];
			u32 xx=tmpIdx>>LOG2WIDTH;
			u32 yy=tmpIdx&(WIDTH-1);
			u32 tmp=mat[tid*ncol+xx];
			if(((tmp>>yy)&1)==0) {
				localSum++;
			}
		}
		nFound[tid]=localSum;
	}

}

__host__ void int2idx(u32* const ve,u32 const size,u32* const output,u32* nIdx) {
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
		for(u32 j=0; j<WIDTH; j++) {
			if(((val>>j)&1)==1) {
				output[k]=i*WIDTH+j;
				k++;
			}
		}
	}
	*nIdx=k;
}


int main(int argc, char *argv[]) {
	setbuf(stdout, NULL);
	clock_t start = clock();

	u32 cutoff=atoi(argv[2]);
	char input_file[256];
	strcpy(input_file,argv[1]);
	strcat(input_file,".nodeState2");
	u32 dim[2];
	FILE *file=fopen(input_file, "rb");
	if (file == NULL) {
		printf("No nodeState input.\n");
		return 0;
	}
	fread(dim, sizeof(u32), 2, file);
	const u32 nnode=dim[0];
	const u32 nbin=dim[1];
	const u32 ntip=nnode+1;

	u32* nodeState=(u32*)(malloc(sizeof(u32)*nbin*nnode));
	fread(nodeState,sizeof(u32),nbin*nnode,file);
	fclose(file);

	strcpy(input_file,argv[1]);
	strcat(input_file,".edge2");
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

	int* state=(int*)(malloc(sizeof(int)*(ntip+nnode)));
	for(u32 i=0; i<ntip; i++) {
		state[i]=1;
	}
	for(u32 i=ntip; i<ntip+nnode; i++) {
		state[i]=-1;
	}

	strcpy(input_file,argv[1]);
	strcat(input_file,".tboutmat");
	file=fopen(input_file, "rb");
	if (file == NULL) {
		printf("No outmat input.\n");
		return 0;
	}
	fread(dim, sizeof(u32), 2, file);
	const u32 nRetain=dim[0];

	u32* outmat;
	cudaHostAlloc((void**)&outmat,sizeof(u32)*nbin*nRetain,cudaHostAllocDefault);
	u32* outmat_gpu;
	cudaMalloc((void**)&outmat_gpu,sizeof(u32)*nbin*nRetain);

	fread(outmat,sizeof(u32),nbin*nRetain,file);
	cudaMemcpy(outmat_gpu,outmat,sizeof(u32)*nbin*nRetain,cudaMemcpyHostToDevice);
	fclose(file);
	cudaFreeHost(outmat);

	u32* nFound=(u32*)(malloc(sizeof(u32)*nRetain));
	u32* nFound_gpu;
	cudaMalloc((void**)&nFound_gpu,sizeof(u32)*nRetain);

	u32* mutateIdx;
	cudaHostAlloc((void**)&mutateIdx,sizeof(u32)*MAXMUTATE,cudaHostAllocDefault);
	u32* mutateIdx_gpu;
	cudaMalloc((void**)&mutateIdx_gpu,sizeof(u32)*MAXMUTATE);

	char output_file[256];
	strcpy(output_file,argv[1]);
	strcat(output_file,".retainNode");
	FILE* output=fopen(output_file,"w");

	u32 currentNode=ntip-1;
	u32 flag=0;
	for(u32 i=0; i<nedge; i++) {
		if(edge[i*2+1]+1>ntip) {
			if(flag==1&&currentNode<=edge[i*2]) { //still within a kept node (currentNode)
				continue;
			}
//			flag=nodeFilterBit(outmat,node_mat.column(edge(i,1)-1-Ntip),keep_cutoff);
			u32 nMutate=0;
			int2idx(nodeState+(edge[i*2+1]-ntip)*nbin,nbin,mutateIdx,&nMutate);
			cudaMemcpy(mutateIdx_gpu,mutateIdx,sizeof(u32)*nMutate,cudaMemcpyHostToDevice);
			mutationNestedKernel<<<(nRetain-1)/BLOCKSIZE+1,BLOCKSIZE>>>(mutateIdx_gpu,nMutate,outmat_gpu,nRetain,nbin,nFound_gpu);
			cudaMemcpy(nFound,nFound_gpu,sizeof(u32)*nRetain,cudaMemcpyDeviceToHost);
			flag=1; //first mark the node as kept
			for(u32 j=0; j<nRetain; j++) {
				if(nFound[j]<cutoff) { //the node has not sufficient unique mutations
					flag=0; //mark the node as collapsed
					break;
				}
			}
			if(flag==1) {
				currentNode=edge[i*2+1];
				fprintf(output,"%u\n",currentNode);
			}
		}
	}
//	printf("\n");
	fclose(output);

	free(nFound);
	cudaFree(nFound_gpu);
	cudaFreeHost(mutateIdx);
	cudaFree(mutateIdx_gpu);
	free(edge);
	free(state);
	cudaFree(outmat_gpu);
	free(nodeState);

	clock_t end=clock();
	double time_taken;
	time_taken = ((double) (end - start)) / CLOCKS_PER_SEC;
	printf("nodeKeepBit: %.2fs elapsed for %u nodes and %u outgroups.\n",time_taken,nnode,nRetain);

	return 0;
}

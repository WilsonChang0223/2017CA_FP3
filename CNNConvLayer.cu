// This program executes a typical convolutional layer in regular CNNs.Neuron sparsity(zero ratio) is 50% and Weight sparsity is 70%.
#include <iostream>
#include "CNNConvLayer.h"
using namespace std;

#define threadperblock 4

int *filtCooNNZ_GPU;
int *filtCooData_GPU;
int *filtCooRow_GPU;
int *filtCooCol_GPU;

int *inNeuCooNNZ_GPU;
int *inNeuCooData_GPU;
int *inNeuCooRow_GPU;
int *inNeuCooCol_GPU;

int *out_GPU;
// This is the CPU version, please don't modify it
void convLayerCPU()
{
	// declarations for bunch of indexing parameters
	int fn, sli, fmy, fmx, y, x;
	int ifmy, ifmx, ofmy, ofmx;
	int filtIdx, inNeuIdx, outNeuIdx, outIdx;
	int filtVol  = FMDEPTH  * FILTSIZE * FILTSIZE;
	int fmArea   = FMSIZE   * FMSIZE;
	int filtArea = FILTSIZE * FILTSIZE;
	int outArea  = FMSIZE/2 * FMSIZE/2;
	int sum;
	// Convolution
	for(fn = 0; fn < FILTNUM; fn++){
		for(fmy = 0; fmy < FMSIZE; fmy += STRIDE){
			for(fmx = 0; fmx < FMSIZE; fmx += STRIDE){
				sum = 0;
				for(sli = 0; sli < FMDEPTH; sli++){
					for(y = 0; y < FILTSIZE; y++){
						for(x = 0; x < FILTSIZE; x++){
							ifmy = fmy - FILTSIZE / 2 + y;
							ifmx = fmx - FILTSIZE / 2 + x;
							filtIdx = fn*filtVol + sli*filtArea + y*FILTSIZE + x;
							inNeuIdx = sli*fmArea + ifmy*FMSIZE + ifmx;
							if(ifmy >= 0 && ifmy < FMSIZE && ifmx >= 0 && ifmx < FMSIZE)
								sum += filt[filtIdx] * inNeu[inNeuIdx];
						}
					}
				}
				// Activation - ReLU
				outNeuIdx = fn*fmArea + fmy*FMSIZE + fmx;
				if(sum <= 0)
					outNeu[outNeuIdx] = 0;
				else
					outNeu[outNeuIdx] = sum;
			}
		}
	}

	// Max Pooling with Window Size 2x2 and stride 2
	int max, tmpVal;
	for(sli = 0; sli < FILTNUM; sli++){
		for(fmy = 0; fmy < FMSIZE/2 ; fmy += 1){
			for(fmx = 0; fmx < FMSIZE/2 ; fmx += 1){
				outNeuIdx = sli*fmArea + fmy*2*FMSIZE + fmx*2;
				max = outNeu[outNeuIdx];
				for(y = 0; y < 2; y++){
					for(x = 0; x < 2; x++){
						ofmy = fmy*2 + y;
						ofmx = fmx*2 + x;
						outNeuIdx = sli*fmArea + ofmy*FMSIZE + ofmx;
						tmpVal = outNeu[outNeuIdx];	
						if(tmpVal > max)
							max = tmpVal;
					}
				}
				outIdx = sli*outArea + fmy*FMSIZE/2 + fmx;
				outCPU[outIdx] = max;
			}
		}
	}
}
void init_Mem_GPU(){
	cudaMalloc(&filtCooNNZ_GPU , FILTNUM*FMDEPTH* sizeof(int));
	cudaMalloc(&filtCooData_GPU, filtCooSize    * sizeof(int)); 
	cudaMalloc(&filtCooRow_GPU , filtCooSize    * sizeof(int));
	cudaMalloc(&filtCooCol_GPU , filtCooSize    * sizeof(int));

	cudaMalloc(&inNeuCooNNZ_GPU , FMDEPTH     * sizeof(int));
	cudaMalloc(&inNeuCooData_GPU, inNeuCooSize* sizeof(int));
	cudaMalloc(&inNeuCooRow_GPU , inNeuCooSize* sizeof(int));
	cudaMalloc(&inNeuCooCol_GPU , inNeuCooSize* sizeof(int));

	cudaMalloc(&out_GPU         , FMSIZE/2*FMSIZE/2*FILTNUM*sizeof(int));

	cudaMemcpy(filtCooNNZ_GPU , filtCooNNZ , FILTNUM*FMDEPTH *sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(filtCooData_GPU, filtCooData, filtCooSize     *sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(filtCooRow_GPU , filtCooRow , filtCooSize     *sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(filtCooCol_GPU , filtCooCol , filtCooSize     *sizeof(int), cudaMemcpyHostToDevice);

	cudaMemcpy(inNeuCooNNZ_GPU , inNeuCooNNZ , FMDEPTH      *sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(inNeuCooData_GPU, inNeuCooData, inNeuCooSize *sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(inNeuCooRow_GPU , inNeuCooRow , inNeuCooSize *sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(inNeuCooCol_GPU , inNeuCooCol , inNeuCooSize *sizeof(int), cudaMemcpyHostToDevice);
}
/***	Implement your CUDA Kernel here	***/
__device__ int tmp_out_dev[FMSIZE*FMSIZE*FILTNUM];

__global__ 
void convLayerGPU(int *filtCooNNZ_dev,int *filtCooData_dev,int *filtCooRow_dev,int *filtCooCol_dev,int *inNeuCooNNZ_dev,int *inNeuCooData_dev,int *inNeuCooRow_dev,int *inNeuCooCol_dev){
	// declarations for bunch of indexing parameters
	int fmArea = FMSIZE* FMSIZE;

	int i = blockIdx.x*blockDim.x + threadIdx.x;

	for (int Idx = threadIdx.x; Idx < fmArea; Idx += blockDim.x)
		tmp_out_dev[Idx+blockIdx.x*fmArea] = 0;

	int FmSizeAccu, FiltSizeAccu;
	int FmSizeAccu_p, FiltSizeAccu_p;
	int FmSize, FiltSize;

	int blockNum   = i/threadperblock;
	int FmDepthIdx = blockNum%FMDEPTH;

	//Start initial Fm Fi Size//
	FmSizeAccu   = inNeuCooNNZ_dev[FmDepthIdx];	
	if (FmDepthIdx == 0)
		FmSizeAccu_p = 0;
	else
		FmSizeAccu_p = inNeuCooNNZ_dev[FmDepthIdx-1];
        FmSize = FmSizeAccu - FmSizeAccu_p;

	FiltSizeAccu = filtCooNNZ_dev[blockNum];
	if (blockNum == 0)
		FiltSizeAccu_p = 0;
	else
		FiltSizeAccu_p = filtCooNNZ_dev[blockNum-1];	
        FiltSize = FiltSizeAccu - FiltSizeAccu_p;	
	//End initial Fm Fi Size//

	//Start COO format Conv//
	for (int Idx = i%threadperblock; Idx < FmSize*FiltSize; Idx += threadperblock){
		int NeuIdx, NeuRow, NeuCol;
		int FiltIdx, FiltRow, FiltCol;
		int OutRow, OutCol, OutDepth;
		NeuIdx  = Idx%FmSize + FmSizeAccu_p;
        	FiltIdx = Idx/FmSize + FiltSizeAccu_p;
	
		NeuRow  = inNeuCooRow_dev[NeuIdx];
		NeuCol  = inNeuCooCol_dev[NeuIdx];
		FiltRow = filtCooRow_dev[FiltIdx];
        	FiltCol = filtCooCol_dev[FiltIdx];
	
		OutDepth = blockNum/FMDEPTH;
		OutRow   = NeuRow + (1 - FiltRow); 
        	OutCol   = NeuCol + (1 - FiltCol);
	
		if (OutRow < 0 || OutCol < 0 || OutRow >= FMSIZE || OutCol >= FMSIZE)
        		continue;
	
		int tmp = filtCooData_dev[FiltIdx] *inNeuCooData_dev[NeuIdx];	
		int index = OutDepth* fmArea + OutRow* FMSIZE + OutCol;
		atomicAdd(&tmp_out_dev[index], tmp);	
      	}	
	//Ende COO format Conv//
}

__global__ 
void MaxPoolGPU(int *out_dev){
//	int i = threadIdx.x + blockIdx.x* blockDim.x;
//	out_dev[i] = tmp_out_dev[i];
	__shared__ int tmp[FMSIZE*FMSIZE];
	tmp[threadIdx.x] = tmp_out_dev[threadIdx.x + blockIdx.x* FMSIZE*FMSIZE];

	if (threadIdx.x%2 == 0){
		if (tmp[threadIdx.x] < tmp[threadIdx.x+1])
			tmp[threadIdx.x] = tmp[threadIdx.x+1];
	}
	__syncthreads();
	
	int max;
	if (threadIdx.x%2 == 0){
		if (threadIdx.x%(2*FMSIZE) < FMSIZE){
			int A = tmp[threadIdx.x];
			int B = tmp[threadIdx.x+FMSIZE];
			if (A > B)
				max = A;
			else
				max = B;
			int fmx = (threadIdx.x)%FMSIZE/2;
			int fmy = threadIdx.x/(2*FMSIZE);
			int outIdx = fmx + fmy*FMSIZE/2  + blockIdx.x* FMSIZE/2* FMSIZE/2;
			if (max < 0)
				max = 0;
			out_dev[outIdx] = max;
		}
	}
	
}

int main()
{
	//variables setting and loading input data
	timespec time_begin, time_end; 
	int convLayerCPUExecTime, convLayerGPUExecTime;
	init();
	initCoo();

	//Convolution by CPU                                                
	clock_gettime(CLOCK_REALTIME, &time_begin);
	convLayerCPU();
	clock_gettime(CLOCK_REALTIME, &time_end);
	convLayerCPUExecTime = timespec_diff_us(time_begin, time_end);
	cout << "CPU time for executing a typical convolutional layer = "  <<  ((float)convLayerCPUExecTime)/1000 << "ms" << endl;
	
	//Convolution by GPU   
	clock_gettime(CLOCK_REALTIME, &time_begin);
	init_Mem_GPU();
	/***	Lunch your CUDA Kernel here	***/
	//Convolution//
	convLayerGPU<<<FILTNUM, threadperblock*FMDEPTH>>>(filtCooNNZ_GPU,filtCooData_GPU,filtCooRow_GPU,filtCooCol_GPU,inNeuCooNNZ_GPU,inNeuCooData_GPU,inNeuCooRow_GPU,inNeuCooCol_GPU); // Lunch the kernel
	cudaDeviceSynchronize(); // Do synchronization before clock_gettime()
	//MaxPooling//
	MaxPoolGPU<<<FILTNUM, FMSIZE*FMSIZE>>>(out_GPU);
	cudaDeviceSynchronize();

	/***	Lunch your CUDA Kernel here	***/
	clock_gettime(CLOCK_REALTIME, &time_end);
	convLayerGPUExecTime = timespec_diff_us(time_begin, time_end);
	cout << "GPU time for executing a typical convolutional layer = "  << ((float)convLayerGPUExecTime)/1000 << "ms" << endl;
	cudaMemcpy(outGPU, out_GPU, FILTNUM * FMSIZE/2 * FMSIZE/2*sizeof(int), cudaMemcpyDeviceToHost);	
	
	//check the anser from CPU and from GPU
	if(checker()){
		cout << "Congratulations! You pass the check." << endl;
		cout << "Speedup: " << (float)convLayerCPUExecTime / convLayerGPUExecTime << endl;
	}
	else
		cout << "Sorry! Your result is wrong." << endl;

	/******** Added ********/
	cudaFree(&filtCooNNZ_GPU );
	cudaFree(&filtCooData_GPU);
	cudaFree(&filtCooRow_GPU );
	cudaFree(&filtCooCol_GPU );
	cudaFree(&inNeuCooNNZ_GPU );
	cudaFree(&inNeuCooData_GPU);
	cudaFree(&inNeuCooRow_GPU );
	cudaFree(&inNeuCooCol_GPU );
	cudaFree(&out_GPU     );
	/******** Added ********/

	//release memory space
	ending();
	
	return 0;
}

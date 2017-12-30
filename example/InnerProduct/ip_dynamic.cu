#include <stdio.h>
#include <sys/time.h>
#include <iostream>
#include <cstdlib>
using namespace std;

#define VECNUM 50000
#define VECLEN 1000
#define HALFNUM 25000000

int *inputA, *inputB;
int *devInputA,  *devInputB, *devOut, *devOut2;
int *outCPU, *outGPU;

void init()
{
	delete outGPU;
	int i, j, idx;

	inputA = new int[VECNUM * VECLEN];
	inputB = new int[VECNUM * VECLEN];


	for(i = 0; i < VECNUM; i++){
		for(j = 0; j < VECLEN; j++){
			idx = i*VECLEN + j;
			if(idx<HALFNUM){
				if(idx%2==0){ //if(idx=even number) =>set value=1 
					inputA[idx] =1;
					inputB[idx] =1;
				}
				else{         //if(idx=odd number) =>set value=0
					inputA[idx] =0;
					inputB[idx] =0;
				}
			}
			else{
				inputA[idx] =3;
				inputB[idx] =3;
			}
			
			
		}
	}

	outCPU = new int[VECNUM]();
	outGPU = new int[VECNUM]();
	
}

void initGPU()
{
	int inputSize = sizeof(int)*VECNUM*VECLEN;

	cudaMalloc(&devInputA, inputSize);	
	cudaMalloc(&devInputB, inputSize);
	cudaMalloc(&devOut, sizeof(int)*VECNUM/2);
    cudaMalloc(&devOut2, sizeof(int)*VECNUM/2);
	
	cudaMemcpy(devInputA, inputA, inputSize, cudaMemcpyHostToDevice);
	cudaMemcpy(devInputB, inputB, inputSize, cudaMemcpyHostToDevice);
	
}
void innerProductCPU()
{
	int i, j, acc, idx;

	for(i = 0; i < VECNUM; i++){
		acc = 0;
		for(j = 0; j < VECLEN; j++){
			idx = i*VECLEN + j;
			acc += inputA[idx] * inputB[idx];
		}
		outCPU[i] = acc;
	}
}

//GPU dynamic version
__global__ void innerProductGPU_firsthalf(int *A, int *B, int *out)
{
	int y = blockIdx.x;
	int x = threadIdx.x;
	int idx = y * VECLEN + x*2;//just only compute the data that its idx is even number
	__shared__ int tmp[VECLEN];
	tmp[x*2] = A[idx] * B[idx];
	
	__syncthreads();

	if(x == 0){
		int i, sum = 0;
		for(i = 0; i < VECLEN/2; i++)
			sum += tmp[i*2];
		out[y] = sum;
	}
}

__global__ void innerProductGPU_secondhalf(int *A, int *B, int *out)
{
	int y = blockIdx.x;
	int x = threadIdx.x;
	int idx = y * VECLEN + x;
	__shared__ int tmp[VECLEN];
	tmp[x] = A[idx] * B[idx];
	
	__syncthreads();

	if(x == 0){
		int i, sum = 0;
		for(i = 0; i < VECLEN; i++)
			sum += tmp[i];
		out[y] = sum;
	}
}


__global__ void innerProductGPU_dynpar(int *A, int *B, int *out,int *A2, int *B2, int *out2) {
    
	dim3 threadsPerBlock_firsthalf(VECLEN/2);
	dim3 threadsPerBlock_secondhalf(VECLEN);
	dim3 numBlocks_firsthalf(VECNUM/2);
	dim3 numBlocks_secondhalf(VECNUM/2);
	
	if(threadIdx.x == 0)
		innerProductGPU_firsthalf<<<numBlocks_firsthalf,threadsPerBlock_firsthalf>>>(A,B,out);
	else if (threadIdx.x == 1)
		innerProductGPU_secondhalf<<<numBlocks_secondhalf,threadsPerBlock_secondhalf>>>(A2,B2,out2);

    cudaDeviceSynchronize();
	
}

//other function
bool checker(){
	int i;

	for(i = 0; i < VECNUM; i++){ 
		if(outCPU[i] != outGPU[i]){
			cout << "The element: " << i << " is wrong!\n";
			cout << "outCPU[" << i << "] = " << outCPU[i] << endl;
			cout << "outGPU[" << i << "] = " << outGPU[i] << endl;
			return false;
		}
	}

	return true;
}

int timespec_diff_us(timespec& t1, timespec& t2)
{                                                                                
  return (t2.tv_sec - t1.tv_sec) * 1e6 + (t2.tv_nsec - t1.tv_nsec) / 1e3;        
} 


int main(int argc, char** argv) 
{
	int outSize = sizeof(int)*VECNUM;
  	init();
	initGPU();	
	timespec time_begin, time_end;   
	
	//CPU version
	clock_gettime(CLOCK_REALTIME, &time_begin);
	innerProductCPU();
	clock_gettime(CLOCK_REALTIME, &time_end);
	// cout << "CPU time for executing inner-product = " << timespec_diff_us(time_begin, time_end) / 1000 << "ms" << endl;
		
	// GPU dynamic version
	dim3 threadsPerBlock(2);
	dim3 numBlocks(1);
	
	clock_gettime(CLOCK_REALTIME, &time_begin);
	innerProductGPU_dynpar<<<numBlocks,threadsPerBlock>>>(devInputA, devInputB, devOut,devInputA+HALFNUM, devInputB+HALFNUM, devOut2);
	cudaDeviceSynchronize();
	clock_gettime(CLOCK_REALTIME, &time_end);
	cout << "GPU time for executing dynamic inner-product = " << timespec_diff_us(time_begin, time_end)  << "us" << endl;

	//data copy from GPU to CPU
	cudaMemcpy(outGPU, devOut, outSize/2, cudaMemcpyDeviceToHost);
	cudaMemcpy(outGPU+25000, devOut2, outSize/2, cudaMemcpyDeviceToHost);
	
	//check
	if(checker())
		cout << "Congratulations! You pass the check." << endl;
	else
		cout << "Sorry! Your result is wrong." << endl;
	
	
	//releas space
    cudaFree(&devInputA);
	cudaFree(&devInputB);
	cudaFree(&devOut);
	cudaFree(&devOut2);	
	delete outGPU;
	
	return 0;
	
}

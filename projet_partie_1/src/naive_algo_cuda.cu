/******************************
 * Fichier naive_algo_cuda.cu *
 ******************************/
#include <stdlib.h>
#include <stdio.h>

#include <sys/time.h>
#include <cuda.h>
#include "../inc/utils.h"


/** 
 * CUDA error control and debugging. 
 */
#ifdef CUDA_DEBUG
#define CUDA_SYNC_ERROR() {						\
    cudaError_t sync_error;						\
    cudaDeviceSynchronize();						\
    sync_error = cudaGetLastError();					\
    if(sync_error != cudaSuccess) {					\
      fprintf(stderr, "[CUDA SYNC ERROR at %s:%d -> %s]\n",		\
	      __FILE__ , __LINE__, cudaGetErrorString(sync_error));	\
      exit(EXIT_FAILURE);						\
    }									\
  }
#else /* #ifdef CUDA_DEBUG */
#define CUDA_SYNC_ERROR()
#endif /* #ifdef CUDA_DEBUG */

#define CUDA_ERROR(cuda_call) {					\
    cudaError_t error = cuda_call;				\
    if(error != cudaSuccess){					\
      fprintf(stderr, "[CUDA ERROR at %s:%d -> %s]\n",		\
	      __FILE__ , __LINE__, cudaGetErrorString(error));	\
      exit(EXIT_FAILURE);					\
    }								\
    CUDA_SYNC_ERROR();						\
  }


/**
 * Function iDivUp()
 * Return integer quotient superior or equal to "a/b"
 * Source : CUDA SDK 4.1
 */
static int iDivUp(int a, int b){
  return ((a % b != 0) ? (a / b + 1) : (a / b));
}


/**
 *
 * Function naive_algo_cuda()
 *
 **/
__global__ void naive_algo_cuda(unsigned long *abs, unsigned long *ord, int n, int l, int h, unsigned long long *S_gpu){

  int a = blockDim.x * blockIdx.x + threadIdx.x;
  int b = blockDim.y * blockIdx.y + threadIdx.y;

  int c = 0, ymin = 0, aux = n/10;
  unsigned long long S_loc; //= 0;

  
  if ((a < n) && (b < n)){
    
    // Compute only when a < b
    if (a < b){
      
      if(b == a+1)
	ymin = h;
      
      else{
       
	ymin = ord[a+1];
	for(c = a+1; c < b; c++){
	  if(ord[c] < ymin)
	    ymin = ord[c];
	} // c loop
      } // else loop
      
      S_loc = (abs[b] - abs[a]) * ymin;
      
      // Maximum on every thread of every blocks (slow)
      atomicMax(S_gpu, S_loc);
      
    } //a<b loop

  } //test bound loop

  return;
}


  /**
 *
 * Function main()
 *
 **/
int main(int argc, char **argv){
 
  double debut=0.0, fin=0.0;
  unsigned long **data, *abs_gpu, *ord_gpu;
  unsigned long long S = 0, *S_gpu;
  int res = 0, i= 0;
  int n = 0, l = 0, h = 0;
  int *n_gpu, *l_gpu, *h_gpu;
  
  if(argc != 2){
    printf("Usage: %s <path_of_data_file>\n", argv[0]);
    return -1;
  }
  char *name = argv[1];

  /* Read parameters */
  res = read_param(name, data, &n, &l, &h);
  if(res != 0){
    printf("read_param :\t ERROR\n");
    return -1;
  }
  
  /* Allocate data table */
  data = (unsigned long **) malloc(2 * sizeof(unsigned long *));
  data[0] = (unsigned long *) malloc(n * sizeof(unsigned long));
  data[1] = (unsigned long *) malloc(n * sizeof(unsigned long));

  
  /* Read coordinates from file */
  res = read_data(name, data, n);
  if(res != 0){
    printf("read_data :\t ERROR\n");
    return -1;
  }
  
  /*printf("\nn=%d l=%d h=%d\n", n, l, h);*/
  /*for(i = 0; i < n; i++){
    printf("%lu,%lu\n", data[i][0], data[i][1]);
    }*/


  /* GPU allocation */
  printf("GPU allocation.\n");
  cudaMalloc((void **)&n_gpu, sizeof(int));
  cudaMalloc((void **)&l_gpu, sizeof(int));
  cudaMalloc((void **)&h_gpu, sizeof(int));
  cudaMalloc((void **)&S_gpu, sizeof(unsigned long long));
  if(n_gpu == NULL || l_gpu == NULL || h_gpu == NULL || S_gpu == NULL)
    printf("Parameters allocation failed\n");


  cudaMalloc((void **)&abs_gpu, n * sizeof(unsigned long));
  cudaMalloc((void **)&ord_gpu, n * sizeof(unsigned long));

  /* CPU -> GPU transfer (synchrones) */
  printf("CPU -> GPU transfer.\n");
  cudaMemcpy(n_gpu, &n, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(l_gpu, &l, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(h_gpu, &h, sizeof(int), cudaMemcpyHostToDevice);

  cudaMemcpy(abs_gpu, data[0], n * sizeof(unsigned long), cudaMemcpyHostToDevice);
  cudaMemcpy(ord_gpu, data[1], n * sizeof(unsigned long), cudaMemcpyHostToDevice);

  cudaMemset(S_gpu, 0, sizeof(unsigned long long));

  
  /* Kernel launching */
  printf("Launching kernel.\n");
  // Using n*n threads but not every one is useful, because of the "i < j" constraint
  dim3 threadsParBloc(32, 32);
  dim3 tailleGrille(iDivUp(n,32), iDivUp(n, 32));

  
  /* Start timing */
  debut = my_gettimeofday();

  /* Do computation:  */
  
  printf("Lauching.\n");
  naive_algo_cuda<<<tailleGrille, threadsParBloc>>>(abs_gpu, ord_gpu, n, l, h, S_gpu);

  printf("Leaving kernel.\n");

  cudaDeviceSynchronize();
  
  /* Recovering maximal area on CPU */
  cudaMemcpy((void *)&S, S_gpu, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
  

  /* End timing */
  fin = my_gettimeofday();
  
  fprintf(stdout, "N = %d\t S = %llu\n", n, S);
  fprintf( stdout, "For n=%d: total computation time in s (with gettimeofday()) :\n",
  n);
  fprintf( stdout, "%g\n",
	   fin - debut);

  
  /* Free */
  printf("\nFreeing and quitting.\n");
  free(data[0]);
  free(data[1]);
  free(data);
  
  cudaFree(l_gpu);
  cudaFree(h_gpu);
  cudaFree(n_gpu);
  cudaFree(S_gpu);

  cudaFree(abs_gpu);
  cudaFree(ord_gpu);
  
  return 0;
}

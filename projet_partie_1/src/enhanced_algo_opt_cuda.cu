/*************************************
 * Fichier enhanced_algo_opt_cuda.cu *
 *************************************/
#include <stdlib.h>
#include <stdio.h>

#include <sys/time.h>
#include "../inc/utils.h"
#include <cuda.h>

// Less than 1000 : no changes; more than 10000 : a bit slower
#define NUM_LOCALS 1000


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
 * Function enhanced_algo_opt_cuda()
 *
 **/
__global__ void enhanced_algo_opt_cuda(unsigned long *abs, unsigned long *ord, int n, int l, int h, unsigned long long *S_gpu, unsigned long long *local_max, int num_locals){

  int a = blockDim.x * blockIdx.x + threadIdx.x;
  int li = a%num_locals;
  int b = 0;

  unsigned long long old_max;
  
  int ymin = 0, aux = n/10;
  unsigned long long S_it = 0, S = 0;

 if ((a < n)){
    
   // Compute only when a < b
   for(b = a+1; b < n; b++){
     
     if(b == a+1)
       ymin = h;

     else if (ymin > ord[b-1])
       ymin = ord[b-1];
       
     //else -- nothing
     // WARNING : no default case
       
     S_it = (abs[b] - abs[a]) * ymin;

     if(S_it > S)
       S = S_it;

   } // for loop
   
   // Optimized with local maximums
   old_max = atomicMax(&local_max[li], S);
      
   if (old_max < S)
     atomicMax(S_gpu, S);
   
 } // test bound loop
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
  unsigned long long S = 0, *S_gpu, *local_max_gpu;

  int num_locals = NUM_LOCALS; // modulo = nombre de maximaux locaux
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

  cudaMalloc((void **)&local_max_gpu, num_locals * sizeof(unsigned long long));

  
  /* CPU -> GPU transfer (synchrones) */
  printf("CPU -> GPU transfer.\n");
  cudaMemcpy(n_gpu, &n, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(l_gpu, &l, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(h_gpu, &h, sizeof(int), cudaMemcpyHostToDevice);

  cudaMemcpy(abs_gpu, data[0], n * sizeof(unsigned long), cudaMemcpyHostToDevice);
  cudaMemcpy(ord_gpu, data[1], n * sizeof(unsigned long), cudaMemcpyHostToDevice);

  cudaMemset(local_max_gpu, 0, num_locals * sizeof(unsigned long long));
  cudaMemset(S_gpu, 0, sizeof(unsigned long long));

  
  /* Kernel launching */
  printf("Launching kernel.\n");
  // Using n*n threads but not every one is useful, because of the "i < j" constraint
  dim3 threadsParBloc(32, 1);
  dim3 tailleGrille(iDivUp(n,32), 1); 
  
  
  /* Start timing */
  debut = my_gettimeofday();
  
  /* Do computation:  */

  printf("Lauching.\n"); 
  enhanced_algo_opt_cuda<<<tailleGrille, threadsParBloc>>>(abs_gpu, ord_gpu, n, l, h, S_gpu, local_max_gpu, num_locals);

  printf("Leaving kernel.\n");

  cudaDeviceSynchronize();

  /* Recopie de l aire maximale sur le CPU */
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
  cudaFree(local_max_gpu);

  return 0;
}

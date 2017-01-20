/*********************************
 * Fichier dpr_cuda.cu *
 *********************************/
#include <stdlib.h>
#include <stdio.h>

#include <math.h>
#include <sys/time.h>
#include <cuda.h>
#include "../inc/utils.h"

//Min nb of points to launch the GPU computation
//1500
#define TRESHOLD_SEQ 90000

//Nb points in each parallel region
//1000
#define SIZE_PARALLEL 50000

/**
 * CUDA error control and debugging.
 **/
#ifdef CUDA_DEBUG
#define CUDA_SYNC_ERROR() {						\
    cudaError_t sync_error;						\
    cudaDeviceSynchronize();						\
    Sync_error = cudaGetLastError();					\
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



__global__ void calcul_min( unsigned long *ord, int ind_start, int ind_end, unsigned long long *ymin, int *ind_min, int size_max_parallel ){

  int a = threadIdx.x;
  int size_tot = (ind_end - ind_start -1);
  printf("size_tot = %d\n", size_tot);
  //On n'effectue pas le calcul aux indices ind_start ni ind_end
  int nb_threads = ceilf((float)size_tot/(float)size_max_parallel);

  //size of region to compute in the current thread
  int size_parallel = ceilf( (float)size_tot/(float)nb_threads );


  //have to be computed before the case of a different size_parallel value
  int ind_start_loc = ind_start + a * size_parallel + 1;
  
  if ( a == (nb_threads - 1) )
    size_parallel = size_tot - (nb_threads - 1) * size_parallel;
  printf("size_parallel = %d\n", size_parallel);
    

  unsigned long min_loc = ord[ind_start_loc];
  int ind_min_loc = ind_start_loc;
  int i = 0;

  //printf("FINDING YMIN\n");
  
  for ( i = ind_start_loc; i < ind_start_loc + size_parallel; i++ ){
    
    //Looking for the lowest ordinate
    if ( ord[i]< min_loc ){
      min_loc = ord[i];
      ind_min_loc = i;
      
    }

  }

  printf("thread %d : min_loc = %llu, ind_min_loc = %d\n", a, min_loc, ind_min_loc);
  printf("YMIN = %llu\n", *ymin);
  atomicMin(ymin, min_loc);
  
  __syncthreads();

  printf("ymin = %llu\n", *ymin);
  if (*ymin == min_loc){
    *ind_min = ind_min_loc;
    printf("thread %d : min_loc = %llu, ind_min_loc = %d\n", a, min_loc, ind_min_loc);
  }
  
  return;
}




/**
 *
 * Function dpr_cuda()
 *
 **/

unsigned long long dpr_cuda(unsigned long **data, int n, int l, unsigned long h, int ind_start, int ind_end){

  printf("BEGIN\n");
  printf("********************H = %d**************\n", h);
  int i = 0;
  
  //ycross min on the whole area, ymin min on the whole area minus the 2 ends
  int ind_min = 0;
 
  unsigned long long crosswise_area = 0, left_area = 0, right_area = 0, result_area = 0, ymin =0;


  
  //Two points left : returns the rectangle defined by the height
  if ( (ind_end - ind_start) == 1 ){
    printf("CAS DE BASE\n");
    return (unsigned long long) (data[0][ind_end]-data[0][ind_start]) * h;
  }

  // No parallel computing if too few points
  if ( (ind_end - ind_start) < TRESHOLD_SEQ ){
    ymin = data[1][ind_start + 1];
    ind_min = ind_start + 1;
    
    printf("FINDING YMIN - SEQUENTIAL\n");
    
    //We don't enter the loop if ind_end - ind_start == 2
    for ( i = ind_start + 2; i < ind_end; i++ ){
      
      //Looking for the lowest ordinate
      if ( data[1][i] < ymin ){
	ymin = data[1][i];
	ind_min = i;
      }
    }
    
  }
  else {
    printf("FINDING YMIN - GPU\n");
    int *ind_min_gpu, *ind_start_gpu, *ind_end_gpu, size_parallel = SIZE_PARALLEL, *size_parallel_gpu;
    unsigned long *ord_gpu;
    unsigned long long *min_gpu;

    
    //INIT GPU PARAMETERS
    /* GPU allocation */
    cudaMalloc((void **)&min_gpu, sizeof(unsigned long long));
    cudaMalloc((void **)&ind_min_gpu, sizeof(int));
    cudaMalloc((void **)&ind_start_gpu, sizeof(int));
    cudaMalloc((void **)&ind_end_gpu, sizeof(int));
    cudaMalloc((void **)&size_parallel_gpu, sizeof(int));
    if(min_gpu == NULL || ind_min_gpu == NULL || ind_start_gpu == NULL || ind_end_gpu == NULL || size_parallel_gpu == NULL)
      printf("Parameters allocation failed\n");
  
    cudaMalloc((void **)&ord_gpu, n * sizeof(unsigned long));
  

    /* CPU -> GPU transfer (synchrones) */
    cudaMemcpy(ord_gpu, data[1], n * sizeof(unsigned long), cudaMemcpyHostToDevice);
    cudaMemcpy(ind_start_gpu, &ind_start, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ind_end_gpu, &ind_end, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(size_parallel_gpu, &size_parallel, sizeof(int), cudaMemcpyHostToDevice);
  
    cudaMemset(min_gpu, h, sizeof(unsigned long long));
    cudaMemset(ind_min_gpu, -1, sizeof(int));
  
    /* Kernel launching */
    printf("Launching kernel.\n");
    
    //Un seul bloc de threads 1D
    int size_tot = (ind_end - ind_start -1);
    int nb_threads = ceil((float)size_tot/(float)SIZE_PARALLEL);
    //printf("nb_threads = %d\n", nb_threads);

    dim3 threadsParBloc(nb_threads, 1);
    dim3 tailleGrille(1, 1);
    printf("********************MIN_GPU = h = %d**************\n", h);
  
    // Compute ymin on GPU
    calcul_min<<<tailleGrille, threadsParBloc>>>(ord_gpu, ind_start, ind_end, min_gpu, ind_min_gpu, size_parallel);

    printf("Leaving kernel.\n");

    /* Recovering min element and index on CPU (element too for testing purposes) */
    cudaMemcpy((void *)&ymin, min_gpu, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaMemcpy((void *)&ind_min, ind_min_gpu, sizeof(int), cudaMemcpyDeviceToHost);


    printf("ind_start = %d\t, ind_min = %d\t, ind_end = %d\n", ind_start, ind_min, ind_end);
    /* cuda Frees */
    cudaFree(min_gpu);
    cudaFree(ind_min_gpu);
    cudaFree(ind_start_gpu);
    cudaFree(ind_end_gpu);
    cudaFree(ord_gpu);

  }
  
  printf("RECURSIVE CALLS\n");
  //printf("ind_start = %d\t, ind_min = %d\t, ind_end = %d\n", ind_start, ind_min, ind_end);
  
  crosswise_area = ymin * (data[0][ind_end] - data[0][ind_start]);

  left_area = dpr_cuda(data, n, l, h, ind_start, ind_min);
  right_area = dpr_cuda(data, n, l, h, ind_min, ind_end);

  //printf("MAX\n");
  
  //Result is the max of these areas
  result_area = crosswise_area;
  if ( left_area > result_area )
    result_area = left_area;
  if ( right_area > result_area )
    result_area = right_area;

  
  //printf("ind_start = %d\t, ind_min = %d\t, ind_end = %d\n left = %llu\t, right = %llu\t, cross = %llu, ext_cross = %llu, result_area = %llu\n\n", ind_start, ind_min, ind_end, left_area, right_area, crosswise_area, ext_crosswise_area, result_area);


  
  return result_area;
  
}

int main(int argc, char **argv){

  double debut=0.0, fin=0.0;
  unsigned long **data;
  unsigned long long S = 0, h = 0;
  int res = 0, i= 0;
  int n = 0, l = 0;

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
  
  /* Start timing */
  debut = my_gettimeofday();

  /* Do computation:  */

  printf("LAUNCHING DPR_CUDA\n");
  
  S = dpr_cuda(data, n, l, h, 0, n-1);
  
  /* End timing */
  fin = my_gettimeofday();
  
  fprintf(stdout, "N = %d\t S = %llu\n", n, S);
  /*fprintf( stdout, "For n=%d: total computation time (with gettimeofday()) : %g s\n\n",
  n, fin - debut);*/
  fprintf( stdout, "%g\n",
	   fin - debut);
  return 0;
}

#include <stdlib.h>
#include <stdio.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <sys/time.h>
#include "../inc/utils.h"


/**
 *
 * Function enhanced_algo()
 *
 **/
unsigned long long enhanced_algo(unsigned long **data, int n, int l, int h){

  // for each (i,j) w/ i<j do
  int a = 0, b = 0, ymin = 0, aux = n/10;
  unsigned long long S = 0, S_ij = 0;
  for(a = 0; a < n; a++){
    for(b = a+1; b < n; b++){
      if(b == a+1)
	ymin = h;

      else if (ymin > data[b-1][1])
	ymin = data[b-1][1];
      /*else
      // do nothing
      } // else loop*/
      S_ij = (data[b][0] - data[a][0]) * ymin; 
      if(S_ij > S)
	S = S_ij;
    } // b loop
    /*if (a%aux == 0)
      printf("%d %%... ", (a*100/n)+10);
    */
  } // a loop
  return S;
}


/**
 *
 * Function enhanced_algo_parallel()
 *
 **/
unsigned long long enhanced_algo_parallel(unsigned long **data, int n, int l, int h){

  // for each (i,j) w/ i<j do
  int a = 0, b = 0, ymin = 0, aux = n/10;
  unsigned long long S = 0, S_ij = 0;
  #pragma omp parallel for shared(S, a) private(b)
  for(a = 0; a < n; a++){
    for(b = a+1; b < n; b++){
      if(b == a+1)
	ymin = h;
      else if (ymin > data[b-1][1])
	ymin = data[b-1][1];
      // else -- nothing
      // WARNING : no default case

      S_ij = (data[b][0] - data[a][0]) * ymin; 
      if(S_ij > S)
	S = S_ij;
    } // b loop
    /*if (a%aux == 0)
      printf("%d %%... ", (a*100/n)+10);
*/
  } // a loop
  return S;
}


/**
 *
 * Function main
 *
 **/
int main(int argc, char **argv){
 
  double debut=0.0, fin=0.0;
  unsigned long **data;
  unsigned long long S = 0;
  int res = 0, i= 0;
  int n = 0, l = 0, h = 0;
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
  data = malloc(n * sizeof(unsigned long *));
  for(i = 0; i < n; i++)
    data[i] = malloc(2 * sizeof(unsigned long));
  
  /* Read coordinates from file */
  res = read_data(name, data, n);
  if(res != 0){
    printf("read_data :\t ERROR\n");
    return -1;
  }
  
  //printf("\nn=%d\tl=%d\th=%d\n", n, l, h);
  /*for(i = 0; i < n; i++){
    printf("%lu,%lu\n", data[i][0], data[i][1]);
    }*/
  
  /* Start timing */
  debut = my_gettimeofday();
  
  /* Do computation:  */
  #ifdef _OPENMP
  #pragma omp parallel shared(n, l, h)
  #pragma omp single
  S = enhanced_algo_parallel(data, n, l, h);
  
  #else
  S = enhanced_algo(data, n, l, h);
  #endif
  
  /* End timing */
  fin = my_gettimeofday();
  
  fprintf(stdout, "N = %d\t S = %llu\n", n, S);
  /*fprintf( stdout, "For n=%d: total computation time (with gettimeofday()) : %g s\n",
    n, fin - debut);*/
  fprintf( stdout, "%g\n", fin - debut);
      
  return 0;
}

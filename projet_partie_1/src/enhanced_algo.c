#include <stdlib.h>
#include <stdio.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <sys/time.h>
#include "../inc/utils.h"


/**
 *
 * Function my_gettimeofday()
 * Used to compute time of execution
 *
 **/
double my_gettimeofday(){
  struct timeval tmp_time;
  gettimeofday(&tmp_time, NULL);
  return tmp_time.tv_sec + (tmp_time.tv_usec * 1.0e-6L);
}


/**
 *
 * Function read_param()
 * "l h
 *  n"
 *
 **/
int read_param(char *name, unsigned long **data, int *n, int *l, int *h){

  FILE* fp = NULL;
  fp = fopen(name, "r");
  if(fp == NULL){
    printf("fopen :\t ERROR\n");
    return -1;
  }
  fscanf(fp, "%u %u", l, h);
  fscanf(fp, "%u", n);
  
  fclose(fp);
  return 0;
}


/**
 *
 * Function read_data()
 * "l h
 *  n
 *  x_0 y_0
 *  ...
 *  x_n y_n"
 *
 **/
int read_data(char *name, unsigned long **data, int n){

  FILE* fp = NULL;
  int i = 0, a = 0, b = 0;
  fp = fopen(name, "r");
  if(fp == NULL){
    printf("fopen :\t ERROR\n");
    return -1;
  }
  /* Ghost reading */
  fscanf(fp, "%u %u", &a, &b);
  fscanf(fp, "%u", &a);
  
  for(i = 0; i < n; i++)
    fscanf(fp, "%lu %lu", &data[i][0], &data[i][1]);
  
  fclose(fp);
  return 0;
}


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
    if (a%aux == 0){
      printf("%d %%\n", (a*100/n)+10);
    }
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

#pragma omp parallel for shared(S)
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
    if (a%aux == 0){
      printf("%d %%\n", (a*100/n)+10);

    }
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
  
  printf("n=%d l=%d h=%d\n", n, l, h);
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
  
  fprintf(stdout, " N = %d\t S = %llu\n", n, S);
  fprintf( stdout, "For n=%d: total computation time (with gettimeofday()) : %g s\n",
	   n, fin - debut);
      
  return 0;
}

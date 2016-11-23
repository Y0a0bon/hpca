#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>


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
    fscanf(fp, "%lu %lu", &data[0][i], &data[1][i]);
  
  fclose(fp);
  return 0;
}

/****************************
 *  Fichier linear_algo.cu  *
 ****************************/
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

#include "../inc/utils.h"
#include "../inc/stack.h"
#include "../inc/linear_algo.h"

struct coord
{
	unsigned long x;
	unsigned long y;
};


/**
 * Function verif_ctr(){}
 **/
int verif_ctr(unsigned long **data, int n){
  int i = 0;
  /* Verify if first point has coordinates (0,0) */
  if(data[0][0] != 0 || data[0][1] != 0)
    return 1;
	
  for(i=0; i<n-1; i++){
    /* Verify each point has a different absciss */
    if(data[i][0] == data[i+1][0]){
      return 1;
    }
  }

  return 0;
}


/**
 * Function linear_algo()
 **/
unsigned long long linear_algo(unsigned long **data, int n, int h, int l){

  int i = 0, j = 0, size = 0; //y_tmp = 0, cursor = 0, cursor_tmp = 0, size = 0;
  unsigned long current_p[] = {0,0}, pred_p[] = {0,0};
  unsigned long long S = 0;
  coord_s coord_tmp; //**heap = malloc(n * sizeof(unsigned long *));
	
	stack_s *st = initialize();
	
  /*printf("linear_algo :\nAllocation\t\t");
  for(i=0; i<n; i++)
    heap[i] = malloc(2 * sizeof(unsigned long));
  printf("OK\n");
  */
	
  for(i=1; i<n; i++){
    current_p[0] = data[i][0];
    current_p[1] = data[i][1];
    pred_p[0] = data[i-1][0];
    pred_p[1] = data[i-1][1];
    printf("%d. Point (%lu,%lu), precede de (%lu,%lu)\n", i, current_p[0],  current_p[1], pred_p[0], pred_p[1]);
    /* Fill the heap */
		/* TODO */
		for(j=size; j<i; j++){
      if(data[j][1] <= pred_p[1]){
					coord_tmp.x = data[j][0];
					coord_tmp.y = data[j][1];
					stack_push(st, coord_tmp);
					size++;
			}
		}
		
		stack_print(st);
		
		
		/*
    for(j=size; j<i; j++){
      if(data[j][1] <= pred_p[1]){
				heap[cursor][0] = data[j][0];
				heap[cursor][1] = data[j][1];
				size++;
      }
    }
		printf("heap : { ");
		for(j=0; j<size; j++)
			printf("(%lu,%lu) ", heap[j][0],heap[j][1]);
    printf("}\n");
    cursor = size-1;
    y_tmp = pred_p[1];
    S = MAX(S, (current_p[0]-pred_p[0])*h);
    printf("Comparaison du rectangle entre les 2 points\n");
    while(cursor != 0){
      if(heap[cursor][1] > current_p[1] && heap[cursor][1] <= y_tmp){
				y_tmp = MIN(y_tmp, heap[cursor][1]);
				//Compute area between current point (right side), 
					 first point of the heap (top side) and the point
					 before in the heap (left side) */
				/*
				cursor_tmp = cursor-1;
				while(heap[cursor_tmp][1] > heap[cursor][1] && cursor_tmp > 0){
					cursor_tmp--;
					//printf("cursor tmp : %d\n", cursor_tmp);
				}
				printf("(%d,%d) On compare S avec le rectange entre (%lu,%lu) et (%lu,%lu)\n", cursor, cursor_tmp, heap[cursor][0], heap[cursor][1], heap[cursor_tmp][0], heap[cursor_tmp][1]);
				S = MAX(S, (current_p[0] - heap[cursor][0])* heap[cursor_tmp][1]);
				//cursor = cursor_tmp;
			}
						cursor--;
			*/
  }

  /*for(i=0; i<n; i++)
    free(heap[i]);
  free(heap);
	*/
  return S;
}


/**
 * Function main()
 **/
int main(int argc, char **argv){

  double debut=0.0, fin=0.0;
  unsigned long **data = NULL;
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

  /* Verify constraints */
  if(verif_ctr(data, n)){
    printf("linear_algo : \t ERROR (constraints not met)\n\n");
    return -1;
  }
  
  /* Start timing */
  debut = my_gettimeofday();

  /* Do computation:  */
  S = linear_algo(data, n, l, h);
  
  /* End timing */
  fin = my_gettimeofday();
  
  fprintf(stdout, "N = %d\t S = %llu\n", n, S);
    fprintf( stdout, "For n=%d: total computation time (with gettimeofday()) : %g s\n\n",
    n, fin - debut);
  fprintf( stdout, "%g\n",
	   fin - debut);
	
  for(i=0; i<n; i++)
    free(data[i]);
  free(data);
	
  return 0;
}

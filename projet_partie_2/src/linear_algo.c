/****************************
 *  Fichier linear_algo.cu  *
 ****************************/
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

#include "../inc/utils.h"
#include "../inc/stack.h"
#include "../inc/linear_algo.h"


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
      printf("Same absciss on x = %lu", data[i][0]);
      return 1;
    }
  }

  return 0;
}


/**
 * Function linear_algo()
 **/
unsigned long long linear_algo(unsigned long **data, int n, int h, int l){

  int i = 0, j = 0, indice = 0, y_tmp = 0, ptmp = 0;
  unsigned long current_p[] = {0,0}, pred_p[] = {0,0};
  unsigned long long S = 0;
  coord_s coord_tmp;
  stack_s *st = initialize(n);

  /* First rectangle of the file */
  S = h*data[1][0];

  for(i=2; i<n-1; i++){
    /* Current point */
    current_p[0] = data[i][0];
    current_p[1] = data[i][1];
    /* Precedent */
    pred_p[0] = data[i-1][0];
    pred_p[1] = data[i-1][1];

    /* Fill the heap */
    if(data[i-2][1] <= pred_p[1]){
      coord_tmp.x = data[i-2][0];
      coord_tmp.y = data[i-2][1];
      stack_push(st, coord_tmp, i-2);
    }
    /* Updating heap */
    indice = stack_top(st);
    while(data[indice][1] > pred_p[1]){
      stack_pop(st);
      indice = stack_top(st);
    }

    /* First rectangle */
    S = MAX(S, (current_p[0]-pred_p[0])*h);

    /* Rectangles behind */
    if(st->size > 0){
      /* First point */
      ptmp = data[st->data_i[st->size-1]][0];
      S = MAX(S, (current_p[0] - ptmp) * pred_p[1]);
      /* Following rectangles */
      for(j=st->size; j >0 ; j--){
	ptmp = data[st->data_i[j-1]][0];
	S = MAX(S, (current_p[0] - ptmp) * data[st->data_i[j]][1]);
      } // j loop 
    } // if
  } // i loop
  
  stack_free(st);


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

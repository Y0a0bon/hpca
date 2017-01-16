#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <omp.h>
#include "../inc/utils.h"

#define TRESHOLD 1000


/**
 *
 * Function dpr()
 *
 **/
unsigned long long dpr(unsigned long **data, int n, int l, int h, int ind_start, int ind_end){
  
  int i = 0;
  
  //ycross min on the whole area, ymin min on the whole area minus the 2 ends
  int ymin = 0, ycross = 0, ind_min = 0, abs_left = 0, abs_right = 0;
  unsigned long long crosswise_area = 0, left_area = 0, right_area = 0, result_area = 0, specific_case = 0, ext_crosswise_area = 0;
  
  //Two points left : returns the rectangle defined by the height
  if ( (ind_end - ind_start) == 1 ){
    return (unsigned long long) (data[ind_end][0]-data[ind_start][0]) * h;
  }

  ymin = data[ind_start + 1][1];
  ind_min = ind_start + 1;

  
  //We don't enter the loop if ind_end - ind_start == 2
  for ( i = ind_start + 2; i < ind_end; i++ ){
    
    //Looking for the lowest ordinate
    if ( data[i][1] < ymin ){
      ymin = data[i][1];
      ind_min = i;
    }
  }
  
  ycross = ymin;

  //ycross will only be used if we work on at least one of the edges.
  //Handles the cases when the crosswise area also has to be computed before first point or after last point
  if( ind_start == 0 || ind_end == (n - 1) ){
    abs_right = data[ind_end][0];
    abs_left = data[ind_start][0];

    if ( ind_start == 0 ){
      abs_left = 0;
      
      if ( data[ind_start][1] < ycross )
	ycross = data[ind_start][1];
    }    
    if ( ind_end == n - 1 ){
      abs_right = l;
      
      if ( data[ind_end][1] < ycross )
	ycross = data[ind_end][1];
    }
    
    //extended area to at least one of the edges
    ext_crosswise_area =  ycross * (abs_right - abs_left);
  }
  
  crosswise_area = ymin * (data[ind_end][0] - data[ind_start][0]);
  if ( ext_crosswise_area > crosswise_area )
    crosswise_area = ext_crosswise_area;

  
  left_area = dpr(data, n, l, h, ind_start, ind_min);
  right_area = dpr(data, n, l, h, ind_min, ind_end);
  
  //Result is the max of these areas
  result_area = crosswise_area;
  if ( left_area > result_area )
    result_area = left_area;
  if ( right_area > result_area )
    result_area = right_area;

  
  //printf("ind_start = %d\t, ind_min = %d\t, ind_end = %d\n left = %llu\t, right = %llu\t, cross = %llu, ext_cross = %llu, result_area = %llu\n\n", ind_start, ind_min, ind_end, left_area, right_area, crosswise_area, ext_crosswise_area, result_area);
  
  //Comparison with 2 specific cases (far left and right sides) if first call of the recursive function
	/*
  if ((ind_start == 0) && (ind_end == n - 1)){

    specific_case = data[ind_start][0]*h;
    if (result_area < specific_case)
      result_area = specific_case;

    specific_case = (l - data[ind_end][0])*h;
    if (result_area < specific_case)
      result_area = specific_case;

  }
	*/
  return result_area;
}


/**
 *
 * Function dpr_parallel()
 *
 **/
unsigned long long dpr_parallel(unsigned long **data, int n, int l, int h, int ind_start, int ind_end){
  
  int i = 0;
  
  //ycross min on the whole area, ymin min on the whole area minus the 2 ends
  int ymin = 0, ycross = 0, ind_min = 0, abs_left = 0, abs_right = 0;
  unsigned long long crosswise_area = 0, left_area = 0, right_area = 0, result_area = 0, specific_case = 0, ext_crosswise_area = 0;
  
  //Two points left : returns the rectangle defined by the height
  if ( (ind_end - ind_start) == 1 ){
    return (unsigned long long) (data[ind_end][0]-data[ind_start][0]) * h;
  }

  ymin = data[ind_start + 1][1];
  ind_min = ind_start + 1;

  
  //We don't enter the loop if ind_end - ind_start == 2
  for ( i = ind_start + 2; i < ind_end; i++ ){
    //Looking for the lowest ordinate
    if ( data[i][1] < ymin ){
      ymin = data[i][1];
      ind_min = i;
    }
  } // for loop
  
  ycross = ymin;

  //ycross will only be used if we work on at least one of the edges.
  //Handles the cases when the crosswise area also has to be computed before first point or after last point
  if( ind_start == 0 || ind_end == (n - 1) ){
    abs_right = data[ind_end][0];
    abs_left = data[ind_start][0];

    if ( ind_start == 0 ){
      abs_left = 0;
      
      if ( data[ind_start][1] < ycross )
	ycross = data[ind_start][1];
    }
    
    if ( ind_end == n - 1 ){
      abs_right = l;
      
      if ( data[ind_end][1] < ycross )
	ycross = data[ind_end][1];
    }
    
    //extended area to at least one of the edges
    ext_crosswise_area =  ycross * (abs_right - abs_left);
  }
  
  crosswise_area = ymin * (data[ind_end][0] - data[ind_start][0]);
  if ( ext_crosswise_area > crosswise_area )
    crosswise_area = ext_crosswise_area;

  /* Switch to single-threading if threshold is reached */
  if(ind_min - ind_start < TRESHOLD){
    left_area = dpr(data, n, l, h, ind_start, ind_min);
    right_area = dpr(data, n, l, h, ind_min, ind_end);
  }
  else{
    {
#pragma omp task shared(left_area)
      left_area = dpr_parallel(data, n, l, h, ind_start, ind_min);
#pragma omp task shared(right_area)
      right_area = dpr_parallel(data, n, l, h, ind_min, ind_end);
#pragma omp taskwait
    }
  }
  
  //Result is the max of these areas
  result_area = crosswise_area;
  if ( left_area > result_area )
    result_area = left_area;
  if ( right_area > result_area )
    result_area = right_area;
  
  //printf("ind_start = %d\t, ind_min = %d\t, ind_end = %d\n left = %llu\t, right = %llu\t, cross = %llu, ext_cross = %llu, result_area = %llu\n\n", ind_start, ind_min, ind_end, left_area, right_area, crosswise_area, ext_crosswise_area, result_area);
  
  //Comparison with 2 specific cases (far left and right sides) if first call of the recursive function
  /*
    if ((ind_start == 0) && (ind_end == n - 1)){

    specific_case = data[ind_start][0]*h;
    if (result_area < specific_case)
    result_area = specific_case;

    specific_case = (l - data[ind_end][0])*h;
    if (result_area < specific_case)
    result_area = specific_case;

    }
  */
  return result_area;
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
  
  /* Start timing */
  debut = my_gettimeofday();

  /* Do computation:  */
#ifdef _OPENMP
  #pragma omp parallel
  #pragma omp single nowait
  S = dpr_parallel(data, n, l, h, 0, n-1);

#else
  S = dpr(data, n, l, h, 0, n-1);
#endif
  
  /* End timing */
  fin = my_gettimeofday();
  
  fprintf(stdout, "N = %d\t S = %llu\n", n, S);
  fprintf( stdout, "For n=%d: total computation time (with gettimeofday()) : %g s\n\n",
	   n, fin - debut);
  /*fprintf( stdout, "%g\n",
    fin - debut);*/
  return 0;
}

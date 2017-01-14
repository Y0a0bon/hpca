#include <stdlib.h>
#include <stdio.h>

#include <sys/time.h>
#include "../inc/utils.h"

/**
 *
 * Function dpr()
 *
 **/

unsigned long long dpr(unsigned long **data, int n, int l, int h, int ind_start, int ind_end){

  int i = 0;
  
  //ycross min on the whole area, ymin min on the whole area minus the 2 ends
  int ymin = 0, ycross = 0, ind_min = 0, abs_left = 0, abs_right = 0;
  unsigned long long crosswise_area = 0, left_area = 0, right_area = 0, result_area = 0;
  
  //Two points left : returns the rectangle defined by the height
  if ( (ind_start - ind_end) == 1 ){
    return (data[ind_end][0]-data[ind_start][0]) * h;
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
  
  if ( data[ind_start][1] < ycross ){
    ycross = data[ind_start][1];
  }
    
  if ( data[ind_end][1] < ycross ){
    ycross = data[ind_end][1];
  }

  //Handles the case when the crosswise area has to be computed before first point or after last point
  //Do sth similar for the 2 other areas ?
  if ( ind_end == n - 1 )
    abs_right = l;
  else
    abs_right = data[ind_end][0];

  if ( ind_start == 0 )
    abs_left = 0;
  else
    abs_left = data[ind_start][0];
  
  crosswise_area = ycross * (abs_right - abs_left);
  left_area = dpr(data, n, l, h, ind_start, ind_min);
  right_area = dpr(data, n, l, h, ind_min, ind_end);

  //Result is the max of these areas
  result_area = crosswise_area;
  if ( left_area > result_area )
    result_area = left_area;
  if ( right_area > result_area )
    result_area = right_area;
  
  return result_area;
  
}

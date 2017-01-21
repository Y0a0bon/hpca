/****************************
 *      Fichier stack.c     *
 ****************************/
#include <stdio.h>
#include <stdlib.h>
#include "../inc/stack.h"

/**
 * Function ()
 **/
stack_s *initialize(int size)
{
  stack_s *st = malloc(sizeof(*st));
  st->data_i = (int *)malloc(size * sizeof(int));
  st->first = NULL;
  st->size = 0;
  return st;
}


/**
 * Function stack_top()
 **/
int stack_top(stack_s *st){
  if(st->first == NULL)
    return 0;
  else
    return st->data_i[st->size-1];
}


/**
 * Function stack_push()
 **/
void stack_push (stack_s *st, coord_s n_data, int ind)
{
  element_s *new = malloc(sizeof(*new));
  if(st == NULL || new == NULL)
    exit(EXIT_FAILURE);
	
  new->data = n_data;
  new->next = st->first;
  st->first = new;
  st->data_i[st->size] = ind;
  st->size++;
  return;
}


/**
 * Function stack_pop()
 **/
coord_s stack_pop (stack_s *st)
{
  if(st == NULL)
    exit(EXIT_FAILURE);
	
  coord_s ret;
  ret.x = 0;
  ret.y = 0;
	
  element_s *stackElement = st->first;
	
  if(st != NULL && st->first != NULL){
    ret = stackElement->data;
    st->first = stackElement->next;
    st->size--;
    free(stackElement);
  }
	
  return ret;
}


/**
 * Function stack_pop_no_free()
 **/
coord_s stack_pop_no_free (stack_s *st)
{
  if(st == NULL)
    exit(EXIT_FAILURE);
	
  coord_s ret;
  ret.x = 0;
  ret.y = 0;
	
  element_s *stackElement = st->first;
	
  if(st != NULL && st->first != NULL){
    ret = stackElement->data;
    st->first = stackElement->next;
    st->size--;
  }
	
  return ret;
}


/**
 * Function stack_print()
 **/
void stack_print(stack_s *st)
{
  if(st == NULL)
    exit(EXIT_FAILURE);
  printf("{ ");
  element_s *actual = st->first;
  while(actual != NULL){
    printf("(%lu, %lu) ", actual->data.x, actual->data.y);
    actual = actual->next;
  }
  printf("}\n");
}


/**
 * Function stack_free()
 **/
void stack_free(stack_s *st){
  free(st->data_i);
  free(st);
}


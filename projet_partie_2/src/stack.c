/**
 * stack.c
 **/
#include <stdio.h>
#include <stdlib.h>
#include "../inc/stack.h"


struct coord
{
	unsigned long x;
	unsigned long y;
};

struct element
{
   //struct stack *prev;
   element_s *next;
   coord_s data;
};

struct stack
{
	element_s *first;
};


stack_s *initialize()
{
    stack_s *st = malloc(sizeof(*st));
    st->first = NULL;

    return st;
}


void stack_push (stack_s *st, coord_s data)
{
  element_s *new = malloc(sizeof(*new));
	if(st == NULL || new == NULL)
		exit(EXIT_FAILURE);
	
	new->data.x = data.x;
	new->data.y = data.y;
	new->next = st->first;
	
  return;
}


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
		free(stackElement);
	}
	
	return ret;
}


void stack_print(stack_s *st)
{
	if(st == NULL)
		exit(EXIT_FAILURE);
	printf("{");
	element_s *actual = st->first;
	while(actual != NULL){
		printf("(%lu, %lu)", actual->data.x, actual->data.y);
		actual = actual->next;
	}
	printf(" }");
}

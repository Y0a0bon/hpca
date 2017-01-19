#include <stdio.h>
#include <stdlib.h>
#include "../inc/stack.h"

struct stack
{
   //struct stack *prev;
   struct stack *next;
   unsigned long data[2];
};

stack_s *stack_new (void)
{
   return (NULL);
}

void stack_push (stack_s ** pp_stack, unsigned long data[2])
{
   if (pp_stack != NULL)
   {
      stack_s *p_p = *pp_stack;
      stack_s *p_l = NULL;

      p_l = malloc (sizeof (*p_l));
      if (p_l != NULL)
      {
         p_l->data = data;
         p_l->next = NULL;
         p_l->prev = p_p;
         if (p_p != NULL)
            p_p->next = p_l;
         *pp_stack = p_l;
      }
      else
      {
         fprintf (stderr, "Memoire insuffisante\n");
         exit (EXIT_FAILURE);
      }
   }
   return;
}

void *stack_pop (stack_s ** pp_stack)
{
   void *ret = NULL;

   if (pp_stack != NULL && *pp_stack != NULL)
   {
      stack_s *p_l = *pp_stack;
      stack_s *p_p = p_l->prev;

      if (p_p != NULL)
         p_p->next = NULL;
      ret = p_l->data;
      free (p_l);
      p_l = NULL;
      *pp_stack = p_p;
   }
   return (ret);
}

void stack_delete (stack_s ** pp_stack)
{
   if (pp_stack != NULL && *pp_stack != NULL)
   {
      while (*pp_stack != NULL)
         stack_pop (pp_stack);
   }
   return;
}

void stack_print(stack_s **pp_stack)
{
		printf("{");
		if(pp_stack == NULL)
			printf(" }");
		else{
			stack_s *stack_tmp = *pp_stack;
			printf("%lu %lu", stack_tmp->data[0], stack_tmp->data[1]);
			while(stack_tmp->prev != NULL){
				*stack_tmp = stack_tmp->prev;
				printf("%lu %lu", stack_tmp->data[0], stack_tmp->data[1]);
			}
			printf(" }");
		}
	
}

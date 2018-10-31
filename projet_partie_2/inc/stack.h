#ifndef H_STACK
#define H_STACK

typedef struct
{
  unsigned long x;
  unsigned long y;
}coord_s;

typedef struct
{
  //struct stack *prev;
  struct element_s *next;
  coord_s data;
}element_s;

typedef struct
{
  element_s *first;
  int *data_i;
  int size;
}stack_s;

stack_s *initialize(int);
int stack_top(stack_s *);
void stack_push (stack_s *, coord_s, int);
coord_s stack_pop (stack_s *);
coord_s stack_pop_no_free (stack_s *st);
void stack_print(stack_s *);
void stack_free(stack_s *);

#endif /* not H_STACK */

#include <stdlib.h>
#include <stdio.h>

#define MY_RAND(I) (long)((I)*drand48())

typedef struct {
  long x;
  long y;
} point;

// Comparaison suivant x:
int compare_point(const void * a, const void * b){
  point* pa = (point *)a;
  point* pb = (point *)b;

  if( pa->x  < pb->x)
    return -1;
  if( pa->x  > pb->x)
    return 1;
  /* pa->x  == pb->x */
  return 0;
}



int main(int argc, char **argv){
  long l,h,n;
  point *tab;
  char nom[100];
  FILE *f;

  if (argc < 4){
    fprintf(stderr, "Usage: a.out l h n");
  }

  l = atol(argv[1]);
  h = atol(argv[2]);
  n = atol(argv[3]);

  printf("Generation aleatoire pour : l=%ld, h=%ld, n=%ld\n",
         l, h, n);

  tab = (point *) malloc(n * sizeof(point));

  ///// Generation aleatoire :
  srand48(42);

  // premier et dernier points :
  tab[0].x = 0;
  tab[0].y = MY_RAND(h);
  tab[n-1].x = l;
  tab[n-1].y = MY_RAND(h);

// autres points :
  for (long i=1; i<n-1; i++){
    tab[i].x = MY_RAND(l);
    tab[i].y = MY_RAND(h);
  }

  ///// Tri suivant x :
  qsort(tab, n, sizeof(point),compare_point);


  ///// Sauvegarde :
  sprintf(nom, "ex_N%ld_alea", n);
  f=fopen(nom, "w+");
  fprintf(f, "%ld %ld\n", l, h);
  fprintf(f, "%ld\n", n);
  for (long i=0; i<n; i++){
    fprintf(f, "%ld %ld\n", tab[i].x, tab[i].y);
  }
  fclose(f);
  printf("Resultat ecrit dans : %s\n", nom);

  free(tab);
  return EXIT_SUCCESS;
}


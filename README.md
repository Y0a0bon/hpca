###################
## HPCA project  ##
###################

by :
M. Benoît
S. Coll


To compile :
************
  $ cd ./projet_partie_1
  $ make cible
  
cible:  naive_algo, enhanced_algo (on CPU),
        naive_algo_cuda, naive_algo_opt_cuda (on GPU)
        enhanced_algo_cuda, enhanced_algo_opt_cuda (on GPU)
        
By default, naive_algo and enhanced_algo are executed sequentially.
To use OpenMP, edit "Makefile" line 5, from :
    OMP = #-fopenmp
to :
    OMP = fopenmp

(will be changed in the future for easier use)

To execute :
************
  $ bin/cible data/ex_NXXX_alea
  
To clean :
**********
  $ make clean





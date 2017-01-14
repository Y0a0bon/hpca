#!/bin/bash
#

l=1000000000
h=1000000000

for i in $(seq 15 20) 25 26 ; do
    #    echo $i 
    n=$((2 ** ${i}))
    #    echo $n
    ./alea $l $h $n 

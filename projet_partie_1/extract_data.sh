#! /bin/bash  

let "i = 1"

while read line  
do   
    if [ $(( $i % 7 )) -eq 0 ] ;
    then
	echo -e "$line" 
    fi
    let "i = i + 1"
done < saved/stats_500_enhanced_cuda.txt

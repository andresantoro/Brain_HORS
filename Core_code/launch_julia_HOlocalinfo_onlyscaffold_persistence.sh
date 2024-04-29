#!/bin/sh



juliapath=`which julia`
compiled_code="./SimplicialTS/Simplicial.so"
nthreads=$1
input_file=$2
output_file_scaffold=$3
scaffold_flag=2 # Flag 1-> frequency, Flag 2 -> persistence


##Launching the code
time julia -J ${compiled_code} --threads ${nthreads} simplicial_multivariate.jl ${input_file} -s ${output_file_scaffold} -f ${scaffold_flag} > /dev/null

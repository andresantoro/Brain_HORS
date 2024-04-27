#!/bin/sh



juliapath=`which julia`
compiled_code="./SimplicialTS/Simplicial.so"
nthreads=$1
input_file=$2
output_file_scaffold=$3
output_file_triangles=$4
scaffold_flag=1 # Flag 1-> frequency, Flag 2 -> persistence


##Launching the code
time julia -J ${compiled_code} --threads ${nthreads} simplicial_multivariate.jl ${input_file} -s ${output_file_scaffold} -f ${scaffold_flag} -o ${output_file_triangles} > /dev/null

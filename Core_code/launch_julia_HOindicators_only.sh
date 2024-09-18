#!/bin/sh



juliapath=`which julia`
compiled_code="./SimplicialTS/Simplicial.so"
nthreads=$1
input_file=$2
output_file=$3

##Launching the code
time julia -J ${compiled_code} --threads ${nthreads} simplicial_multivariate.jl ${input_file} > ${output_file}


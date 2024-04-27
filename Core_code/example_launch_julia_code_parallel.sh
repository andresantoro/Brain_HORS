#!/bin/sh

juliapath=`which julia`
compiled_code="./SimplicialTS/Simplicial.so"
nthreads=6
input_file="./SimplicialTS/data/TS_Schaefer60S_gsr_bp_z.mat"
maxT=10
output_file_scaffold="HO_scaffold_frequency"
scaffold_flag=2 # Flag 1-> frequency, Flag 2 -> persistence
output_file_triangles="HO_triangles"


###Launching the code  using 6 threads, saving scaffold and violating triangles list
# time julia -J ${compiled_code} --threads ${nthreads} simplicial_multivariate.jl ${input_file} -s ${output_file_scaffold} -f ${scaffold_flag} -o ${output_file_triangles}

##Launching the code only for time 1-20 using 6 threads, saving scaffold and violating triangles list
time julia -J ${compiled_code} --threads ${nthreads} simplicial_multivariate.jl ${input_file} -t 1 20 -s ${output_file_scaffold} -f ${scaffold_flag} -o ${output_file_triangles}


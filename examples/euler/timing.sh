#!/bin/zsh

export PYOPENCL_CTX=0:1
export LOOPY_NO_CACHE=1
export PYOPENCL_NO_CACHE=1
export POCL_KERNEL_CACHE=0
export CUDA_CACHE_DISABLE=1

export ARGS="--lazy --resolution 128"

orders=(1 2 4 5)  # skip order = 3 to avoid bad reshape logic

echo "testing TP cases"
for i in "${orders[@]}"
do
    echo "testing order $i"
    eval "python vortex.py --tpe --order $i $ARGS" 2>&1 | \
        grep -a "walltime" > timing/timing-$i-tp.txt
    cat timing/timing-$i-tp.txt
    echo "done testing order $i"
done
echo "done testing TP cases"

echo "testing simplicial cases"
for i in "${orders[@]}"
do
    echo "testing order $i"
    eval "python vortex.py --order $i $ARGS" 2>&1 | \
        grep -a "walltime" > timing/timing-$i-smp.txt
    cat timing/timing-$i-smp.txt
    echo "done testing order $i"
done
echo "done testing simplicial cases"


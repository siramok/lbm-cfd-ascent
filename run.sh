module reset
module use /soft/modulefiles
module load spack-pe-gnu
module load cudatoolkit-standalone/12.2.2
module load visualization/ascent
module load cmake

NODES=`wc -l < $PBS_NODEFILE`
RANKS_PER_NODE=4
TOTAL_RANKS=$(( $NODES * $RANKS_PER_NODE ))

cp ascent_files/* build
cd build
rm *.png
mpiexec -n $TOTAL_RANKS -ppn $RANKS_PER_NODE ./lbmcfd

source ~/ascent-bidirectional/sourceme

export CC=$(which cc)
export CXX=$(which CC)

# Reconfigure
rm -rf build
cmake -DAscent_DIR=~/ascent-bidirectional/ascent/scripts/build_ascent/install/ascent-develop/ -B build -S .

# Copy our working files into the build directory
cp ascent_files/* build/

# Rebuild
cd build
make
#reset

# Run
./start_jupyter.sh
NODES=`wc -l < $PBS_NODEFILE`
RANKS_PER_NODE=5
TOTAL_RANKS=$(( $NODES * $RANKS_PER_NODE ))
mpiexec -n $TOTAL_RANKS -ppn $RANKS_PER_NODE  ./lbmcfd


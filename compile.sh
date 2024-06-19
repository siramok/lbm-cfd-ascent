module reset
module use /soft/modulefiles
module load spack-pe-gnu
module load cudatoolkit-standalone/12.2.2
module load visualization/ascent
module load cmake

export CC=$(which cc)
export CXX=$(which CC)

# Reconfigure
rm -rf build
mkdir build
cp ascent_files/* build
cd build
cmake -DAscent_DIR=/soft/visualization/ascent/develop/2024-05-03-8baa78c/ascent-develop/ -S ..
make

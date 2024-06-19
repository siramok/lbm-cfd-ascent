# Python venv containing jupyter and the ascent-jupyter-bridge
source env/bin/activate

# Reconfigure
rm -rf build
cmake -DAscent_DIR=/home/siramok/code/ascent/scripts/build_ascent/install/ascent-develop -B build -S .

# Copy our working files into the build directory
cp ascent_files/* build/

# Rebuild
cd build
make
# reset

# Run
mpiexec -n 2 ./lbmcfd

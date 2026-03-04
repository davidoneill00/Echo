
// 1: Requirement HDF5. Installation
conda install -c conda-forge "hdf5>=1.14" h5py numpy

// 2: Setting up a run. 
Write this later

// 3: Running 
rm -rf build
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
./echo

// 4. Analysing Data
source .echo311/bin/activate
python plotting/plot.py
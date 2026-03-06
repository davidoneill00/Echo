
// 1: Environment with packages
source .echo311/bin/activate

// 2: Building
rm -rf build
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j

// 3: Running 
./echo ../inputs.txt

// 4. Analysing Data
python plotting/plot.py
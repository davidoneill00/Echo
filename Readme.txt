 arch -arm64 c++ -O3 -Wall -shared -std=c++17 -fPIC \
    $(/opt/homebrew/Caskroom/miniconda/base/bin/python -m pybind11 --includes) \
    bfs_solver.cpp \
    -o bfs_solver$(/opt/homebrew/Caskroom/miniconda/base/bin/python3-config --extension-suffix) \
    -undefined dynamic_lookup

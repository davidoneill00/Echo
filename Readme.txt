arch -arm64 c++ -O3 -Wall -shared -std=c++17 -fPIC \
  $(python -m pybind11 --includes) \
  Method.cpp \
  -o "Method_cpp$(python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")" \
  -undefined dynamic_lookup

python run.py

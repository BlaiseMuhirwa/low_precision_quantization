
# create the build directory if it does not 
# exist. 
mkdir -p build
cd build 
# make clean 
cmake ../CMakeLists.txt -DPYTHON_EXECUTABLE=$(which python3)

# Make on 6 cores
# make lpq -j 6 

# Make the Python library 
make low_precision_quantizer -j 6


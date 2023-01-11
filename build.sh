
# create the build directory if it does not 
# exist. 
mkdir -p build
cd build 
# make clean 
cmake ../CMakeLists.txt 
make -j 



# create the build directory if it does not 
# exist. 
mkdir -p build
cd build 
cmake ../src 

# compile and link the project 
cmake --build . 

# Run the executable 
./Main 
clang-format -i src/*.h 
clang-format -i src/*.cc
clang-format -i src/tests/*.cc

clang-format -i bindings/*.cc 

black python_scripts

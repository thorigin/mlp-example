
HEADERS := $(shell find ./mlp -iname "*.hpp")
CFLAGS=-Wall -Wpedantic -pedantic-errors -std=c++11
INCLUDE_DIRS := -I.
example/iris.out: example/iris.cpp $(HEADERS)
	$(CXX) $(CFLAGS) $(INCLUDE_DIRS) -g -O3 -mavx -o example/iris.out example/iris.cpp

test:
	./example/iris.out ./example/iris.csv
clean: 
	-rm ./example/iris.out

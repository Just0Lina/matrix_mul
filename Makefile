all:
	g++ src/multiply.cpp apps/matMulTest.cpp -O3 -march=native -lgtest -lpthread && ./a.out

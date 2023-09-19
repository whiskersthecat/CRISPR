testNNW: neuralnetworkstatscommandline_TEST.cc neuralnetwork2.h
	g++ neuralnetworkstatscommandline_TEST.cc -o testNNW -Wall -Wextra -Werror -O3 -std=c++11
testdebug: neuralnetworkstatscommandline_TEST.cc neuralnetwork2.h
	g++ neuralnetworkstatscommandline_TEST.cc -o testNNW -g -Wall -Wextra -Werror -std=c++11
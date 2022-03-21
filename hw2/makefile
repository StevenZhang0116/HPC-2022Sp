CXX = g++
CXXFLAGS = -std=c++11 -Wall -O3 -march=native

ifeq "$(CXX)" "icpc"
CXXFLAGS += -qopenmp
else
CXXFLAGS += -fopenmp
endif

TARGETS = $(basename $(wildcard *.c)) $(basename $(wildcard *.cpp)) 

all : $(TARGETS)

%:%.c *.h
	$(CXX) $(CXXFLAGS) $< -o $@
%:%.cpp *.h
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	-$(RM) $(TARGETS) *~

.PHONY: all, clean
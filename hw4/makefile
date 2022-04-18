# Reference: https://stackoverflow.com/questions/34108372/creating-a-makefile-for-cuda-programs
# 			 https://github.com/TravisWThompson1/Makefile_Example_CUDA_CPP_To_Executable

CXX = g++
CXXFLAGS = -std=c++11 -O3 -march=native -fopenmp

CUDA_INCDIR = -I $(CUDA_HOME)/include -I $(CUDA_HOME)/samples/common/inc
CUDA_LIBS = -lblas -L${CUDA_HOME}/lib64 -lcudart -lcublas

NVCC = nvcc
NVCCFLAGS = -std=c++11
NVCCFLAGS += -Xcompiler "-fopenmp" 

TARGETS = $(basename $(wildcard *.cpp)) $(basename $(wildcard *.c)) $(basename $(wildcard *.cu))

all : $(TARGETS)

%:%.cpp
	$(CXX) $(CXXFLAGS) $(CUDA_INCDIR) $< $(CUDA_LIBS) -o $@

%:%.c
	$(CXX) $(CXXFLAGS) $(CUDA_INCDIR) $< $(CUDA_LIBS) -o $@

%:%.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

clean:
	-$(RM) $(TARGETS) *~

.PHONY: all, clean
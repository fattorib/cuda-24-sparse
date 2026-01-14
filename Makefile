STD		          =-std=c++20
GPU_FLAGS         =-arch=sm_89 
NVCCFLAGS         =-O3 -Xptxas=--verbose -Xptxas=--warn-on-spills -lineinfo -lcuda --use_fast_math --expt-relaxed-constexpr -Xcompiler -fopenmp
FORMAT            =-style="{BasedOnStyle: Google, IndentWidth: 4, TabWidth: 4, UseTab: ForIndentation, ColumnLimit: 0}"
TARGET	  		  =gemm

build-dir:
	if [ ! -d build ]; then mkdir build; fi

all: $(TARGET)

$(TARGET):  build-dir
	nvcc $(STD) -I csrc/include -I /usr/local/cusparselt/include $(GPU_FLAGS) $(NVCCFLAGS) $(TARGET).cu -L/usr/local/cusparselt/lib -lcusparseLt -o build/$(TARGET)

format: 
	clang-format $(FORMAT) -i csrc/include/*.hpp
	clang-format $(FORMAT) -i csrc/include/*.cuh
	clang-format $(FORMAT) -i *.cu

clean:
	rm -f build/$(TARGET)
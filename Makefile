NVCC=/opt/cuda/bin/nvcc

all:
	$(NVCC) -c src/Mandelbrot.cu -o bin/mandelbrot.o -I"src"
	g++ -c src/main.cpp -o bin/main.o
	$(NVCC) bin/main.o bin/mandelbrot.o -o main

run:
	./main

edit_cvt:
	nvim java/src/ca/Dyamen/Main.java

compile_cvt:
	javac java/src/ca/Dyamen/Main.java

cvt:
	java -cp java/src ca.Dyamen.Main out/out.dat

clean:
	rm -r -f bin/*
	rm -r -f java/src/ca/Dyamen/*.class

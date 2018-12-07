N := 1000
mu := 0
sigma := 1

histograma.png: sample_0.dat
	python graficar.py

sample_0.dat: puntos.x
	./puntos.x $(mu) $(sigma) $(N)

puntos.x: puntos.c
	gcc -fopenmp -o puntos.x puntos.c -lm

clean: 
	rm -f *.dat *.pdf *.x 
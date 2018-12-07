#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <time.h> 
#define PI 3.14159265359

/*Modelo para trabajar*/
float modelo(float x,float mean, float std)
{
    float y;
    y=exp(-1*pow((x-mean),2)/(2*pow(std,2)))/sqrt(2*PI*pow(std,2));
    return y;
}

/* Crear una distribucion gausiana con cadenas de Markov*/ 
float* generarCadena(float mean, float std,int size)
{
    float* lista;
    int N = size;
    lista = malloc(N*sizeof(float));
    lista[0]=mean;
    float listaPropuesta;
    float F;
    
    for (int i=1; i<N;i++)
    {
        srand(time(0));
        listaPropuesta = lista[i-1] + ((double)rand()/(double)RAND_MAX*2-1)*std;
        F = modelo(listaPropuesta,mean,std)/modelo(lista[i-1],mean,std);
        if(F>=1)
        {
        lista[i]=listaPropuesta;
        }
        else
        {
            if ((double)rand()/(double)RAND_MAX<F)
            {
            lista[i]=listaPropuesta;    
            }
            else{
            lista[i]=lista[i-1];
            }
        }
        
    }
      
return lista;
}

/*Metodo para crear el archivo de los datos*/ 
void crearEldat(float* vector , int N,int n)
{
FILE *in;
char filename[100];
int i;

sprintf(filename, "sample_%d.dat",n);
in = fopen(filename,"w");
for(i=0; i < N; i++)
{
  fprintf(in, "%f\n", vector[i]);
}
    
fclose(in);
}


int main(int argc, char ** argv)
{
float mu = atof(argv[1]);
float std = atof(argv[2]);
int N = atof(argv[3]);
    
#pragma omp parallel
{
int thread_id = omp_get_thread_num();
float* array;
array = generarCadena(mu,std,N);
crearEldat(array, N,thread_id);
}

return 0;
}
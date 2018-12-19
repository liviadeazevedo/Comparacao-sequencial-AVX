#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <immintrin.h>

#define ALIGNED_AVX 32 //Alinhamento dos dados para serem utilizados nos registradores __m256.

//Realizará testes apenas com matrizes quadradas mas a ideia proposta serve para todos os casos de multiplicação de matrizes.

//run == 1: Vetorializado ; run != 1: Sequencial

void printMatrix(float *m, float w, float h){
   int i, j;

   printf("\n");

   for (j = 0; j < h; j++){
      for (i = 0; i < w; i++){
         int k = j * w + i;
         printf("%.2f ", m[k]);
      }
      printf("\n");
   }

}

int main (int argc, char **argv){
   int iA, jA, iB, jB, iC, jC;
   int width = atoi(argv[1]), height = atoi(argv[2]),run = atoi(argv[3]);//run: vetorializado == 1 e sequencial == qualquer numero.

    int linha1 = width;
    int linha2 = width;
    int coluna1 = height;
    int coluna2 = height;

    float* A = (float*) _mm_malloc(width * height * sizeof(float),ALIGNED_AVX);
    float* B = (float*) _mm_malloc(width * height * sizeof(float),ALIGNED_AVX);
    float* C = (float*) _mm_malloc(width * height * sizeof(float),ALIGNED_AVX);

   //printf("\nMultiplicando matriz\n");
   for (jC = 0; jC < width; jC++){
      for (iC = 0; iC < height; iC++){
         int kC = jC * width + iC;
         A[kC] = (float) kC + 1;
	 	 C[kC] = 0; //inicializando a matriz C.
         if (jC == iC)
            B[kC] = 1.0f;
         else
            B[kC] = 0.0f;

      }
   }

   if(run == 1)
   {
      printf("Entrei no vetorializado\n");

      int i,j,k,p,l,a;
    __m256 ymm0,ymm1,ymm2,ymm3,ymm4,ymm5,ymm6,ymm7;
    int caso, quantidadeElementosVetorializar;

    for(i = 0;i < linha1;i++)
    {
	    for(j = 0;j < coluna1;j++)
	    {
	      ymm0 = _mm256_set1_ps(A[i * linha1 + j]); //A[i][j]

	      for(k = 0;k < coluna2;k += 56)
	      {

	    quantidadeElementosVetorializar = (coluna2 - k);

		if(quantidadeElementosVetorializar >= 56)
		  caso = 0;

		if(quantidadeElementosVetorializar < 56 && quantidadeElementosVetorializar > 48)
		  caso = 1;

		if(quantidadeElementosVetorializar <= 48 && quantidadeElementosVetorializar > 40)
		  caso = 2;

		if(quantidadeElementosVetorializar <= 40 && quantidadeElementosVetorializar > 32)
		  caso = 3;

		if(quantidadeElementosVetorializar <= 32 && quantidadeElementosVetorializar > 24)
		  caso = 4;

		if(quantidadeElementosVetorializar <= 24 && quantidadeElementosVetorializar > 16)
		  caso = 5;

		if(quantidadeElementosVetorializar <= 16 && quantidadeElementosVetorializar > 8)
		  caso = 6;

		if(quantidadeElementosVetorializar <= 8)
		  caso = 7;

		switch(caso)
		{
		  case 0:
		  {
		        ymm1 = _mm256_add_ps(_mm256_loadu_ps(&C[i * linha1 + k]),_mm256_mul_ps(ymm0,_mm256_loadu_ps(&B[j * linha2 + k])));
		        ymm2 = _mm256_add_ps(_mm256_loadu_ps(&C[i * linha1 + k + 8]),_mm256_mul_ps(ymm0,_mm256_loadu_ps(&B[j * linha2 + (k + 8)])));
		        ymm3 = _mm256_add_ps(_mm256_loadu_ps(&C[i * linha1 + k + 16]),_mm256_mul_ps(ymm0,_mm256_loadu_ps(&B[j * linha2 + k + 16])));
		        ymm4 = _mm256_add_ps(_mm256_loadu_ps(&C[i * linha1 + k + 24]),_mm256_mul_ps(ymm0,_mm256_loadu_ps(&B[j * linha2 + k + 24])));
		        ymm5 = _mm256_add_ps(_mm256_loadu_ps(&C[i * linha1 + k + 32]),_mm256_mul_ps(ymm0,_mm256_loadu_ps(&B[j * linha2 + k + 32])));
		        ymm6 = _mm256_add_ps(_mm256_loadu_ps(&C[i * linha1 + k + 40]),_mm256_mul_ps(ymm0,_mm256_loadu_ps(&B[j * linha2 + k + 40])));
		        ymm7 = _mm256_add_ps(_mm256_loadu_ps(&C[i * linha1 + k + 48]),_mm256_mul_ps(ymm0,_mm256_loadu_ps(&B[j * linha2 + k + 48])));

		        _mm256_storeu_ps(&C[i * linha1 + k],ymm1);
		        _mm256_storeu_ps(&C[i * linha1 + k + 8],ymm2);
		        _mm256_storeu_ps(&C[i * linha1 + k + 16],ymm3);
		        _mm256_storeu_ps(&C[i * linha1 + k + 24],ymm4);
		        _mm256_storeu_ps(&C[i * linha1 + k + 32],ymm5);
		        _mm256_storeu_ps(&C[i * linha1 + k + 40],ymm6);
		        _mm256_storeu_ps(&C[i * linha1 + k + 48],ymm7);

		        break;
		  }

		  case 1:
		  {
		        ymm1 = _mm256_add_ps(_mm256_loadu_ps(&C[i * linha1 + k]),_mm256_mul_ps(ymm0,_mm256_loadu_ps(&B[j * linha2 + k])));
		        ymm2 = _mm256_add_ps(_mm256_loadu_ps(&C[i * linha1 + k + 8]),_mm256_mul_ps(ymm0,_mm256_loadu_ps(&B[j * linha2 + (k + 8)])));
		        ymm3 = _mm256_add_ps(_mm256_loadu_ps(&C[i * linha1 + k + 16]),_mm256_mul_ps(ymm0,_mm256_loadu_ps(&B[j * linha2 + k + 16])));
		        ymm4 = _mm256_add_ps(_mm256_loadu_ps(&C[i * linha1 + k + 24]),_mm256_mul_ps(ymm0,_mm256_loadu_ps(&B[j * linha2 + k + 24])));
		        ymm5 = _mm256_add_ps(_mm256_loadu_ps(&C[i * linha1 + k + 32]),_mm256_mul_ps(ymm0,_mm256_loadu_ps(&B[j * linha2 + k + 32])));
		        ymm6 = _mm256_add_ps(_mm256_loadu_ps(&C[i * linha1 + k + 40]),_mm256_mul_ps(ymm0,_mm256_loadu_ps(&B[j * linha2 + k + 40])));
		        ymm7 = _mm256_add_ps(_mm256_loadu_ps(&C[i * linha1 + k + 48]),_mm256_mul_ps(ymm0,_mm256_loadu_ps(&B[j * linha2 + k + 48])));

		        _mm256_storeu_ps(&C[i * linha1 + k],ymm1);
		        _mm256_storeu_ps(&C[i * linha1 + k + 8],ymm2);
		        _mm256_storeu_ps(&C[i * linha1 + k + 16],ymm3);
		        _mm256_storeu_ps(&C[i * linha1 + k + 24],ymm4);
		        _mm256_storeu_ps(&C[i * linha1 + k + 32],ymm5);
		        _mm256_storeu_ps(&C[i * linha1 + k + 40],ymm6);

		        for(l = 0,p = k;(p + 48) < coluna2;l++,p++)
		        {
		                C[i * linha1 + (p + 48)] = ymm7[l];
		        }

		        break;
		  }

		  case 2:
		  {
		        ymm1 = _mm256_add_ps(_mm256_loadu_ps(&C[i * linha1 + k]),_mm256_mul_ps(ymm0,_mm256_loadu_ps(&B[j * linha2 + k])));
		        ymm2 = _mm256_add_ps(_mm256_loadu_ps(&C[i * linha1 + k + 8]),_mm256_mul_ps(ymm0,_mm256_loadu_ps(&B[j * linha2 + (k + 8)])));
		        ymm3 = _mm256_add_ps(_mm256_loadu_ps(&C[i * linha1 + k + 16]),_mm256_mul_ps(ymm0,_mm256_loadu_ps(&B[j * linha2 + k + 16])));
		        ymm4 = _mm256_add_ps(_mm256_loadu_ps(&C[i * linha1 + k + 24]),_mm256_mul_ps(ymm0,_mm256_loadu_ps(&B[j * linha2 + k + 24])));
		        ymm5 = _mm256_add_ps(_mm256_loadu_ps(&C[i * linha1 + k + 32]),_mm256_mul_ps(ymm0,_mm256_loadu_ps(&B[j * linha2 + k + 32])));
		        ymm6 = _mm256_add_ps(_mm256_loadu_ps(&C[i * linha1 + k + 40]),_mm256_mul_ps(ymm0,_mm256_loadu_ps(&B[j * linha2 + k + 40])));

		        _mm256_storeu_ps(&C[i * linha1 + k],ymm1);
		        _mm256_storeu_ps(&C[i * linha1 + k + 8],ymm2);
		        _mm256_storeu_ps(&C[i * linha1 + k + 16],ymm3);
		        _mm256_storeu_ps(&C[i * linha1 + k + 24],ymm4);
		        _mm256_storeu_ps(&C[i * linha1 + k + 32],ymm5);

		        for(l = 0,p = k;(p + 40) < coluna2;l++,p++)
		        {
		              C[i * linha1 + (p + 40)]  = ymm6[l];
		        }

		        break;
		  }

		  case 3:
		  {
		        ymm1 = _mm256_add_ps(_mm256_loadu_ps(&C[i * linha1 + k]),_mm256_mul_ps(ymm0,_mm256_loadu_ps(&B[j * linha2 + k])));
		        ymm2 = _mm256_add_ps(_mm256_loadu_ps(&C[i * linha1 + k + 8]),_mm256_mul_ps(ymm0,_mm256_loadu_ps(&B[j * linha2 + (k + 8)])));
		        ymm3 = _mm256_add_ps(_mm256_loadu_ps(&C[i * linha1 + k + 16]),_mm256_mul_ps(ymm0,_mm256_loadu_ps(&B[j * linha2 + k + 16])));
		        ymm4 = _mm256_add_ps(_mm256_loadu_ps(&C[i * linha1 + k + 24]),_mm256_mul_ps(ymm0,_mm256_loadu_ps(&B[j * linha2 + k + 24])));
		        ymm5 = _mm256_add_ps(_mm256_loadu_ps(&C[i * linha1 + k + 32]),_mm256_mul_ps(ymm0,_mm256_loadu_ps(&B[j * linha2 + k + 32])));

		        _mm256_storeu_ps(&C[i * linha1 + k],ymm1);
		        _mm256_storeu_ps(&C[i * linha1 + k + 8],ymm2);
		        _mm256_storeu_ps(&C[i * linha1 + k + 16],ymm3);
		        _mm256_storeu_ps(&C[i * linha1 + k + 24],ymm4);

		          for(l = 0,p = k;(p + 32) < coluna2;l++,p++)
		          {
		                C[i * linha1 + (p + 32)] = ymm5[l];
		          }

		          break;
		}

		  case 4:
		  {
		        ymm1 = _mm256_add_ps(_mm256_loadu_ps(&C[i * linha1 + k]),_mm256_mul_ps(ymm0,_mm256_loadu_ps(&B[j * linha2 + k])));
		        ymm2 = _mm256_add_ps(_mm256_loadu_ps(&C[i * linha1 + k + 8]),_mm256_mul_ps(ymm0,_mm256_loadu_ps(&B[j * linha2 + (k + 8)])));
		        ymm3 = _mm256_add_ps(_mm256_loadu_ps(&C[i * linha1 + k + 16]),_mm256_mul_ps(ymm0,_mm256_loadu_ps(&B[j * linha2 + k + 16])));
		        ymm4 = _mm256_add_ps(_mm256_loadu_ps(&C[i * linha1 + k + 24]),_mm256_mul_ps(ymm0,_mm256_loadu_ps(&B[j * linha2 + k + 24])));

		        _mm256_storeu_ps(&C[i * linha1 + k],ymm1);
		        _mm256_storeu_ps(&C[i * linha1 + k + 8],ymm2);
		        _mm256_storeu_ps(&C[i * linha1 + k + 16],ymm3);

		        for(l = 0,p = k;(p + 24) < coluna2;l++,p++)
		        {
		              C[i * linha1 + (p + 24)] = ymm4[l];
		        }

		          break;

		  }

		  case 5:
		  {
		        ymm1 = _mm256_add_ps(_mm256_loadu_ps(&C[i * linha1 + k]),_mm256_mul_ps(ymm0,_mm256_loadu_ps(&B[j * linha2 + k])));
		        ymm2 = _mm256_add_ps(_mm256_loadu_ps(&C[i * linha1 + k + 8]),_mm256_mul_ps(ymm0,_mm256_loadu_ps(&B[j * linha2 + (k + 8)])));
		        ymm3 = _mm256_add_ps(_mm256_loadu_ps(&C[i * linha1 + k + 16]),_mm256_mul_ps(ymm0,_mm256_loadu_ps(&B[j * linha2 + k + 16])));

		        _mm256_storeu_ps(&C[i * linha1 + k],ymm1);
		        _mm256_storeu_ps(&C[i * linha1 + k + 8],ymm2);

		      for(l = 0,p = k;(p + 16) < coluna2;l++,p++)
		      {
		            C[i * linha1 + (p + 16)] = ymm3[l];
		      }

		        break;
		  }

		  case 6:
		  {
		        ymm1 = _mm256_add_ps(_mm256_loadu_ps(&C[i * linha1 + k]),_mm256_mul_ps(ymm0,_mm256_loadu_ps(&B[j * linha2 + k])));
		        ymm2 = _mm256_add_ps(_mm256_loadu_ps(&C[i * linha1 + k + 8]),_mm256_mul_ps(ymm0,_mm256_loadu_ps(&B[j * linha2 + (k + 8)])));

		        _mm256_storeu_ps(&C[i * linha1 + k],ymm1);

		      for(l = 0,p = k;(p + 8) < coluna2;l++,p++)
		      {
		            C[i * linha1 + (p + 8)] = ymm2[l];
		      }

		      break;
		  }

		  case 7:
		  {
		        ymm1 = _mm256_add_ps(_mm256_loadu_ps(&C[i * linha1 + k]),_mm256_mul_ps(ymm0,_mm256_loadu_ps(&B[j * linha2 + k])));

		      for(l = 0,p = k;l < quantidadeElementosVetorializar;l++,p++)
		      {
		          C[i * linha1 + p] = ymm1[l];
		      }

		      break;
		   }
		}
	    //------------------------
	      }
	   }
	}
   }
   else
   {
      printf("Entrei no sequêncial\n");

         for (jC = 0; jC < height; jC++){
            for (iC = 0; iC < width; iC++){
               int kC = jC * width + iC;
               float aux = 0.0f;
               jA = jC;
               for (iA = 0; iA < width; iA++){
                  jB = iA;
                  iB = iC;
                  int kA = jA * width + iA;
                  int kB = jB * width + iB;
                  aux += A[kA] * B[kB];

               }
               C[kC] = aux;
            }
         }
    }

   //printMatrix(C, width, height);
   _mm_free (A);
   _mm_free (B);
   _mm_free (C);
   return EXIT_SUCCESS;
}

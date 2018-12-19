#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <immintrin.h>

#define ALIGNED_AVX 32 //Alinhamento dos dados para serem utilizados nos registradores __m256.

int main(int argc, char **argv)  //parâmetros: a opção(tipo de matriz), a ordem da matriz a ser gerada e o tipo de execução.
{
	int opcao;
	int n, i, j,run;
	int EhIdentidade = 1;

	opcao = atoi(argv[1]); //matriz a ser gerada.
	n = atoi(argv[2]); //tamanho da matriz
	run = atoi(argv[3]); //run == 1: Vetorializado; run != 1: sequencial

	int TamVetor = (n*(n+1))/2;

	float* bandaSup = (float*) _mm_malloc(TamVetor * sizeof(float),ALIGNED_AVX);
	float* bandaInf = (float*) _mm_malloc(TamVetor * sizeof(float),ALIGNED_AVX);
	float* diagonal = (float*) _mm_malloc(n * sizeof(float),ALIGNED_AVX);

	switch(opcao)
	{
		case 1://matriz qualquer...
		{
			for(i=0;i<n;i++)
				diagonal[i] = (float) i;

			for(i=0;i<TamVetor;i++)
			{
				bandaInf[i] = (float) i+3;
				bandaSup[i] = (float) i+2;
			}
			break;
		}
		case 2://matriz bandas 0 e diagonal diferente de 1
		{
			for(i=0;i<n;i++)
				diagonal[i] = (float) i;

			for(i=0;i<TamVetor;i++)
			{
				bandaInf[i] = 0.0f;
				bandaSup[i] = 0.0f;
			}
			break;
		}
		case 3://matriz identidade
		{
			for(i=0;i<n;i++)
			{
			  diagonal[i] = 1.0f;
			}

			for(i=0;i<TamVetor;i++)
			{
				bandaInf[i] = 0.0f;
				bandaSup[i] = 0.0f;
			}
			break;
		}
		case 4://matriz diagonal = 1 e bandas diferente de 0
		{
			for(i=0;i<n;i++)
				diagonal[i] = 1.0f;

			for(i=0;i<TamVetor;i++)
			{
				bandaInf[i] = (float) i+3;
				bandaSup[i] = (float) i+2;
			}
			break;
		}
	}

    __m256 ymm0,ymm1,ymm2,ymm3,ymm4,ymm5,ymm6,ymm7,ymm8,ymm9,ymm10,ymm11,ymm12,ymm13,ymm14;
    int a,caso,quantidadeElementosVetorializar;

	if(EhIdentidade)
	{
		printf("Calculando ...\n");
		float somaDiagonal = 0;
		float somaBandas = 0;

		//------------------Soma da diagonal---------------------------

		if(run == 1)//vetorializado
		{
		    printf("Entrei no vetorializado\n");

			for(i = 0;i < n;i += 128)
			{
				quantidadeElementosVetorializar = (n - i);

				if(quantidadeElementosVetorializar >= 128)
				  caso = 0;

				if(quantidadeElementosVetorializar < 128 && quantidadeElementosVetorializar > 120)
				  caso = 1;

				if(quantidadeElementosVetorializar <= 120 && quantidadeElementosVetorializar > 112)
				  caso = 2;

				if(quantidadeElementosVetorializar <= 112 && quantidadeElementosVetorializar > 104)
				  caso = 3;

				if(quantidadeElementosVetorializar <= 104 && quantidadeElementosVetorializar > 96)
				  caso = 4;

				if(quantidadeElementosVetorializar <= 96 && quantidadeElementosVetorializar > 88)
				  caso = 5;

				if(quantidadeElementosVetorializar <= 88 && quantidadeElementosVetorializar > 80)
				  caso = 6;

				if(quantidadeElementosVetorializar <= 80 && quantidadeElementosVetorializar > 72)
				  caso = 7;

				if(quantidadeElementosVetorializar <= 72 && quantidadeElementosVetorializar > 64)
				  caso = 8;

				if(quantidadeElementosVetorializar <= 64 && quantidadeElementosVetorializar > 56)
				  caso = 9;

				if(quantidadeElementosVetorializar <= 56 && quantidadeElementosVetorializar > 48)
				  caso = 10;

				if(quantidadeElementosVetorializar <= 48 && quantidadeElementosVetorializar > 40)
				  caso = 11;

				if(quantidadeElementosVetorializar <= 40 && quantidadeElementosVetorializar > 32)
				  caso = 12;

				if(quantidadeElementosVetorializar <= 32 && quantidadeElementosVetorializar > 24)
				  caso = 13;

				if(quantidadeElementosVetorializar <= 24 && quantidadeElementosVetorializar > 16)
				  caso = 14;

				if(quantidadeElementosVetorializar <= 16 && quantidadeElementosVetorializar > 8)
				  caso = 15;

				if(quantidadeElementosVetorializar <= 8)
				  caso = 16;

				switch(caso)
				{
					case 0:
					{
						ymm0 = _mm256_add_ps(_mm256_loadu_ps(&diagonal[i]),_mm256_loadu_ps(&diagonal[i + 8]));
						ymm1 = _mm256_add_ps(_mm256_loadu_ps(&diagonal[i + 16]),_mm256_loadu_ps(&diagonal[i + 24]));
						ymm2 = _mm256_add_ps(_mm256_loadu_ps(&diagonal[i + 32]),_mm256_loadu_ps(&diagonal[i + 40]));
						ymm3 = _mm256_add_ps(_mm256_loadu_ps(&diagonal[i + 48]),_mm256_loadu_ps(&diagonal[i + 56]));
						ymm4 = _mm256_add_ps(_mm256_loadu_ps(&diagonal[i + 64]),_mm256_loadu_ps(&diagonal[i + 72]));
						ymm5 = _mm256_add_ps(_mm256_loadu_ps(&diagonal[i + 80]),_mm256_loadu_ps(&diagonal[i + 88]));
						ymm6 = _mm256_add_ps(_mm256_loadu_ps(&diagonal[i + 96]),_mm256_loadu_ps(&diagonal[i + 104]));
						ymm7 = _mm256_add_ps(_mm256_loadu_ps(&diagonal[i + 112]),_mm256_loadu_ps(&diagonal[i + 120]));

						ymm8 = _mm256_add_ps(ymm0,ymm1);
						ymm9 = _mm256_add_ps(ymm2,ymm3);
						ymm10 = _mm256_add_ps(ymm4,ymm5);
						ymm11 = _mm256_add_ps(ymm6,ymm7);

						ymm12 = _mm256_add_ps(ymm8,ymm9);
						ymm13 = _mm256_add_ps(ymm10,ymm11);

						ymm14 = _mm256_add_ps(ymm12,ymm13);

						for (a = 0; a < 8; a++)
						{
							somaDiagonal += ymm14[a];
						}

						break;
					}

					case 1:
					{
						ymm0 = _mm256_add_ps(_mm256_loadu_ps(&diagonal[i]),_mm256_loadu_ps(&diagonal[i + 8]));
						ymm1 = _mm256_add_ps(_mm256_loadu_ps(&diagonal[i + 16]),_mm256_loadu_ps(&diagonal[i + 24]));
						ymm2 = _mm256_add_ps(_mm256_loadu_ps(&diagonal[i + 32]),_mm256_loadu_ps(&diagonal[i + 40]));
						ymm3 = _mm256_add_ps(_mm256_loadu_ps(&diagonal[i + 48]),_mm256_loadu_ps(&diagonal[i + 56]));
						ymm4 = _mm256_add_ps(_mm256_loadu_ps(&diagonal[i + 64]),_mm256_loadu_ps(&diagonal[i + 72]));
						ymm5 = _mm256_add_ps(_mm256_loadu_ps(&diagonal[i + 80]),_mm256_loadu_ps(&diagonal[i + 88]));
						ymm6 = _mm256_add_ps(_mm256_loadu_ps(&diagonal[i + 96]),_mm256_loadu_ps(&diagonal[i + 104]));

						ymm8 = _mm256_add_ps(ymm0,ymm1);
						ymm9 = _mm256_add_ps(ymm2,ymm3);
						ymm10 = _mm256_add_ps(ymm4,ymm5);
						ymm11 = _mm256_add_ps(ymm6,_mm256_loadu_ps(&diagonal[i + 112]));

						ymm12 = _mm256_add_ps(ymm8,ymm9);
						ymm13 = _mm256_add_ps(ymm10,ymm11);

						ymm14 = _mm256_add_ps(ymm12,ymm13);

						for (a = 0; a < 8; a++)
						{
							somaDiagonal += ymm14[a];

							if((i + 120) + a < n)
								somaDiagonal += diagonal[i + 120 + a];
						}

						break;
					}

					case 2:
					{

						ymm0 = _mm256_add_ps(_mm256_loadu_ps(&diagonal[i]),_mm256_loadu_ps(&diagonal[i + 8]));
						ymm1 = _mm256_add_ps(_mm256_loadu_ps(&diagonal[i + 16]),_mm256_loadu_ps(&diagonal[i + 24]));
						ymm2 = _mm256_add_ps(_mm256_loadu_ps(&diagonal[i + 32]),_mm256_loadu_ps(&diagonal[i + 40]));
						ymm3 = _mm256_add_ps(_mm256_loadu_ps(&diagonal[i + 48]),_mm256_loadu_ps(&diagonal[i + 56]));
						ymm4 = _mm256_add_ps(_mm256_loadu_ps(&diagonal[i + 64]),_mm256_loadu_ps(&diagonal[i + 72]));
						ymm5 = _mm256_add_ps(_mm256_loadu_ps(&diagonal[i + 80]),_mm256_loadu_ps(&diagonal[i + 88]));
						ymm6 = _mm256_add_ps(_mm256_loadu_ps(&diagonal[i + 96]),_mm256_loadu_ps(&diagonal[i + 104]));

						ymm8 = _mm256_add_ps(ymm0,ymm1);
						ymm9 = _mm256_add_ps(ymm2,ymm3);
						ymm10 = _mm256_add_ps(ymm4,ymm5);

						ymm12 = _mm256_add_ps(ymm8,ymm9);
						ymm13 = _mm256_add_ps(ymm10,ymm6);

						ymm14 = _mm256_add_ps(ymm12,ymm13);

						for (a = 0; a < 8; a++)
						{
							somaDiagonal += ymm14[a];

							if((i + 112) + a < n)
								somaDiagonal += diagonal[i + 112 + a];
						}


						break;
					}

					case 3:
					{

						ymm0 = _mm256_add_ps(_mm256_loadu_ps(&diagonal[i]),_mm256_loadu_ps(&diagonal[i + 8]));
						ymm1 = _mm256_add_ps(_mm256_loadu_ps(&diagonal[i + 16]),_mm256_loadu_ps(&diagonal[i + 24]));
						ymm2 = _mm256_add_ps(_mm256_loadu_ps(&diagonal[i + 32]),_mm256_loadu_ps(&diagonal[i + 40]));
						ymm3 = _mm256_add_ps(_mm256_loadu_ps(&diagonal[i + 48]),_mm256_loadu_ps(&diagonal[i + 56]));
						ymm4 = _mm256_add_ps(_mm256_loadu_ps(&diagonal[i + 64]),_mm256_loadu_ps(&diagonal[i + 72]));
						ymm5 = _mm256_add_ps(_mm256_loadu_ps(&diagonal[i + 80]),_mm256_loadu_ps(&diagonal[i + 88]));

						ymm8 = _mm256_add_ps(ymm0,ymm1);
						ymm9 = _mm256_add_ps(ymm2,ymm3);
						ymm10 = _mm256_add_ps(ymm4,ymm5);
						ymm11 = _mm256_loadu_ps(&diagonal[i + 96]);

						ymm12 = _mm256_add_ps(ymm8,ymm9);
						ymm13 = _mm256_add_ps(ymm10,ymm11);

						ymm14 = _mm256_add_ps(ymm12,ymm13);

						for (a = 0; a < 8; a++)
						{
							somaDiagonal += ymm14[a];

							if((i + 104) + a < n)
								somaDiagonal += diagonal[i + 104 + a];
						}



						break;
					}

					case 4:
					{

						ymm0 = _mm256_add_ps(_mm256_loadu_ps(&diagonal[i]),_mm256_loadu_ps(&diagonal[i + 8]));
						ymm1 = _mm256_add_ps(_mm256_loadu_ps(&diagonal[i + 16]),_mm256_loadu_ps(&diagonal[i + 24]));
						ymm2 = _mm256_add_ps(_mm256_loadu_ps(&diagonal[i + 32]),_mm256_loadu_ps(&diagonal[i + 40]));
						ymm3 = _mm256_add_ps(_mm256_loadu_ps(&diagonal[i + 48]),_mm256_loadu_ps(&diagonal[i + 56]));
						ymm4 = _mm256_add_ps(_mm256_loadu_ps(&diagonal[i + 64]),_mm256_loadu_ps(&diagonal[i + 72]));
						ymm5 = _mm256_add_ps(_mm256_loadu_ps(&diagonal[i + 80]),_mm256_loadu_ps(&diagonal[i + 88]));

						ymm8 = _mm256_add_ps(ymm0,ymm1);
						ymm9 = _mm256_add_ps(ymm2,ymm3);
						ymm10 = _mm256_add_ps(ymm4,ymm5);

						ymm12 = _mm256_add_ps(ymm8,ymm9);

						ymm14 = _mm256_add_ps(ymm12,ymm10);

						for (a = 0; a < 8; a++)
						{
							somaDiagonal += ymm14[a];

							if((i + 96) + a < n)
								somaDiagonal += diagonal[i + 96 + a];
						}


						break;
					}

					case 5:
					{

						ymm0 = _mm256_add_ps(_mm256_loadu_ps(&diagonal[i]),_mm256_loadu_ps(&diagonal[i + 8]));
						ymm1 = _mm256_add_ps(_mm256_loadu_ps(&diagonal[i + 16]),_mm256_loadu_ps(&diagonal[i + 24]));
						ymm2 = _mm256_add_ps(_mm256_loadu_ps(&diagonal[i + 32]),_mm256_loadu_ps(&diagonal[i + 40]));
						ymm3 = _mm256_add_ps(_mm256_loadu_ps(&diagonal[i + 48]),_mm256_loadu_ps(&diagonal[i + 56]));
						ymm4 = _mm256_add_ps(_mm256_loadu_ps(&diagonal[i + 64]),_mm256_loadu_ps(&diagonal[i + 72]));

						ymm8 = _mm256_add_ps(ymm0,ymm1);
						ymm9 = _mm256_add_ps(ymm2,ymm3);
						ymm10 = _mm256_add_ps(ymm4,_mm256_loadu_ps(&diagonal[i + 80]));

						ymm12 = _mm256_add_ps(ymm8,ymm9);

						ymm14 = _mm256_add_ps(ymm12,ymm10);

						for (a = 0; a < 8; a++)
						{
							somaDiagonal += ymm14[a];

							if((i + 88) + a < n)
								somaDiagonal += diagonal[i + 88 + a];
						}

						break;
					}

					case 6:
					{
						ymm0 = _mm256_add_ps(_mm256_loadu_ps(&diagonal[i]),_mm256_loadu_ps(&diagonal[i + 8]));
						ymm1 = _mm256_add_ps(_mm256_loadu_ps(&diagonal[i + 16]),_mm256_loadu_ps(&diagonal[i + 24]));
						ymm2 = _mm256_add_ps(_mm256_loadu_ps(&diagonal[i + 32]),_mm256_loadu_ps(&diagonal[i + 40]));
						ymm3 = _mm256_add_ps(_mm256_loadu_ps(&diagonal[i + 48]),_mm256_loadu_ps(&diagonal[i + 56]));
						ymm4 = _mm256_add_ps(_mm256_loadu_ps(&diagonal[i + 64]),_mm256_loadu_ps(&diagonal[i + 72]));

						ymm8 = _mm256_add_ps(ymm0,ymm1);
						ymm9 = _mm256_add_ps(ymm2,ymm3);

						ymm12 = _mm256_add_ps(ymm8,ymm9);

						ymm14 = _mm256_add_ps(ymm12,ymm4);

						for (a = 0; a < 8; a++)
						{
							somaDiagonal += ymm14[a];

							if((i + 80) + a < n)
								somaDiagonal += diagonal[i + 80 + a];
						}
						break;
					}

					case 7:
					{
						ymm0 = _mm256_add_ps(_mm256_loadu_ps(&diagonal[i]),_mm256_loadu_ps(&diagonal[i + 8]));
						ymm1 = _mm256_add_ps(_mm256_loadu_ps(&diagonal[i + 16]),_mm256_loadu_ps(&diagonal[i + 24]));
						ymm2 = _mm256_add_ps(_mm256_loadu_ps(&diagonal[i + 32]),_mm256_loadu_ps(&diagonal[i + 40]));
						ymm3 = _mm256_add_ps(_mm256_loadu_ps(&diagonal[i + 48]),_mm256_loadu_ps(&diagonal[i + 56]));

						ymm8 = _mm256_add_ps(ymm0,ymm1);
						ymm9 = _mm256_add_ps(ymm2,ymm3);
						ymm10 = _mm256_loadu_ps(&diagonal[i + 64]);

						ymm12 = _mm256_add_ps(ymm8,ymm9);

						ymm14 = _mm256_add_ps(ymm12,ymm10);

						for (a = 0; a < 8; a++)
						{
							somaDiagonal += ymm14[a];

							if((i + 72) + a < n)
								somaDiagonal += diagonal[i + 72 + a];
						}
						break;
					}

					case 8:
					{
						ymm0 = _mm256_add_ps(_mm256_loadu_ps(&diagonal[i]),_mm256_loadu_ps(&diagonal[i + 8]));
						ymm1 = _mm256_add_ps(_mm256_loadu_ps(&diagonal[i + 16]),_mm256_loadu_ps(&diagonal[i + 24]));
						ymm2 = _mm256_add_ps(_mm256_loadu_ps(&diagonal[i + 32]),_mm256_loadu_ps(&diagonal[i + 40]));
						ymm3 = _mm256_add_ps(_mm256_loadu_ps(&diagonal[i + 48]),_mm256_loadu_ps(&diagonal[i + 56]));

						ymm8 = _mm256_add_ps(ymm0,ymm1);
						ymm9 = _mm256_add_ps(ymm2,ymm3);

						ymm12 = _mm256_add_ps(ymm8,ymm9);

						for (a = 0; a < 8; a++)
						{
							somaDiagonal += ymm12[a];

							if((i + 64) + a < n)
								somaDiagonal += diagonal[i + 64 + a];
						}
						break;
					}

					case 9:
					{
						ymm0 = _mm256_add_ps(_mm256_loadu_ps(&diagonal[i]),_mm256_loadu_ps(&diagonal[i + 8]));
						ymm1 = _mm256_add_ps(_mm256_loadu_ps(&diagonal[i + 16]),_mm256_loadu_ps(&diagonal[i + 24]));
						ymm2 = _mm256_add_ps(_mm256_loadu_ps(&diagonal[i + 32]),_mm256_loadu_ps(&diagonal[i + 40]));

						ymm8 = _mm256_add_ps(ymm0,ymm1);
						ymm9 = _mm256_add_ps(ymm2,_mm256_loadu_ps(&diagonal[i + 48]));

						ymm12 = _mm256_add_ps(ymm8,ymm9);

						for (a = 0; a < 8; a++)
						{
							somaDiagonal += ymm12[a];

							if((i + 56) + a < n)
								somaDiagonal += diagonal[i + 56 + a];
						}
						break;
					}

					case 10:
					{
						ymm0 = _mm256_add_ps(_mm256_loadu_ps(&diagonal[i]),_mm256_loadu_ps(&diagonal[i + 8]));
						ymm1 = _mm256_add_ps(_mm256_loadu_ps(&diagonal[i + 16]),_mm256_loadu_ps(&diagonal[i + 24]));
						ymm2 = _mm256_add_ps(_mm256_loadu_ps(&diagonal[i + 32]),_mm256_loadu_ps(&diagonal[i + 40]));

						ymm8 = _mm256_add_ps(ymm0,ymm1);

						ymm12 = _mm256_add_ps(ymm8,ymm2);

						for (a = 0; a < 8; a++)
						{
							somaDiagonal += ymm12[a];

							if((i + 48) + a < n)
								somaDiagonal += diagonal[i + 48 + a];
						}
						break;
					}

					case 11:
					{
						ymm0 = _mm256_add_ps(_mm256_loadu_ps(&diagonal[i]),_mm256_loadu_ps(&diagonal[i + 8]));
						ymm1 = _mm256_add_ps(_mm256_loadu_ps(&diagonal[i + 16]),_mm256_loadu_ps(&diagonal[i + 24]));
						ymm2 = _mm256_add_ps(_mm256_loadu_ps(&diagonal[i + 32]),_mm256_loadu_ps(&diagonal[i + 40]));

						ymm8 = _mm256_add_ps(ymm0,ymm1);
						ymm9 = _mm256_loadu_ps(&diagonal[i + 32]);

						ymm12 = _mm256_add_ps(ymm8,ymm9);

						for (a = 0; a < 8; a++)
						{
							somaDiagonal += ymm12[a];

							if((i + 40) + a < n)
								somaDiagonal += diagonal[i + 40 + a];
						}
						break;
					}

					case 12:
					{
						ymm0 = _mm256_add_ps(_mm256_loadu_ps(&diagonal[i]),_mm256_loadu_ps(&diagonal[i + 8]));
						ymm1 = _mm256_add_ps(_mm256_loadu_ps(&diagonal[i + 16]),_mm256_loadu_ps(&diagonal[i + 24]));

						ymm8 = _mm256_add_ps(ymm0,ymm1);

						for (a = 0; a < 8; a++)
						{
							somaDiagonal += ymm8[a];

							if((i + 32) + a < n)
								somaDiagonal += diagonal[i + 32 + a];
						}
						break;
					}

					case 13:
					{
						ymm0 = _mm256_add_ps(_mm256_loadu_ps(&diagonal[i]),_mm256_loadu_ps(&diagonal[i + 8]));

						ymm8 = _mm256_add_ps(ymm0,_mm256_loadu_ps(&diagonal[i + 16]));

						for (a = 0; a < 8; a++)
						{
							somaDiagonal += ymm8[a];

							if((i + 24) + a < n)
								somaDiagonal += diagonal[i + 24 + a];
						}
						break;
					}

					case 14:
					{
						ymm0 = _mm256_add_ps(_mm256_loadu_ps(&diagonal[i]),_mm256_loadu_ps(&diagonal[i + 8]));

						for (a = 0; a < 8; a++)
						{
							somaDiagonal += ymm0[a];

							if((i + 16) + a < n)
								somaDiagonal += diagonal[i + 16 + a];
						}
						break;
					}

					case 15:
					{
						ymm0 = _mm256_loadu_ps(&diagonal[i]);

						for (a = 0; a < 8; a++)
						{
							somaDiagonal += ymm0[a];

							if((i + 8) + a < n)
								somaDiagonal += diagonal[i + 8 + a];
						}
						break;
					}

					case 16:
					{
						ymm0 = _mm256_loadu_ps(&diagonal[i]);

						for (a = 0; a < quantidadeElementosVetorializar; a++)
						{
							somaDiagonal += ymm0[a];
						}
						break;
					}
				}//SWITCH
			}//FOR
		}
		else //sequencial
		{
		    printf("Entrei no sequêncial\n");

		    for(i=0;i<n;i++)
		    {
			    somaDiagonal += diagonal[i];
		    }
		}

		//printf("diagonal = %.2f\n", somaDiagonal);

		//--------------------------------------------------------------------

		//------------------------------Soma bandas---------------------------
		if(somaDiagonal == n)
		{
		  if(run == 1) //vetorializado
		  {
		      //printf("Entrei no vetorializado\n");

		      for(i = 0;i < TamVetor;i += 64)
		      {
					quantidadeElementosVetorializar = (TamVetor - i);

					if(quantidadeElementosVetorializar >= 64)
					  caso = 0;

					if(quantidadeElementosVetorializar < 64 && quantidadeElementosVetorializar > 56)
					  caso = 1;

					if(quantidadeElementosVetorializar <= 56 && quantidadeElementosVetorializar > 48)
					  caso = 2;

					if(quantidadeElementosVetorializar <= 48 && quantidadeElementosVetorializar > 40)
					  caso = 3;

					if(quantidadeElementosVetorializar <= 40 && quantidadeElementosVetorializar > 32)
					  caso = 4;

					if(quantidadeElementosVetorializar <= 32 && quantidadeElementosVetorializar > 24)
					  caso = 5;

					if(quantidadeElementosVetorializar <= 24 && quantidadeElementosVetorializar > 16)
					  caso = 6;

					if(quantidadeElementosVetorializar <= 16 && quantidadeElementosVetorializar > 8)
					  caso = 7;

					if(quantidadeElementosVetorializar <= 8)
					  caso = 8;

					switch(caso)
					{
						case 0:
						{
							ymm0 = _mm256_add_ps(_mm256_loadu_ps(&bandaSup[i]),_mm256_loadu_ps(&bandaInf[i]));
							ymm1 = _mm256_add_ps(_mm256_loadu_ps(&bandaSup[i + 8]),_mm256_loadu_ps(&bandaInf[i + 8]));
							ymm2 = _mm256_add_ps(_mm256_loadu_ps(&bandaSup[i + 16]),_mm256_loadu_ps(&bandaInf[i + 16]));
							ymm3 = _mm256_add_ps(_mm256_loadu_ps(&bandaSup[i + 24]),_mm256_loadu_ps(&bandaInf[i + 24]));
							ymm4 = _mm256_add_ps(_mm256_loadu_ps(&bandaSup[i + 32]),_mm256_loadu_ps(&bandaInf[i + 32]));
							ymm5 = _mm256_add_ps(_mm256_loadu_ps(&bandaSup[i + 40]),_mm256_loadu_ps(&bandaInf[i + 40]));
							ymm6 = _mm256_add_ps(_mm256_loadu_ps(&bandaSup[i + 48]),_mm256_loadu_ps(&bandaInf[i + 48]));
							ymm7 = _mm256_add_ps(_mm256_loadu_ps(&bandaSup[i + 56]),_mm256_loadu_ps(&bandaInf[i + 56]));

							ymm8 = _mm256_add_ps(ymm0,ymm1);
							ymm9 = _mm256_add_ps(ymm2,ymm3);
							ymm10 = _mm256_add_ps(ymm4,ymm5);
							ymm11 = _mm256_add_ps(ymm6,ymm7);

							ymm12 = _mm256_add_ps(ymm8,ymm9);
							ymm13 = _mm256_add_ps(ymm10,ymm11);

							ymm14 = _mm256_add_ps(ymm12,ymm13);

							for (a = 0; a < 8; a++)
							{
								somaBandas += ymm14[a];
							}

							break;
						}

						case 1:
						{
							ymm0 = _mm256_add_ps(_mm256_loadu_ps(&bandaSup[i]),_mm256_loadu_ps(&bandaInf[i]));
							ymm1 = _mm256_add_ps(_mm256_loadu_ps(&bandaSup[i + 8]),_mm256_loadu_ps(&bandaInf[i + 8]));
							ymm2 = _mm256_add_ps(_mm256_loadu_ps(&bandaSup[i + 16]),_mm256_loadu_ps(&bandaInf[i + 16]));
							ymm3 = _mm256_add_ps(_mm256_loadu_ps(&bandaSup[i + 24]),_mm256_loadu_ps(&bandaInf[i + 24]));
							ymm4 = _mm256_add_ps(_mm256_loadu_ps(&bandaSup[i + 32]),_mm256_loadu_ps(&bandaInf[i + 32]));
							ymm5 = _mm256_add_ps(_mm256_loadu_ps(&bandaSup[i + 40]),_mm256_loadu_ps(&bandaInf[i + 40]));
							ymm6 = _mm256_add_ps(_mm256_loadu_ps(&bandaSup[i + 48]),_mm256_loadu_ps(&bandaInf[i + 48]));
							ymm7 = _mm256_add_ps(_mm256_loadu_ps(&bandaSup[i + 56]),_mm256_loadu_ps(&bandaInf[i + 56]));

							ymm8 = _mm256_add_ps(ymm0,ymm1);
							ymm9 = _mm256_add_ps(ymm2,ymm3);
							ymm10 = _mm256_add_ps(ymm4,ymm5);

							ymm12 = _mm256_add_ps(ymm8,ymm9);
							ymm13 = _mm256_add_ps(ymm10,ymm6);

							ymm14 = _mm256_add_ps(ymm12,ymm13);

							for (a = 0; a < 8; a++)
							{
								somaBandas += ymm14[a];

								if((i + 56) + a < TamVetor)
									somaBandas += ymm7[a];
							}

							break;
						}

						case 2:
						{
							ymm0 = _mm256_add_ps(_mm256_loadu_ps(&bandaSup[i]),_mm256_loadu_ps(&bandaInf[i]));
							ymm1 = _mm256_add_ps(_mm256_loadu_ps(&bandaSup[i + 8]),_mm256_loadu_ps(&bandaInf[i + 8]));
							ymm2 = _mm256_add_ps(_mm256_loadu_ps(&bandaSup[i + 16]),_mm256_loadu_ps(&bandaInf[i + 16]));
							ymm3 = _mm256_add_ps(_mm256_loadu_ps(&bandaSup[i + 24]),_mm256_loadu_ps(&bandaInf[i + 24]));
							ymm4 = _mm256_add_ps(_mm256_loadu_ps(&bandaSup[i + 32]),_mm256_loadu_ps(&bandaInf[i + 32]));
							ymm5 = _mm256_add_ps(_mm256_loadu_ps(&bandaSup[i + 40]),_mm256_loadu_ps(&bandaInf[i + 40]));
							ymm6 = _mm256_add_ps(_mm256_loadu_ps(&bandaSup[i + 48]),_mm256_loadu_ps(&bandaInf[i + 48]));

							ymm8 = _mm256_add_ps(ymm0,ymm1);
							ymm9 = _mm256_add_ps(ymm2,ymm3);
							ymm10 = _mm256_add_ps(ymm4,ymm5);

							ymm12 = _mm256_add_ps(ymm8,ymm9);

							ymm14 = _mm256_add_ps(ymm12,ymm10);

							for (a = 0; a < 8; a++)
							{
								somaBandas += ymm14[a];

								if((i + 48) + a < TamVetor)
									somaBandas += ymm6[a];
							}

							break;
						}

						case 3:
						{
							ymm0 = _mm256_add_ps(_mm256_loadu_ps(&bandaSup[i]),_mm256_loadu_ps(&bandaInf[i]));
							ymm1 = _mm256_add_ps(_mm256_loadu_ps(&bandaSup[i + 8]),_mm256_loadu_ps(&bandaInf[i + 8]));
							ymm2 = _mm256_add_ps(_mm256_loadu_ps(&bandaSup[i + 16]),_mm256_loadu_ps(&bandaInf[i + 16]));
							ymm3 = _mm256_add_ps(_mm256_loadu_ps(&bandaSup[i + 24]),_mm256_loadu_ps(&bandaInf[i + 24]));
							ymm4 = _mm256_add_ps(_mm256_loadu_ps(&bandaSup[i + 32]),_mm256_loadu_ps(&bandaInf[i + 32]));
							ymm5 = _mm256_add_ps(_mm256_loadu_ps(&bandaSup[i + 40]),_mm256_loadu_ps(&bandaInf[i + 40]));

							ymm8 = _mm256_add_ps(ymm0,ymm1);
							ymm9 = _mm256_add_ps(ymm2,ymm3);

							ymm12 = _mm256_add_ps(ymm8,ymm9);

							ymm14 = _mm256_add_ps(ymm12,ymm4);

							for (a = 0; a < 8; a++)
							{
								somaBandas += ymm14[a];

								if((i + 40) + a < TamVetor)
									somaBandas += ymm5[a];
							}

							break;
						}

						case 4:
						{
							ymm0 = _mm256_add_ps(_mm256_loadu_ps(&bandaSup[i]),_mm256_loadu_ps(&bandaInf[i]));
							ymm1 = _mm256_add_ps(_mm256_loadu_ps(&bandaSup[i + 8]),_mm256_loadu_ps(&bandaInf[i + 8]));
							ymm2 = _mm256_add_ps(_mm256_loadu_ps(&bandaSup[i + 16]),_mm256_loadu_ps(&bandaInf[i + 16]));
							ymm3 = _mm256_add_ps(_mm256_loadu_ps(&bandaSup[i + 24]),_mm256_loadu_ps(&bandaInf[i + 24]));
							ymm4 = _mm256_add_ps(_mm256_loadu_ps(&bandaSup[i + 32]),_mm256_loadu_ps(&bandaInf[i + 32]));

							ymm8 = _mm256_add_ps(ymm0,ymm1);
							ymm9 = _mm256_add_ps(ymm2,ymm3);

							ymm12 = _mm256_add_ps(ymm8,ymm9);

							for (a = 0; a < 8; a++)
							{
								somaBandas += ymm12[a];

								if((i + 32) + a < TamVetor)
									somaBandas += ymm4[a];
							}


							break;
						}

						case 5:
						{
							ymm0 = _mm256_add_ps(_mm256_loadu_ps(&bandaSup[i]),_mm256_loadu_ps(&bandaInf[i]));
							ymm1 = _mm256_add_ps(_mm256_loadu_ps(&bandaSup[i + 8]),_mm256_loadu_ps(&bandaInf[i + 8]));
							ymm2 = _mm256_add_ps(_mm256_loadu_ps(&bandaSup[i + 16]),_mm256_loadu_ps(&bandaInf[i + 16]));
							ymm3 = _mm256_add_ps(_mm256_loadu_ps(&bandaSup[i + 24]),_mm256_loadu_ps(&bandaInf[i + 24]));

							ymm8 = _mm256_add_ps(ymm0,ymm1);

							ymm12 = _mm256_add_ps(ymm8,ymm2);

							for (a = 0; a < 8; a++)
							{
								somaBandas += ymm12[a];

								if((i + 24) + a < TamVetor)
									somaBandas += ymm3[a];
							}

							break;
						}

						case 6:
						{
							ymm0 = _mm256_add_ps(_mm256_loadu_ps(&bandaSup[i]),_mm256_loadu_ps(&bandaInf[i]));
							ymm1 = _mm256_add_ps(_mm256_loadu_ps(&bandaSup[i + 8]),_mm256_loadu_ps(&bandaInf[i + 8]));
							ymm2 = _mm256_add_ps(_mm256_loadu_ps(&bandaSup[i + 16]),_mm256_loadu_ps(&bandaInf[i + 16]));

							ymm8 = _mm256_add_ps(ymm0,ymm1);

							for (a = 0; a < 8; a++)
							{
								somaBandas += ymm8[a];

								if((i + 16) + a < TamVetor)
									somaBandas += ymm2[a];
							}


							break;
						}

						case 7:
						{
							ymm0 = _mm256_add_ps(_mm256_loadu_ps(&bandaSup[i]),_mm256_loadu_ps(&bandaInf[i]));
							ymm1 = _mm256_add_ps(_mm256_loadu_ps(&bandaSup[i + 8]),_mm256_loadu_ps(&bandaInf[i + 8]));

							for (a = 0; a < 8; a++)
							{
								somaBandas += ymm0[a];

								if((i + 8) + a < TamVetor)
									somaBandas += ymm1[a];
							}


							break;
						}

						case 8:
						{
							ymm0 = _mm256_add_ps(_mm256_loadu_ps(&bandaSup[i]),_mm256_loadu_ps(&bandaInf[i]));

							for (a = 0; a < quantidadeElementosVetorializar; a++)
							{
								somaBandas += ymm0[a];
							}

							break;
						}
					}//SWITCH
				}//FOR
		  }//IF
		  else //sequencial
		  {
				for(i=0;i<TamVetor;i++)
				{
				    somaBandas += bandaSup[i] + bandaInf[i];
				}
		  }
			if(somaBandas != 0)
				printf("não é identidade\n");
			else
			{
				//printf("bandas = %.2f\n", somaBandas);
				printf("é identidade\n");
			}
		}
		else
			printf("não é identidade\n");
	}

	_mm_free(bandaInf);
	_mm_free(bandaSup);
	_mm_free(diagonal);

	return 0;
}

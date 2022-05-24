#ifndef __DFENET__
#define __DFENET__

#include <stdio.h>
#include <string.h>
#include <math.h>

#define MIN(x,y) ((x)<(y)?(x):(y))
#define MAX(x,y) ((x)>(y)?(x):(y))

#define IMG_HEIGHT 240
#define IMG_WIDTH 320
#define S 1
#define OnChipIB_Width  ((Tr-1)*S+K)
#define OnChipIB_Height ((Tc-1)*S+K) 
#define K 3 //Ksize
#define Tn 16
#define Tm 16
#define Tr 40 //(320/8)
#define Tc 30 //(240/8)
#define layer_num 10 

void DFENet_FPGA(float *Input, float *Output, float *Weight0, float *Weight1, float *Bias0, 
                 float *Bias1, const int TR, const int TC, const int TRow, const int TCol);

#endif
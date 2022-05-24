#ifndef __DFENET__
#define __DFENET__
#include <ap_int.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

typedef ap_int<8> int8;
typedef ap_uint<8> uint8;
typedef ap_int<16> int16;
typedef ap_uint<16> uint16;
typedef ap_int<32> int32;
typedef ap_int<32> uint32;

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

void DFENet_FPGA(int8 *Input, int32 *Output, int8 *Weight0, int8 *Weight1, int8 *Bias0, int8 *Bias1, uint32 TR,  
                 uint32 TC, uint32 TRow, uint32 TCol, int *WeightQ, int *BiasQ, int *InputQ, int *InterWidthQ);

#endif
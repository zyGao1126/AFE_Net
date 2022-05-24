#include <iostream>
#include "DFENet.h"

//load ini image block from DDR to feature_buffer
//input:[1,240,320] 
//r, c means how many blocks have passed
//here TRow = OnChipIB_Width; TCol = OnChipIB_Height; the difference between TRow and TR is whether to add padding
void load_input(int8 *Input, int32 feature_buffer[Tn][OnChipIB_Width][OnChipIB_Height],
                int TR, int TC, int r, int c, int TRow, int TCol) 
{
    int load_offset;
    for (int t2 = 0; t2 < TRow; t2++)
        for (int t3 = 0; t3 < TCol; t3++) {
#pragma HLS PIPELINE                       
            //reach two bound, two padding
            if ((t2 == 0) || (t2 == TRow - 1) || (t3 == 0) || (t3 == TCol - 1))
                feature_buffer[0][t2][t3] = 0;
            else {
                //since input can not reach padding
                load_offset = (r * TR * IMG_HEIGHT) + (c * TR * TC) + ((t2 - 1) * TC) + (t3 - 1);
                feature_buffer[0][t2][t3] = Input[load_offset];
            }
        }
}

// load weight from DDR to weight_buffer
void load_weight_3x3(int8 *Weight, int8 weight_buffer[Tm][Tn][K][K], int TMP_L, int TN) 
{
    int tm_offset;
    //since TM always equal to 16
    int TM = 16;
    if (TMP_L == 0)
        tm_offset = 0;
    else 
        tm_offset = (1*16*3*3) + (TMP_L-1) * (16*16*3*3);
        
    for (int t3 = 0; t3 < TM; t3++) 
        for (int t4 = 0; t4 < TN; t4++)  
            for (int t1 = 0; t1 < K; t1++)
                for (int t2 = 0; t2 < K; t2++) {    
#pragma HLS PIPELINE                                   
                    int block_m_offset = t3 * TN * K * K;
                    int block_n_offset = t4 * K * K;
                    //previous block + current block m + current block n + kernel 
                    weight_buffer[t3][t4][t1][t2] = Weight[tm_offset + block_m_offset + block_n_offset + t1 * K + t2];
                }
}

void load_weight_1x1(int8 *Weight, int8 weight_buffer[Tn], int TMP_L)
{
    int tm_offset = (1*16*3*3 + 9*16*16*3*3) + (TMP_L-1)*(1*16*1*1);
    int TN = 16;
    for (int t = 0; t < TN; t++)
#pragma HLS PIPELINE    
        weight_buffer[t] = Weight[tm_offset + t];
}

void load_bias_3x3(int8 *Bias, int8 bias_buffer[Tm], int TMP_L, int BiasQ, int InterWidthQ)
{
    int TM = 16;
    int bias_offset = TMP_L * 16;
    for (int t = 0; t < TM; t++) 
#pragma HLS PIPELINE    
        bias_buffer[t] = Bias[bias_offset + t] >> (BiasQ - InterWidthQ);
}

int32 load_bias_1x1(int8 *Bias, int TMP_L, int BiasQ, int InterWidthQ)
{
    int bias_offset = 10*16 + (TMP_L-1) * 1;
    //std::cout << "bias buffer: " << bias_buffer << std::endl;
    return Bias[bias_offset] >> (BiasQ - InterWidthQ);
}

//DDR should store [240x320x10], so a variable which describes how many block has passed should be added;
void write_back_output(int32 output_buffer[Tr][Tc], int32 *Output, int r, int c, const int TR_MIN, const int TC_MIN, int TMP_L) 
{       
    int write_offset;
    const int OHxOW = IMG_WIDTH * IMG_HEIGHT;
    for (int tr = 0; tr < TR_MIN; tr++) {
        //here TM = 1;
        for (int tc = 0; tc < TC_MIN; tc++) {
            // output dimension + single whole image dimension + single whole image dimension + block dimension + block dimension;
            write_offset = (TMP_L - 1) * OHxOW + (r * TR_MIN * IMG_HEIGHT) + (c * TR_MIN * TC_MIN) + tr * TC_MIN + tc;
            // std::cout << "write offset: " << write_offset << std:endl;
            Output[write_offset] = output_buffer[tr][tc];
        }
    }
#ifdef TEST_FPGA_OUTPUT
    std:cout << "-------check Output---------" << std:endl;
    int count_num = Tr * Tc;
    int cont = 0;
    for (int pixel = 0; pixel < count_num; pixel++) 
    {
        cont++;
        std:cout << Output[(TMP_L - 1) * 4800 + pixel] << '\t';
        if (cont % 4 == 0)
            std:cout << std:endl;
    }
    std:cout << "----------------------------" << std:endl;
#endif //TEST_FPGA_OUTPUT
}

int16 BitPEs(int8 w_8b, int8 in_8b, int weightQ, int inputQ, int interwidthQ)
{
	int16 res1 = (w_8b * in_8b) >> (weightQ + inputQ - interwidthQ);
	return res1;
}

int32 Conver_3x3_split(int8 w[K][K], int8 in_0, int8 in_1, int8 in_2, int8 in_3, int8 in_4, 
                            int8 in_5, int8 in_6, int8 in_7, int8 in_8, int WeightQ, int InputQ, int InterWidthQ)
{
	int16 mul0, mul1, mul2, mul3, mul4, mul5, mul6, mul7, mul8;
	int32 add0, add1, add2;

#pragma HLS PIPELINE
	mul0 = BitPEs((int8)w_8b[0][0], in_0, WeightQ, InputQ, InterWidthQ);
	mul1 = BitPEs((int8)w_8b[0][1], in_1, WeightQ, InputQ, InterWidthQ);
	mul2 = BitPEs((int8)w_8b[0][2], in_2, WeightQ, InputQ, InterWidthQ);
	mul3 = BitPEs((int8)w_8b[1][0], in_3, WeightQ, InputQ, InterWidthQ);
	mul4 = BitPEs((int8)w_8b[1][1], in_4, WeightQ, InputQ, InterWidthQ);
	mul5 = BitPEs((int8)w_8b[1][2], in_5, WeightQ, InputQ, InterWidthQ);
	mul6 = BitPEs((int8)w_8b[2][0], in_6, WeightQ, InputQ, InterWidthQ);
	mul7 = BitPEs((int8)w_8b[2][1], in_7, WeightQ, InputQ, InterWidthQ);
	mul8 = BitPEs((int8)w_8b[2][2], in_8, WeightQ, InputQ, InterWidthQ);

	add0 = mul0 + mul1 + mul2;
	add1 = mul3 + mul4 + mul5;
	add2 = mul6 + mul7 + mul8;

	return add0 + add1 + add2;
}

//read data from feature_buffer; compute 3x3 conv and save current result; add current result and last result in feature_buffer;
void Conv3x3_sum(int8 *Weight, int8 *Bias, int32 feature_buffer_read[Tn][OnChipIB_Width][OnChipIB_Height], int32 feature_buffer_write_current[Tm][OnChipIB_Width][OnChipIB_Height], 
                 int32 feature_buffer_write_add[Tn][OnChipIB_Width][OnChipIB_Height], int8 weight_buffer[Tm][Tn][K][K], int8 bias_buffer[Tm],
                 int TMP_L, int TN, int TR_MIN, int TC_MIN, int TRow, int TCol, bool IsReLU, int inputQ, int weightQ, int biasQ, int interwidthQ) 
{
#pragma HLS ARRAY_PARTITION variable=feature_buffer_read complete dim=1
#pragma HLS ARRAY_PARTITION variable=feature_buffer_write_current complete dim=1
#pragma HLS ARRAY_PARTITION variable=feature_buffer_write_add complete dim=1    

    // when TMP_L = 10, conv_3x3 has finished
    if (TMP_L == 10)
        return;

    load_weight_3x3(Weight, weight_buffer, TMP_L, TN);
    load_bias_3x3(Bias, bias_buffer, TMP_L, biasQ, interwidthQ);

#ifdef TEST_WEIGHT_3x3
    std::cout << "-------check load weight 3x3---------" << std::endl;
    for (int mm = 0; mm < TM; mm++) {
        int cont = 0;
        for (int nn = 0; nn < TN; nn++) {
            for (int k = 0; k < 3; k++) {
                std::cout << weight_buffer[mm][nn][k][0] << '\t' << weight_buffer[mm][nn][k][1] << '\t' << weight_buffer[mm][nn][k][2] << '\t';
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
    }
    std::cout << "--------------------------------" << std::endl;
#endif //TEST_WEIGHT_3x3

#ifdef TEST_BIAS_3x3
    std::cout << "-------check load bias 3x3---------" << std::endl;
    for (int mm = 0; mm < TM; mm++) {
        std::cout << bias_buffer[mm] << std::endl;
    }
    std::cout << "--------------------------------" << std::endl;
#endif //TEST_BIAS_3x3

    static int32 partial_add[Tm]; 
#pragma HLS ARRAY_PARTITION variable=partial_add complete dim=1    
    static int16 partial_mul[Tm][Tn]; 
#pragma HLS ARRAY_PARTITION variable=partial_mul complete dim=1
#pragma HLS ARRAY_PARTITION variable=partial_mul complete dim=2

    static int8 window_buffer[Tn][K][K]; 
#pragma HLS ARRAY_PARTITION variable=window_buffer complete dim=1
#pragma HLS ARRAY_PARTITION variable=window_buffer complete dim=2
#pragma HLS ARRAY_PARTITION variable=window_buffer complete dim=3    
    static int8 line_buffer[Tn][K - 1][OnChipIB_Width];

    int tm, tc, tr, tn;
    int i, j, m;

    //add padding
    for (m = 0; m < TM_6b; m++) {
        for (i = 0; i < TRow_6b; i++) {
#pragma HLS UNROLL            
            feature_buffer_write_current[m][i][0] = 0;
            feature_buffer_write_current[m][i][TCol - 1] = 0;
            feature_buffer_write_add[m][i][0] = feature_buffer_write_current[m][i][0] + feature_buffer_read[m][i][0];
            feature_buffer_write_add[m][i][TCol - 1] = feature_buffer_write_current[m][i][TCol - 1] + feature_buffer_read[m][i][TCol - 1];
        }

        for (j = 0; j < TCol_6b; j++) {
#pragma HLS UNROLL             
            feature_buffer_write_current[m][0][j] = 0;
            feature_buffer_write_current[m][TRow - 1][j] = 0;
            feature_buffer_write_add[m][0][j] = feature_buffer_write_current[m][0][j] + feature_buffer_read[m][0][j];
            feature_buffer_write_add[m][TRow - 1][j] = feature_buffer_write_current[m][TRow - 1][j] + feature_buffer_read[m][TRow - 1][j];
        } 
    }  
    
    // initial tr should consider padding
    for (tr = 1; tr < TR_MIN_6b + 1; tr++) 
    {
        for (tc = 1; tc < TC_MIN_6b + 1; tc++) 
        {
#pragma HLS PIPELINE
#pragma HLS DEPENDENCE variable = feature_buffer_write_current inter false
#pragma HLS DEPENDENCE variable = line_buffer inter false
#pragma HLS ALLOCATION instances = Conver_3x3_split limit=256 function                 
            for (tn = 0; tn < TN_6b; tn++) {
                // update winfow buffer
                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 2; j++)
                        window_buffer[tn][i][j] = window_buffer[tn][i][j + 1];

                window_buffer[tn][0][2] = (line_buffer[tn][0][tc]);
                window_buffer[tn][1][2] = (line_buffer[tn][0][tc] = line_buffer[tn][1][tc]);
                window_buffer[tn][2][2] = (line_buffer[tn][1][tc] = feature_buffer_read[tn][tr][tc]);
            }

            // make sure preload the first two lines
            if (tr <= 1 || tc <= 1) ;
            else 
            { 
                for (tm = 0; tm < TM_6b; tm++) 
                {
                    int32 partial_sum = 0;
                    //since input channel has only one block, need not consider other condition
                    partial_add[tm] = bias_buffer[tm];
                    for (tn = 0; tn < TN_6b; tn++) {                        
                        partial_mul[tm][tn] = Conver_3x3_split(weight_buffer[tm][tn], window_buffer[tn][0][0], window_buffer[tn][0][1], 
                                                               window_buffer[tn][0][2], window_buffer[tn][1][0], window_buffer[tn][1][1], 
                                                               window_buffer[tn][1][2], window_buffer[tn][2][0], window_buffer[tn][2][1], 
                                                               window_buffer[tn][2][2]);
                        partial_sum += partial_mul[tm][tn];
                    }
                    feature_buffer_write_current[tm][tr][tc] = partial_add[tm] + partial_sum;
                    //ReLU
                    feature_buffer_write_current[tm][tr][tc] = ((feature_buffer_write_current[tm][tr][tc] < 0 && IsReLU) ? 0 : feature_buffer_write_current[tm][tr][tc]);    

                    if (TMP_L == 0)
                        feature_buffer_write_add[tm][tr][tc] = feature_buffer_write_current[tm][tr][tc];
                    //here since arrays of the same dimension can be added, feature_buffer_read can be tm not tn;
                    else
                        feature_buffer_write_add[tm][tr][tc] = feature_buffer_write_current[tm][tr][tc] + feature_buffer_read[tm][tr][tc];
                }
            }
        }           
    }
}

//read data from feature_buffer and compute conv_1x1; write result into DDR;  
//note that here feature_buffer_read is not the same in Conv3x3_sum;
void Conv1x1_write_back(int8 *Output, int8 *Weight, int8 *Bias, int32 weight_buffer[Tn], int32 output_buffer[Tr][Tc], 
                        int32 feature_buffer_read[Tm][OnChipIB_Width][OnChipIB_Height], int TMP_L, int TR_MIN, int TC_MIN, 
                        int TN, int r, int c, int InputQ, int WeightQ, int BiasQ, int InterWidthQ)
{  
#pragma HLS ARRAY_PARTITION variable=feature_buffer_read complete dim=1

    //in first loop it has nothing to write back 
    if (TMP_L == 0)
        return;

    int32 bias;
    load_weight_1x1(Weight, weight_buffer, TMP_L);
    bias = load_bias_1x1(Bias, TMP_L, BiasQ, InterWidthQ);

#ifdef TEST_WEIGHT_1x1
    std::cout << "-------check load weight 1x1---------" << std::endl;
    for (int nn = 0; nn < 16; nn++) {
        std::cout << weight_buffer[nn] << std::endl;
    }
    std::cout << "--------------------------------" << std::endl;
#endif //TEST_WEIGHT_1x1

#ifdef TEST_BIAS_1x1
    std::cout << "-------check load bias 1x1---------" << std::endl;
    std::cout << bias << std::endl;
    std::cout << "-----------------------------------" << std::endl;
#endif //TEST_BIAS_1x1      

    int tc, tr, tn;
    int32 partial_add = bias;
    //do not calculate padding part;
    for (tr = 1; tr < TR_MIN + 1; tr++) 
        for (tc = 1; tc < TC_MIN + 1; tc++) 
        {  
#pragma HLS PIPELINE
#pragma HLS DEPENDENCE variable=output_buffer inter false                    
            int32 partial_sum = 0;
            for (tn = 0; tn < TN; tn++) {
                partial_sum += (weight_buffer[tn] * feature_buffer_read[tn][tr][tc]) >> (WeightQ + InputQ - InterWidthQ);                    
            }
            output_buffer[tr - 1][tc - 1] = partial_add + partial_sum;
        }

    write_back_output(output_buffer, Output, r, c, TR_MIN, TC_MIN, TMP_L);
}

//*Weight0 and *Weight1 are just two port, both are point to the same address.
void DFENet_FPGA(int8 *Input, int32 *Output, int8 *Weight0, int8 *Weight1, int8 *Bias0, int8 *Bias1, uint32 TR,  
                 uint32 TC, uint32 TRow, uint32 TCol, int *WeightQ, int *BiasQ, int *InputQ, int *InterWidthQ)
{
#pragma HLS INTERFACE m_axi  depth=1*240*320  port=Input  offset=slave  bundle=DATA_IN_BUS
#pragma HLS INTERFACE m_axi  depth=10*240*320  port=Output  offset=slave  bundle=DATA_OUT_BUS
#pragma HLS INTERFACE m_axi  depth=22320  port=Weight0  offset=slave  bundle=DATA_W_BUS_0
#pragma HLS INTERFACE m_axi  depth=22320  port=Weight1  offset=slave  bundle=DATA_W_BUS_1
#pragma HLS INTERFACE m_axi  depth=170  port=Bias0  offset=slave  bundle=DATA_B_BUS_0
#pragma HLS INTERFACE m_axi  depth=170  port=Bias1  offset=slave  bundle=DATA_B_BUS_1

#pragma HLS INTERFACE s_axilite register port=return bundle=CTRL_BUS
#pragma HLS INTERFACE s_axilite register port=TR bundle=CTRL_BUS
#pragma HLS INTERFACE s_axilite register port=TC bundle=CTRL_BUS
#pragma HLS INTERFACE s_axilite register port=TRow bundle=CTRL_BUS
#pragma HLS INTERFACE s_axilite register port=TCol bundle=CTRL_BUS

#pragma HLS INTERFACE s_axilite register port=WeightQ bundle=CTRL_BUS
#pragma HLS INTERFACE s_axilite register port=BetaQ bundle=CTRL_BUS
#pragma HLS INTERFACE s_axilite register port=InputQ bundle=CTRL_BUS
#pragma HLS INTERFACE s_axilite register port=InterWidthQ bundle=CTRL_BUS

    static int32 output_buffer[Tr][Tc];
    static int8 weight_buffer0[Tm][Tn][K][K];
#pragma HLS ARRAY_PARTITION variable=weight_buffer0 complete dim=3
#pragma HLS ARRAY_PARTITION variable=weight_buffer0 complete dim=4
    static int8 weight_buffer1[Tn];
#pragma HLS ARRAY_PARTITION variable=weight_buffer1 complete dim=1    
    static int8 bias_buffer0[Tm]; 
#pragma HLS ARRAY_PARTITION variable=bias_buffer0 complete dim=1    

    static int32 feature_buffer0_0[Tn][OnChipIB_Width][OnChipIB_Height]; //write to this buffer after 3x3 conv; read this buffer to compute 1x1
#pragma HLS ARRAY_PARTITION variable=feature_buffer0_0 complete dim=1    
    static int32 feature_buffer0_1[Tn][OnChipIB_Width][OnChipIB_Height]; //add current 3x3 conv result and last 3x3 conv result to this buffer
#pragma HLS ARRAY_PARTITION variable=feature_buffer0_1 complete dim=1    
    static int32 feature_buffer1_0[Tn][OnChipIB_Width][OnChipIB_Height]; //same as 0_0
#pragma HLS ARRAY_PARTITION variable=feature_buffer1_0 complete dim=1    
    static int32 feature_buffer1_1[Tn][OnChipIB_Width][OnChipIB_Height]; //read data to compute 3x3 conv
#pragma HLS ARRAY_PARTITION variable=feature_buffer1_1 complete dim=1

    int TMP_R, TMP_C;
    int r, c;
    int TR_MIN, TC_MIN;
    bool pingpongm = false;
    bool IsReLU = 1;

	int rLoops = ceil(IMG_WIDTH / TR); 
	int cLoops = ceil(IMG_HEIGHT / TC);

    //no need to tile Tm, since all channel should be loaded;
    for (TMP_R = 0, r = 0; r < rLoops; r++, TMP_R += TR) 
    {    
        TR_MIN = MIN(TR, IMG_WIDTH - TMP_R);
        for (TMP_C = 0, c = 0; c < cLoops; c++, TMP_C += TC) 
        {
            TC_MIN = MIN(TC, IMG_HEIGHT - TMP_C);
            load_input(Input, feature_buffer1_1, TR, TC, r, c, TRow, TCol);
            
#ifdef TEST_FPGA_INPUT
            std::cout << "-------check fpga input---------" << std::endl;
            for (int ww = 0; ww < OnChipIB_Width; ww++) {
                int cont = 0;
                for (int hh = 0; hh < OnChipIB_Height; hh++) {
                    cont++;
                    std::cout << feature_buffer1_1[0][ww][hh] << '\t';
                    if (cont % 4 == 0 || hh == OnChipIB_Height - 1)
                        std::cout << std::endl;
                }
            }
            std::cout << "--------------------------------" << std::endl;
#endif //TEST_FPGA_INPUT

            // 20 conv in total; 10 for 3x3_conv, 10 for 1x1_conv;
            // for each iteration, it calculates the 3x3_conv of current layer to get feature map and 
            // calculate the 1x1_conv of last layer to get score map;
            // "+1" beacuse the first iteration can not calculate 1x1, we need one more extra iteration to cal it.
            for (int TMP_L = 0; TMP_L < layer_num + 1; TMP_L++) 
            {
                int TN;
                if (TMP_L == 0)
                    TN = 1;
                else   
                    TN = 16;

                if (pingpongm == 0) 
                {                
                    Conv3x3_sum(Weight0, Bias0, feature_buffer1_1, feature_buffer0_0, feature_buffer0_1, weight_buffer0, bias_buffer0, 
                                TMP_L, TN, TR_MIN, TC_MIN, TRow, TCol, IsReLU, InputQ[TMP_L], WeightQ[TMP_L], BiasQ[TMP_L], InterWidthQ[TMP_L]);
                    Conv1x1_write_back(Output, Weight1, Bias1, weight_buffer1, output_buffer, feature_buffer1_0, 
                                       TMP_L, TR_MIN, TC_MIN, TN, r, c, InputQ[TMP_L], WeightQ[TMP_L], BiasQ[TMP_L], InterWidthQ[TMP_L]);
                    pingpongm = 1;
                }
                else 
                {
                    Conv3x3_sum(Weight0, Bias0, feature_buffer0_1, feature_buffer1_0, feature_buffer1_1, weight_buffer0, bias_buffer0, 
                                TMP_L, TN, TR_MIN, TC_MIN, TRow, TCol, IsReLU, InputQ[TMP_L], WeightQ[TMP_L], BiasQ[TMP_L], InterWidthQ[TMP_L]);
                    Conv1x1_write_back(Output, Weight1, Bias1, weight_buffer1, output_buffer, feature_buffer0_0, 
                                       TMP_L, TR_MIN, TC_MIN, TN, r, c, InputQ[TMP_L], WeightQ[TMP_L], BiasQ[TMP_L], InterWidthQ[TMP_L]);
                    pingpongm = 0;                    
                }
            }
        }
    }    
}


#include "netConfig.h"
#include "../kernel/DFENet.h"
#include <iostream>
#include <time.h>

void load_input_maxQ(const char* input_maxQ_name, int *inputQ_buf)
{
    FILE *fp_w = fopen(input_maxQ_name, "rb");
    if(!fp_w) 
        std::cout << "error load input_maxQ.bin" << std::endl;
    fread(inputQ_buf, sizeof(int), LAYER_NUM + 1, fp_w);
    fclose(fp_w);
}

void load_weight_maxQ(const char* weight_maxQ_name, int *weightQ_buf)
{
    FILE *fp_w = fopen(weight_maxQ_name, "rb");
    if(!fp_w) 
        std::cout << "error load weight_maxQ.bin" << std::endl;
    fread(weightQ_buf, sizeof(int), LAYER_NUM + 1, fp_w);
    fclose(fp_w);
}

void load_bias_maxQ(const char* bias_maxQ_name, int *biasQ_buf)
{
    FILE *fp_w = fopen(bias_maxQ_name, "rb");
    if(!fp_w) 
        std::cout << "error load bias_maxQ.bin" << std::endl;
    fread(biasQ_buf, sizeof(int), LAYER_NUM + 1, fp_w);
    fclose(fp_w);
}

void load_InterWidth_maxQ(const char* InterWidth_maxQ_name, int *InterWidthQ_buf)
{
    FILE *fp_w = fopen(InterWidth_maxQ_name, "rb");
    if(!fp_w) 
        std::cout << "error load interwidth_maxQ.bin" << std::endl;
    fread(InterWidthQ_buf, sizeof(int), LAYER_NUM + 1, fp_w);
    fclose(fp_w);
}

//note that input data should follow the order of PL side
void load_input(const char* input_name, int8_t *input_buf)
{
    FILE *fp_w = fopen(input_name, "rb");
    if(!fp_w) 
        std::cout << "error load input.bin" << std::endl;
    fread(input_buf, sizeof(int8_t), INPUT_MEM_LEN, fp_w);
    fclose(fp_w);
}

void load_weight(const char* weight_name, int8_t *weight_buf) 
{
    FILE *fp_w = fopen(weight_name, "rb");
    if(!fp_w) {
        std::cout << "error load weights.bin" << std::endl;
        exit(-1);
    }
    fread(weight_buf, sizeof(int8_t), MAX_WEIGHT_LEN, fp_w);
    fclose(fp_w);
}

void load_bias(const char* bias_name, int8_t *beta_buf)
{
    FILE *fp_w = fopen(bias_name, "rb");
    if(!fp_w) {
        std::cout << "error load bias.bin" << std::endl;
        exit(-1);
    }
    fread(beta_buf, sizeof(int8_t), MAX_BIAS_LEN, fp_w);
    fclose(fp_w);
}

void ps_gfeNet(float *input, int32_t *output_buf) 
{
    int8_t *weight_fixed_buf = (int8_t *)calloc(MAX_WEIGHT_LEN, sizeof(int8_t));
    int8_t *beta_fixed_buf   = (int8_t *)calloc(MAX_BIAS_LEN, sizeof(int8_t));
    int8_t *input_fixed_buf = (int8_t *)calloc(INPUT_MEM_LEN, sizeof(int8_t));
    int *inputQ_buf = (int *)calloc(LAYER_NUM + 1, sizeof(int));
    int *weightQ_buf = (int *)calloc(LAYER_NUM + 1, sizeof(int));
    int *biasQ_buf = (int *)calloc(LAYER_NUM + 1, sizeof(int));
    int *InterWidthQ_buf = (int *)calloc(LAYER_NUM + 1, sizeof(int));
    
    const char *input_name = "Input_ap8.bin";    
    const char *weight_name = "Weight_ap8.bin";
    const char *bias_name = "Bias_ap8.bin";
    const char *input_maxQ_name = "Input_maxQ.bin";
    const char *weight_maxQ_name = "Weight_maxQ.bin";
    const char *bias_maxQ_name = "Bias_maxQ.bin";
    const char *InterWidth_maxQ_name = "InterWidth_maxQ.bin";

    load_input(input_name, input_fixed_buf);

#ifdef TEST_LOAD_INPUT
    std::cout << "-------check load input---------" << std::endl;
    const char * input_data_ps_file = "input_data_ps.txt";
    write_int8_to_txt(input_data_ps_file, input_fixed_buf, 240 * 320);    
#endif //TEST_LOAD_INPUT     

    load_weight(weight_name, weight_fixed_buf);

#ifdef TEST_LOAD_WEIGHT
    std::cout << "-------check load weight---------" << std::endl;
    const char * weight_data_ps_file = "weight_data_ps.txt";
    write_int8_to_txt(weight_data_ps_file, weight_fixed_buf, MAX_WEIGHT_LEN); 
#endif //TEST_LOAD_WEIGHT    

    load_bias(bias_name, beta_fixed_buf);

#ifdef TEST_LOAD_BIAS
    std::cout << "-------check load bias---------" << std::endl;
    const char * bias_data_ps_file = "bias_data_ps.txt";
    write_int8_to_txt(bias_data_ps_file, beta_fixed_buf, MAX_BIAS_LEN); 
#endif //TEST_LOAD_BIAS

    load_input_maxQ(input_maxQ_name, inputQ_buf);

    load_weight_maxQ(weight_maxQ_name, weightQ_buf);

    load_bias_maxQ(bias_maxQ_name, biasQ_buf);

    load_InterWidth_maxQ(InterWidth_maxQ_name, InterWidthQ_buf);

    int TR = MIN(((OnChipIB_Width - K) / S + 1), Tr);
    int TC = MIN(((OnChipIB_Height - K) / S + 1), Tc);
	int TRow = (TR - 1) * S + K;
    int TCol = (TC - 1) * S + K;

    time_t first, second;
    first = time(NULL);

    // launch PL kernel
    DFENet_FPGA(input_fixed_buf, output_buf, weight_fixed_buf, weight_fixed_buf, beta_fixed_buf, beta_fixed_buf, 
                TR, TC, TRow, TCol, inputQ_buf, weightQ_buf, biasQ_buf, InterWidthQ_buf);        

    second = time(NULL);
    printf("Predicted in %f seconds.\n", difftime(second, first));

    free(weight_fixed_buf);
    free(beta_fixed_buf);
    free(input_fixed_buf);
    free(inputQ_buf);
    free(weightQ_buf);
    free(biasQ_buf);  
    free(InterWidthQ_buf);          
}
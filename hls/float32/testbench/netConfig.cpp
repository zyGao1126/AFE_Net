#include "netConfig.h"
#include "../kernel/DFENet.h"
#include <iostream>
#include <time.h>

void load_weight(const char* weight_name, float *weight_buf) 
{
    FILE *fp_w = fopen(weight_name, "rb");
    if(!fp_w) {
        std::cout << "error load weights.bin" << std::endl;
        exit(-1);
    }
    fread(weight_buf, sizeof(float), MAX_WEIGHT_LEN, fp_w);
    fclose(fp_w);
}

void load_bias(const char* bias_name, float *beta_buf)
{
    FILE *fp_w = fopen(bias_name, "rb");
    if(!fp_w) {
        std::cout << "error load bias.bin" << std::endl;
        exit(-1);
    }
    fread(beta_buf, sizeof(float), MAX_BIAS_LEN, fp_w);
    fclose(fp_w);
}

void ps_gfeNet(float *input, float *output_buf) 
{
    float *weight_buf = (float *)calloc(MAX_WEIGHT_LEN, sizeof(float));
    float *beta_buf   = (float *)calloc(MAX_BIAS_LEN, sizeof(float));
    float *input_buf = (float *)calloc(INPUT_MEM_LEN + SLACK, sizeof(float));
    
    const char *weight_name = "./weight/weight_data.bin";
    const char *bias_name = "./weight/bias_data.bin";
    float *in_ptr = input_buf + SLACK / 2;
    memcpy(in_ptr, input, INPUT_WIDTH * INPUT_HEIGHT * sizeof(float));

#ifdef TEST_LOAD_INPUT
    std::cout << "-------check load input---------" << std::endl;
    const char * input_data_ps_file = "input_data_ps.txt";
    write_to_txt(input_data_ps_file, in_ptr, 240 * 320);    
#endif //TEST_LOAD_INPUT     

    load_weight(weight_name, weight_buf);

#ifdef TEST_LOAD_WEIGHT
    std::cout << "-------check load weight---------" << std::endl;
    const char * weight_data_ps_file = "weight_data_ps.txt";
    write_to_txt(weight_data_ps_file, weight_buf, MAX_WEIGHT_LEN); 
#endif //TEST_LOAD_WEIGHT    

    load_bias(bias_name, beta_buf);

#ifdef TEST_LOAD_BIAS
    std::cout << "-------check load bias---------" << std::endl;
    const char * bias_data_ps_file = "bias_data_ps.txt";
    write_to_txt(bias_data_ps_file, beta_buf, MAX_BIAS_LEN); 
#endif //TEST_LOAD_BIAS

    int TR = MIN(((OnChipIB_Width - K) / S + 1), Tr);
    int TC = MIN(((OnChipIB_Height - K) / S + 1), Tc);
	int TRow = (TR - 1) * S + K;
    int TCol = (TC - 1) * S + K;

    time_t first, second;
    first = time(NULL);

    // launch PL kernel
    DFENet_FPGA(in_ptr, output_buf, weight_buf, weight_buf, beta_buf, beta_buf, TR, TC, TRow, TCol);

    second = time(NULL);
    printf("Predicted in %f seconds.\n", difftime(second, first));

    free(weight_buf);
    free(beta_buf);
    free(input_buf);
}
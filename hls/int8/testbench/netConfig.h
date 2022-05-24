#ifndef NETCONFIG
#define NETCONFIG

#define INPUT_HEIGHT 320
#define INPUT_WIDTH 240
//max_weight_len = 1x16x3x3 + 9x(16x16x3x3) + 10x(16x1x1x1)
#define MAX_WEIGHT_LEN 21040
//max_bias_len = 16x10 + 1x10
#define MAX_BIAS_LEN 170 
//corresponding to layer_num in rfnet_fpga_v1.cpp 
#define LAYER_NUM 10
#define INPUT_MEM_LEN 1 * INPUT_WIDTH * INPUT_HEIGHT
//here input_width is equal as output_width, so can write like this
#define OUTPUT_MEM_LEN 10 * INPUT_WIDTH * INPUT_HEIGHT
// set aside some memory slack
#define SLACK 240*2

void ps_gfeNet(float *input, int32_t *output_buf);
void write_to_txt(const char* name, float *input, int len);
void write_int8_to_txt(const char* name, int8_t *input, int len);

#endif

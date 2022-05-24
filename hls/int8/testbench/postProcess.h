#ifndef POSTPROCESS
#define POSTPROCESS

#include "preProcess.h"

typedef struct s_inferConfig
{
public:
    static const int pooling_ksize = 15; 
    static const int pooling_stride = 1;
    static const int com_strength = 5; //5
    static const int nms_strength = 100;
    static const int conv_ksize = 15;
    static const int radius = 8;
    static const int thresh = 0;
    static const int nms_ksize = 5;
    static const int TOPK = 512;
    static const int gaussian_ksize = 15;
    static const int gaussian_sigma = 0.5;
    static const int P_size = 32;
    constexpr static const int scale_list[] = {3,5,7,9,11,13,15,17,19,21};
} inferConfig;

float **define_arr_2D(const int height, const int width);
void post_process(int32_t* const ini_score_maps, float** const patch_buf, Image im); 

#endif
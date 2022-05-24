#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include "preProcess.h"
#include "postProcess.h"
#include "netConfig.h"

int main(int argc, char *argv[])
{
    printf("------ BEGIN GFE_NET TEST -------\n");
    ///////////////////////////////// load image /////////////////////////////////
    char img_buff[256];
    char *input_img = img_buff;
	if(argc == 1)
		strncpy(input_img, "./test.jpg", 256);
	else
		strncpy(input_img, argv[1], 256);
    
    Image im = load_image_stb_gray(input_img, 3); //3 channel img
    printf("Input img: %s\n w = %d, h = %d, c = %d\n", input_img, im.m_w, im.m_h, im.m_c);
    // Image im_norm = image_std(im);
    Image im_norm = image_norm(im);

#ifdef TEST_DATA_NORM
    std::cout << "-------test norm data---------" << std::endl;
    int pos, neg = 0;
    for (int j = 0; j < im_norm.m_w * im_norm.m_h * im_norm.m_c; ++j)
    {
        if ((float)im_norm.m_data[j] > 1)        
            pos++;
        else if ((float)im_norm.m_data[j] < -1)  
            neg++;
        std::cout << "data: " << (float)im_norm.m_data[j] << '\t';
        if (j % 4 == 0)
            std::cout << std::endl; 
    }
    std::cout << "pos: " << pos << "\t" << "neg: " << neg << std::endl;
#endif //TEST_DATA_NORM

    int w_resized = 240;
    int h_resized = 320;
    Image im_resized = image_resize(im_norm, w_resized, h_resized); //[240,320,1]
    float *InputPixel = im_resized.m_data;

#ifdef TEST_DATA_RESIZE
    std::cout << "-------check resized data---------" << std::endl;
    const char * resized_data_file = "resized_data.txt";
    write_to_txt(resized_data_file, InputPixel, 240 * 320);
    int count = 0;
    for (int i = 0; i < w_resized * h_resized; i++) {
        if (im_resized.m_data[i] > 1 || im_resized.m_data[i] < -1)
            count++;
    }
    std::cout << "count: " << count << std::endl;
    std::cout << "-------finish check resized data---------" << std::endl;
#endif //TEST_DATA_RESIZE
  
    //////////////////////////// launch GFENet kernel ////////////////////////////
    float *output_buf = (float *)calloc(OUTPUT_MEM_LEN, sizeof(float));
    ps_gfeNet(InputPixel, output_buf);
    printf("FINISH DETECTOR!\n");

#ifdef TEST_OUTPUT_BUF
    std::cout << "-------test output buf---------" << std::endl;
    const char * output_buf_file = "./output_buf.txt";
    write_to_txt(output_buf_file, output_buf, 10 * 240 * 320);
    std::cout << "-------finish test output buf--------" << std::endl;
    std::cout << std::endl;
#endif //TEST_OUTPUT_BUF
    
    ///////////////////////////////// post process //////////////////////////////////
    float **patch_buf = define_arr_2D(inferConfig::TOPK, inferConfig::P_size * inferConfig::P_size);
    post_process(output_buf, patch_buf, im);
    printf("FINISH POST PROCESS!\n");
    
    free_image(im);
    free_image(im_norm);
    free_image(im_resized);

    printf("------ FINISH GFE_NET TEST -------\n");

    return 0;

}
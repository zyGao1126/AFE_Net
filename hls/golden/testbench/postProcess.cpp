#include "postProcess.h"
#include "netConfig.h"

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include <vector>
#include <unordered_map>
#include <queue>

//return value inserted a specific interval from the begining value;
//such as [1,0.5,0,...,-1, 1,0.5,...,-1, ..., 1,0.5,...,-1]
//num means how many number between [-1,1] includes bound; count means the number value to return from the begining of 0; 
float linspace(float begin_val, float end_val, int num, int count)
{
    float interval = (end_val - begin_val) / (num - 1);
    //cout << "interval: " << interval << endl;
    float result = begin_val + (count % num) * interval;
    //cout << "result: " << result << endl;
    return result;
}

//return value such as [1,1,...1, 0.5,0.5,...0.5, 0,0,...,0, ..., -1,-1,...,-1]
//overlap_time means how many times the same number repeated, num and count are same as function linspace.
float overlapspace(float begin_val, float end_val, int overlap_time, int num, int count)
{
    float interval = (end_val - begin_val) / (num - 1);
    float result = begin_val + floor(count / overlap_time) * interval;
    return result;
}

void write_vec_to_txt(const char *name, std::vector<std::pair<int, int>> vec)
{
    std::ofstream fout(name);
    for (auto iter = vec.begin(); iter != vec.end(); ++iter)
        fout << iter->first << "\t" << iter->second << std::endl;
}

//this function defines dymanic 3D array and return the pointer
float *** define_arr_3D(int channel, int height, int width)
{
    float ***ptr = new float **[channel]; 
    for (int i = 0; i < channel; i++) {
        ptr[i] = new float *[height];
        for (int j = 0; j < height; j++)
            ptr[i][j] = new float [width];
    }
    return ptr;
}

//this function defines dymanic 2D array and return the pointer
float **define_arr_2D(const int height, const int width)
{
    float **ptr = new float *[height]; 
    for (int i = 0; i < height; i++) {
        ptr[i] = new float [width];
    }
    return ptr;    
}

void maxpool(float ***output, float *input, const int height, const int width)
{
    int tr, tc, tm;
    int i, j, x, y;
    int ksize = inferConfig::pooling_ksize;
    int padding = ksize / 2; //7
    int offset, offset1, offset2;
    //add padding
    for (tm = 0; tm < LAYER_NUM; tm++) {
        for (y = 0; y < padding; y++) 
            for (x = 0; x < INPUT_WIDTH; x++) {
                offset1 = (tm * INPUT_HEIGHT * INPUT_WIDTH) + (y * INPUT_WIDTH) + x;
                output[tm][y][x] = exp(inferConfig::com_strength * (input[offset1] - 10));
                offset2 = tm * INPUT_HEIGHT * INPUT_WIDTH + (y + (INPUT_HEIGHT - padding)) * INPUT_WIDTH + x;      
                output[tm][y + (INPUT_HEIGHT - padding)][x] = exp(inferConfig::com_strength * (input[offset2] - 10));
            }

        for (y = 0; y < INPUT_HEIGHT; y++) 
            for (x = 0; x < padding; x++) {
                offset1 = tm * INPUT_HEIGHT * INPUT_WIDTH + y * INPUT_WIDTH + x;
                output[tm][y][x] = exp(inferConfig::com_strength * (input[offset1] - 10));
                offset2 = tm * INPUT_HEIGHT * INPUT_WIDTH + y * INPUT_WIDTH + x + (INPUT_WIDTH - padding);
                output[tm][y][x + (INPUT_WIDTH - padding)] = exp(inferConfig::com_strength * (input[offset2] - 10));             
            }
    }

    //begin max pooling
    for (tc = 0; tc < INPUT_HEIGHT - ksize + 1; tc++)
        for (tr = 0; tr < INPUT_WIDTH - ksize + 1; tr++) {
            float tmp = -DBL_MAX;
            //find max value for all ksize range and layer num;
            for (tm = 0; tm < LAYER_NUM; tm++) {
                offset = tm * INPUT_HEIGHT * INPUT_WIDTH + tc * INPUT_WIDTH + tr;
                //i: col; j: row; 
                for (i = 0; i < ksize; i++)
                    for (j = 0; j < ksize; j++) {                              
                        if (input[offset + i * INPUT_WIDTH + j] > tmp)
                            tmp = input[offset + i * INPUT_WIDTH + j];
                    }                
            }

            //calculate exp_maps using input and tmp value;
            for (tm = 0; tm < LAYER_NUM; tm++) {   
                offset = tm * INPUT_HEIGHT * INPUT_WIDTH + tc * INPUT_WIDTH + tr;
                float value = inferConfig::com_strength * (input[offset] - tmp);
                output[tm][tc + padding][tr + padding] = exp(value);
            }
        }

#ifdef TEST_MAX_POOLING
    std::cout << "-------check Max Pooling result---------" << std::endl;
    int count = 0;
    for (int m = 0; m < LAYER_NUM; m++)
        for (i = 0; i < INPUT_HEIGHT; i++)
            for (j = 0; j < INPUT_WIDTH; j++) {
                count++;
                std::cout << output[m][i][j] << '\t'; 
                if (count % 4 == 0 && count != 0)
                    std::cout << std::endl;
            }
    std::cout << "------------------------------------" << std::endl;
#endif //TEST_MAX_POOLING
    std::cout << "-----Finish Max Pooling-----" << std::endl;
}

//calculate convlution and get probilities.
void conv_3D(float** const output, float*** const input)
{
    int tr, tc, tm;
    int kx, ky;
    int x, y;
    int padding = inferConfig::conv_ksize / 2;

    //add padding
    for (y = 0; y < padding; y++)
        for (x = 0; x < INPUT_WIDTH; x++) {
            output[y][x] = 0;
            output[y + (INPUT_HEIGHT - padding)][x] = 0;
        }
    for (y = 0; y < INPUT_HEIGHT; y++)
        for (x = 0; x < padding; x++) {
            output[y][x] = 0;
            output[y][x+(INPUT_WIDTH-padding)] = 0;                
        }

    //begin calculate
    for (tc = 0; tc < INPUT_HEIGHT - inferConfig::conv_ksize + 1; tc++)
        for (tr = 0; tr < INPUT_WIDTH - inferConfig::conv_ksize + 1; tr++) {
            float sum = 0;            
            for (tm = 0; tm < LAYER_NUM; tm++)
                for (kx = 0; kx < inferConfig::conv_ksize; kx++)
                    for (ky = 0; ky < inferConfig::conv_ksize; ky++)
                        // here weight is constant 1
                        sum += (input[tm][tc + kx][tr + ky] * 1);
            output[tc+padding][tr+padding] = sum;
        }
    
#ifdef TEST_CONV_3D
    std::cout << "-------check Conv 3D result---------" << std::endl;
    int count = 0;
    for (int i = 0; i < INPUT_HEIGHT; i++)
        for (int j = 0; j < INPUT_WIDTH; j++) {
            count++;
            std::cout << output[i][j] << '\t'; 
            if (count % 4 == 0 && count != 0)
                std::cout << std::endl;
        }
    std::cout << "------------------------------------" << std::endl;
#endif

    std::cout << "-----Finish Conv 3D-----" << std::endl;
}

//here conv_result become max_output after this function; pooling_result become kp_prob after this function;
void get_probs(float*** const pooling_result, float** const conv_result)
{
    int tm, tr, tc;
    //int index = 1e-8;
    int index = 1;
    float output;
    for (tc = 0; tc < INPUT_HEIGHT; tc++)
        for (tr = 0; tr < INPUT_WIDTH; tr++) {
            float tmp = -DBL_MAX;
            for (tm = 0; tm < LAYER_NUM; tm++) {
                output = pooling_result[tm][tc][tr] / (conv_result[tc][tr] + index);
                pooling_result[tm][tc][tr] = output;
                if (output > tmp)
                    tmp = output;
            }
            conv_result[tc][tr] = tmp;
        }

#ifdef TEST_PROB
    std::cout << "-------check probility result---------" << std::endl;
    int count = 0;
    for (int m = 0; m < layer_num; m++)
        for (int i = 0; i < INPUT_HEIGHT; i++)
            for (int j = 0; j < INPUT_WIDTH; j++) {
                count++;
                std::cout << pooling_result[m][i][j] << '\t'; 
                if (count % 4 == 0 && count != 0)
                    std::cout << std::endl;
            }
    std::cout << "------------------------------------" << std::endl;
#endif //TEST_PROB

    std::cout << "-----Finish Get Probability-----" << std::endl;
}

//max_prob becomes final_score_map after soft_nms_1d;
//output final_score_map corresponds to [im1w_score] in def process;
void soft_nms_1d(float** const final_scale_map, float** const max_prob, float*** const input)
{
    int tm, tr, tc;
    float value;
    float softmax;
    //define this array to store intermediate results;
    float tmp_arr[LAYER_NUM];
    
    for (tc = inferConfig::radius; tc < INPUT_HEIGHT - inferConfig::radius + 1; tc++)
        for (tr = inferConfig::radius; tr < INPUT_WIDTH - inferConfig::radius + 1; tr++) {
            float tmp_sum = 0;
            for (tm = 0; tm < LAYER_NUM; tm++) {
                value = exp(inferConfig::nms_strength * (input[tm][tc][tr] - max_prob[tc][tr]));
                //sum according to the last dimension
                tmp_sum = value + tmp_sum;
                tmp_arr[tm] = value; 
            }
            float tmp_score_map = 0;
            float tmp_scale_map = 0;
            for (tm = 0; tm < LAYER_NUM; tm++) {
                softmax = tmp_arr[tm] / (tmp_sum + 1e-8);
                tmp_score_map += (softmax * input[tm][tc][tr]);
                // tmp_scale_map += (softmax * inferConfig::scale_list[tm]);
                tmp_scale_map = tmp_scale_map + softmax;
            }
            max_prob[tc][tr] = tmp_score_map;
            final_scale_map[tc][tr] = tmp_scale_map;
        }

    int x, y;
    //filter border
    for (y = 0; y < INPUT_HEIGHT; y++)
        for (x = 0; x < inferConfig::radius; x++) {
            max_prob[y][x] = 0;
            max_prob[y][x + INPUT_WIDTH - inferConfig::radius] = 0;                
        }

    for (y = 0; y < inferConfig::radius; y++)
        for (x = 0; x < INPUT_WIDTH; x++) {
            max_prob[y][x] = 0;
            max_prob[y + INPUT_HEIGHT - inferConfig::radius][x] = 0;                                     
        }

#ifdef TEST_NMS_1D
    std::cout << "-------check soft nms 1d result---------" << std::endl;
    int count = 0;
    /*
    for (int i = 0; i < INPUT_HEIGHT; i++)
        for (int j = 0; j < INPUT_WIDTH; j++) 
        {
            count++;
            cout << max_prob[i][j] << '\t'; 
            if (count % 4 == 0 && count != 0)
                cout << endl;
        }
    */
    for (int i = 0; i < INPUT_HEIGHT; i++)
        for (int j = 0; j < INPUT_WIDTH; j++) {
            count++;
            std::cout << final_scale_map[i][j] << '\t'; 
            if (count % 4 == 0 && count != 0)
                std::cout << std::endl;
        }
    
    std::cout << "----------------------------------------" << std::endl;
#endif

    std::cout << "-----Finish Soft NMS 1d-----" << std::endl;
}

void nms(float** const final_score_map)
{
    int tr, tc;
    int x, y;
    int nms_height = INPUT_HEIGHT + (inferConfig::nms_ksize - 1);
    int nms_width = INPUT_WIDTH + (inferConfig::nms_ksize - 1);
    int nms_padding = inferConfig::nms_ksize / 2; //2
    float tmp;
    float **nms_slice_map = define_arr_2D(nms_height, nms_width);

    //add padding;
    for (y = 0; y < nms_padding; y++)
        for (x = 0; x < nms_width; x++) {
            nms_slice_map[y][x] = 0;
            nms_slice_map[y + nms_height - nms_padding][x] = 0;
        }
    for (y = 0; y < nms_height; y++)
        for (x = 0; x < nms_padding; x++) {
            nms_slice_map[y][x] = 0;
            nms_slice_map[y][x + nms_width - nms_padding] = 0;
        }     

    // assign value to nms_slice_map;
    for (tc = 0; tc < nms_height - 2 * nms_padding; tc++)
        for (tr = 0; tr < nms_width - 2 * nms_padding; tr++) {
            //nms torch.where(input < thresh);
            final_score_map[tc][tr] = (final_score_map[tc][tr] < 0 ? 0 : final_score_map[tc][tr]);             
            nms_slice_map[tc + nms_padding][tr + nms_padding] = final_score_map[tc][tr];
        }

    // max pooling (ksize = 5)
    // here bound is INPUT_HEIGHT and INPUT_WIDTH since pooling can not load pixels beyond the range of INPUT_HEIGHT;
    for (tc = 0; tc < INPUT_HEIGHT; tc++)
        for (tr = 0; tr < INPUT_WIDTH; tr++) {
            tmp = DBL_MIN;
            for (int i = 0; i < inferConfig::nms_ksize; i++)
                for (int j = 0; j < inferConfig::nms_ksize; j++) 
                    tmp = (nms_slice_map[tc + i][tr + j] > tmp ? nms_slice_map[tc + i][tr + j] : tmp);

            //torch.ge(center_map, max_slice)
            //if final_score_map > tmp, do no operation, or mask = 0 which means final_score_map = 0;
            final_score_map[tc][tr] = (final_score_map[tc][tr] > tmp ? final_score_map[tc][tr] : 0);         
        }

    free(nms_slice_map);

#ifdef TEST_NMS
    std::cout << "-------check nms result---------" << std::endl;
    int count = 0;
    for (int i = 0; i < INPUT_HEIGHT; i++)
        for (int j = 0; j < INPUT_WIDTH; j++) 
        {
            count++;
            std::cout << final_score_map[i][j] << '\t'; 
            if (count % 4 == 0 && count != 0)
                std::cout << std::endl;
        }
    std::cout << "----------------------------------------" << std::endl;
#endif
    std::cout << "-----Finish NMS-----" << std::endl;
}

void choose_topK_map(std::vector<std::pair<int, int>>& topk_score_vec, float** const final_score_map, bool mask_flag) 
{
    //this vector stores the TOPK pixels location;
    std::vector<std::pair<int, int>> tmp_vec;
    //store the values and locations of TOPK pixels;
    std::unordered_map<float, std::pair<int, int>> tmp_map;
    std::priority_queue<float, std::vector<float>, std::greater<float>> small_heap;

    for(int tc = 0; tc < INPUT_HEIGHT; tc++)
        for (int tr = 0; tr < INPUT_WIDTH; tr++) {
            if (final_score_map[tc][tr] == 0)
                continue;
            else {
                // heap is not full
                if (tmp_map.size() < inferConfig::TOPK) {
                    small_heap.push(final_score_map[tc][tr]);
                    tmp_map.insert({final_score_map[tc][tr], std::make_pair(tr, tc)});
                }
                else {
                    float tmp_value = small_heap.top();
                    if (final_score_map[tc][tr] > tmp_value) {
                        small_heap.pop();
                        small_heap.push(final_score_map[tc][tr]);
                        // unordered_map cost O(1) time to erase and insert
                        tmp_map.erase(tmp_value);
                        tmp_map.insert({final_score_map[tc][tr], std::make_pair(tr,tc)});
                    }
                }
            }
        }

    if (mask_flag == true) {
        //write mask, first initilize final_score_map as 0
        for (int i = 0; i < INPUT_HEIGHT; i++)
            for (int j = 0; j < INPUT_WIDTH; j++)
                final_score_map[i][j] = 0;
    }      
    for (auto iter = tmp_map.begin(); iter != tmp_map.end(); ++iter) {
            topk_score_vec.push_back(iter->second);
        if (mask_flag == true)
            final_score_map[(iter->second).second][(iter->second).first] = iter->first;
    }
         
#ifdef TEST_TOPK_VECTOR
    std::cout << "-------check topk vector result---------" << std::endl;
    int count = 0;
    std::cout << "check keypoints location" << std::endl;
    for (auto iter = tmp_vec.begin(); iter != tmp_vec.end(); ++iter) {
        std::cout << "number: " << count << '\t' << "location: " << iter->first << " " << iter->second << std::endl;
        count++;
    }
    if (mask_flag == true) {
        std::cout << "check topk mask" << std::endl;
        count = 0;
        for (int i = 0; i < INPUT_HEIGHT; i++)
            for (int j = 0; j < INPUT_WIDTH; j++) {
                count++;
                std::cout << final_score_map[i][j] << '\t'; 
                if (count % 4 == 0 && count != 0)
                    std::cout << std::endl;
            }   
    }
    else
        std::cout << "nothing to test" << std::endl;
    
    std::cout << "----------------------------------------" << std::endl;
#endif //TEST_TOPK_VECTOR

    std::cout << "-----Finish Choose TopK Map-----" << std::endl;  
}


void gaussian_filter(float **topk_score_map)
{
    float **gaussian_kernel = define_arr_2D(inferConfig::gaussian_ksize, inferConfig::gaussian_ksize);
    int mu_x = inferConfig::gaussian_ksize / 2;
    for (int x = 0; x < inferConfig::gaussian_ksize; x++)
        for (int y = 0; y < inferConfig::gaussian_ksize; y++)
            // formula of calculate gaussian_kernel
            gaussian_kernel[x][y] = exp((-(y - mu_x) ^ 2) / (2 * (inferConfig::gaussian_sigma ^ 2)) + ((x - mu_x) ^ 2) / (2 * (inferConfig::gaussian_sigma ^ 2))); 

    int padding = inferConfig::gaussian_ksize / 2; //7
    //begin conv;
    for (int tc = padding; tc < INPUT_HEIGHT - padding; tc++)
        for (int tr = padding; tr < INPUT_WIDTH - padding; tr++) {
            float sum = 0;                       
            for (int kx = -padding; kx < inferConfig::gaussian_ksize-padding; kx++)
                for (int ky = -padding; ky < inferConfig::gaussian_ksize-padding; ky++)
                    sum = sum + (topk_score_map[tc][tr] * gaussian_kernel[kx+padding][ky+padding]);
            topk_score_map[tc][tr] = sum;

            //apply clamp, this will not happen in normal case;
            topk_score_map[tc + padding][tr + padding] = (topk_score_map[tc + padding][tr + padding] > 1 ? 1 : topk_score_map[tc + padding][tr + padding]);
            topk_score_map[tc + padding][tr + padding] = (topk_score_map[tc + padding][tr + padding] < 0 ? 0 : topk_score_map[tc + padding][tr + padding]);
        }        

    int x, y;
    //add padding;
    for (y = 0; y < padding; y++)
        for (x = 0; x < INPUT_WIDTH; x++) {
            topk_score_map[y][x] = 0;
            topk_score_map[y + (INPUT_HEIGHT - padding)][x] = 0;
        }
    for (y = 0; y < INPUT_HEIGHT; y++)
        for (x = 0; x < padding; x++) {
            topk_score_map[y][x] = 0;
            topk_score_map[y][x + (INPUT_WIDTH - padding)] = 0;                
        }
    free(gaussian_kernel);

#ifdef TEST_CONV_2D
    std::cout << "-------check conv 2D result---------" << td::endl;
    int count_test = 0;
    for (int i = 0; i < INPUT_HEIGHT; i++)
        for (int j = 0; j < INPUT_WIDTH; j++) 
        {
            count_test++;
            std::cout << topk_score_map[i][j] << '\t'; 
            if (count_test % 4 == 0 && count_test != 0)
                std::cout << std::endl;
        }
    std::cout << "----------------------------------------" << std::endl;
#endif //TEST_CONV_2D  
    std::cout << "-----Finish Gaussian Filter-----" << std::endl;
}


template <typename T>
void deallocate_3D(T ***ptr) {
    free(ptr);
    ptr = nullptr;
}

template <typename T>
void deallocate_2D(T **ptr) {
    free(ptr);
    ptr = nullptr;
}

void soft_nms(float* const scale_logits, float** const patch_buf, const Image& im)
{
    // initialize configuration parameters
    int col_len = INPUT_HEIGHT, row_len = INPUT_WIDTH, ch_len = LAYER_NUM;
    bool mask_flag = true;
    std::vector<std::pair<int, int>> topk_score_vec;

    // initialize buffer space
    float ***buf_3d = define_arr_3D(ch_len, col_len, row_len);
    float **buf_2d = define_arr_2D(col_len, row_len);
    float **final_scale_map = define_arr_2D(col_len, row_len);

    // write pixels from scale_logits to buf_3d 
    maxpool(buf_3d, scale_logits, col_len, row_len);

    // do conv operation and write value from buf_3d into buf_2d
    conv_3D(buf_2d, buf_3d);

    // get probability in layer dimension
    get_probs(buf_3d, buf_2d);

    // get local maximum value and local scale map 
    soft_nms_1d(final_scale_map, buf_2d, buf_3d);

    // free space
    deallocate_3D(buf_3d);

    // non-maximum suppression 
    nms(buf_2d);

    // use gaussian filter to smooth the data
    gaussian_filter(buf_2d);
    
    // select topK keypoints
    choose_topK_map(topk_score_vec, buf_2d, mask_flag);

    // free space
    deallocate_2D(final_scale_map);    
    deallocate_2D(buf_2d);

#ifdef WRITE_KEYPOINTS
    const char *kps_vec = "./kps_vec.txt";
    write_vec_to_txt(kps_vec, topk_score_vec);
    cout << "Finish Write kps vector to txt" << endl;
#endif //WRITE_KEYPOINTS

}

//this function finally obtains the patch of each keypoint using various of mathematical transformations
//ini_score_maps: [1,240,320,10]
void post_process(float* const ini_score_maps, float** const patch_buf, Image im) 
{
    time_t first, second;
    first = time(NULL);  
    soft_nms(ini_score_maps, patch_buf, im);
    second = time(NULL);
    printf("infer in %f seconds.\n", difftime(second, first));
}
#include "preProcess.h"
#include "stb_image.h"
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <fstream>
#include <iostream>

// in debug mode, write intermediate result into .txt
void write_to_txt(const char* name, float *input, int len)
{
    std::ofstream file(name);
    if (!file) {
        std::cout << "Unable to open otfile";
        exit(1);
    }
    int count = 0;
    for (int i = 0; i < len; i++)
    {
        count++;
        file << input[i] << "\t";
        if (count % 4 == 0)
            file << std::endl;
    }
    file.close();
    std::cout << "finish write " << name << " to txt." << std::endl;
}

void write_int8_to_txt(const char* name, int8_t *input, int len)
{
    std::ofstream file(name);
    if (!file) {
        std::cout << "Unable to open otfile";
        exit(1);
    }
    int count = 0;
    for (int i = 0; i < len; i++) {
        count++;
        file << input[i] << "\t";
        if (count % 4 == 0 )
            file << std::endl;
    }
    file.close();
    std::cout << "finish write " << name << " to txt." << std::endl;
}

// initilize an empty image
Image make_empty_image(const int w, const int h, const int c)
{
    Image out(w, h, c, nullptr);
    return out;
}

//construct a new image struct
Image make_image(const int w, const int h, const int c)
{
    Image out = make_empty_image(w, h, c);
    out.m_data = (float *)calloc(h * w * c, sizeof(float));
    return out;
}

Image load_image_stb_gray(const char *filename, int channels)
{
    int w, h, c;
    unsigned char *data = stbi_load(filename, &w, &h, &c, channels); //RGBRGB... order 
    if (!data) {
        fprintf(stderr, "Cannot load image \"%s\"\nSTB Reason: %s\n", filename, stbi_failure_reason());
        exit(0);
    }
    if (channels)
        c = channels;
    Image im = make_image(w, h, 1);
    int k = 0;
    for(int j = 0; j < h; ++j) {
        for(int i = 0; i < w; ++i) {
            int dst_index = i + w * j;
            // formula of calculating image gray value
            im.m_data[dst_index] = 0.299 * (float)data[k] + 0.587 * (float)data[k + 1] + 0.114 * (float)data[k + 2];
            k = k + 3;
        }
    }    
    free(data);
    return im;
}

float get_pixel(Image &m, const int width, const int height, const int channel) 
{
    assert (width < m.m_w && height < m.m_h && channel < m.m_c && "[get_pixel]: illegal input dimension!");
    return m.m_data[(channel * m.m_w * m.m_h) + (height * m.m_w) + width];
}

void set_pixel(Image &m, const int width, const int height, const int channel, const float val)
{
    if (width < 0 || height < 0 || channel < 0 || width >= m.m_w || height >= m.m_h || channel >= m.m_c) 
        return;
    assert(width < m.m_w && height < m.m_h && channel < m.m_c && "[set_pixel]: illegal input dimension!");
    m.m_data[(channel * m.m_h * m.m_w) + (height * m.m_w) + width] = val;
}

void add_pixel(Image &m, const int width, const int height, const int channel, const float val)
{
    assert(width < m.m_w && height < m.m_h && channel < m.m_c && "[add_pixel]: illegal input dimension!");
    m.m_data[(channel * m.m_h * m.m_w) + (height * m.m_w) + width] += val;
}   


// standardize the image
Image image_std(Image &im) 
{
    int w, h, c;
    float tmp, mean_value, std_value;
    float sum_value = 0;

    // calculate mean value 
    for (c = 0; c < im.m_c; ++c)
        for (h = 0; h < im.m_h; ++h)
            for (w = 0; w < im.m_w; ++w) {
                tmp = get_pixel(im, w, h, c);
                sum_value += tmp;
            }
    mean_value = sum_value / (im.m_h * im.m_w * im.m_c);

    // calculate standard deviation value
    sum_value = 0;
    for (c = 0; c < im.m_c; ++c) 
        for (h = 0; h < im.m_h; ++h) 
            for (w = 0; w < im.m_w; ++w) {
                tmp = pow((get_pixel(im, w, h, c) - mean_value), 2);
                sum_value += tmp;
            }
    std_value = sqrt(sum_value / (im.m_h * im.m_w * im.m_c));

    // initilize normalize image
    Image im_norm = make_image(im.m_w, im.m_h, 1);
    for (c = 0; c < im.m_c; ++c) 
        for (h = 0; h < im_norm.m_h; ++h) 
            for (w = 0; w < im_norm.m_w; ++w) {
                // data offset
                int dst_index = w + (h * im_norm.m_w) + (c * im_norm.m_h * im_norm.m_w);
                im_norm.m_data[dst_index] = (im.m_data[dst_index] - mean_value) / std_value;
            }
 
    return im_norm;
}

// normalize the image between [0, 1]
Image image_norm(Image &im)
{
    int w, h, c;
    float tmp;
    float max_value = -255.;
    float min_value = 255.;

    // calculate max and min value 
    for (c = 0; c < im.m_c; c++)
        for (h = 0; h < im.m_h; h++)
            for (w = 0; w < im.m_w; w++) {
                tmp = get_pixel(im, w, h, c);
                max_value = (tmp > max_value ? tmp : max_value);
                min_value = (tmp < min_value ? tmp : min_value); 
            }
            
    // initilize normalize image
    Image im_norm = make_image(im.m_w, im.m_h, 1);
    for (c = 0; c < im.m_c; c++) 
        for (h = 0; h < im_norm.m_h; h++) 
            for (w = 0; w < im_norm.m_w; w++) {
                // data offset
                int dst_index = w + (h * im_norm.m_w) + (c * im_norm.m_h * im_norm.m_w);
                im_norm.m_data[dst_index] = (im.m_data[dst_index] - min_value) / (max_value - min_value);
            }
 
    return im_norm;
}

void free_image(Image &m)
{
    if(m.m_data)  free(m.m_data);
}

Image image_resize(Image& im, const int w_resized, const int h_resized)
{
    Image im_resized = make_image(w_resized, h_resized, 1); 
    Image part = make_image(w_resized, im.m_h, im.m_c);
    int h, w, c;
    float w_scale = static_cast<float>(im.m_w) / static_cast<float>(w_resized);
    float h_scale = static_cast<float>(im.m_h) / static_cast<float>(h_resized);
    float val;

    // change width-direction data 
    for (c = 0; c < im.m_c; c++) 
        for (h = 0; h < im.m_h; h++) 
            for (w = 0; w < w_resized; w++) {
                float sx = w * w_scale;
                int ix = static_cast<int>(sx);
                float dx = sx - ix;
                // linear interpolation
                val = (1 - dx) * get_pixel(im, ix, h, c) + dx * get_pixel(im, ix + 1, h, c);
                set_pixel(part, w, h, c, val);
            }

    // change height-direction data
    for (c = 0; c < im.m_c; c++) 
        for (h = 0; h < h_resized; h++) {
            float sy = h * h_scale;
            int iy = static_cast<int>(sy);
            float dy = sy - iy;
            for (w = 0; w < w_resized; w++) {
                val = (1 - dy) * get_pixel(part, w, iy, c);
                set_pixel(im_resized, w, h, c, val);
            }
            for (w = 0; w < w_resized; w++) {
                val = dy * get_pixel(part, w, iy + 1, c);
                add_pixel(im_resized, w, h, c, val);
            }
        }

    free_image(part);
    return im_resized;
} 
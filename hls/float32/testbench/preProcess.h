#ifndef PREPROCESS
#define PREPROCESS

// store the RGB image
typedef struct s_Image {
public:
    int m_w;
    int m_h;
    int m_c;
    float *m_data;
    s_Image(): m_w(0), m_h(0), m_c(0), m_data(nullptr) {}
    s_Image(int w, int h, int c, float *data): m_w(w), m_h(h), m_c(c), m_data(data) {}
}Image;

Image load_image_stb_gray(const char *filename, int channels);
Image image_std(Image &im);
Image image_norm(Image &im);
Image image_resize(Image& im, int w_resized, int h_resized);
void write_to_txt(const char* name, float *input, int len);
void free_image(Image &m);

#endif

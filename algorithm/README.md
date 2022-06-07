This part is mainly for extracting weight and bias. In order to get faster reference speed, we also quantize the model from float32 to int8 using linear symmetric quantization
per layer. Here weight, bias and their responding quantization scales are written in .bin file.
The detailed float32 compute process on FPGA can be found at [link](https://github.com/zyGao1126/AFE_Net/tree/master/hls/float32), and int8 compute process on FPGA can be found at
[link](https://github.com/zyGao1126/AFE_Net/tree/master/hls/int8).

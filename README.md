# AFENet and its FPGA Accelerator in Xilinx Ultra96v2 SoC
A demo for accelerating AFENet in Xilinx Ultra96v2 FPGA. Here, AFE_Net(Adapted-Filter-Enhancement Network) is our another work which focus on 
improving image matching accuary by extracting high-accuracy keypoints and generating robust descriptors. The source code will be released after publication.

you can follow these steps to running this accelerator.

### 1. Software simulation
see more details in [README](https://github.com/zyGao1126/AFE_Net/blob/master/algorithm/README.md)

### 2. HLS Accelerator and Simulation
see more details in [README](https://github.com/zyGao1126/AFE_Net/blob/master/hls/README.md)

### 3. Vivado Block Design
see more details in [README](https://github.com/zyGao1126/AFE_Net/blob/master/vivado/README.md)

### 4. run on FPGA




In Xilinx Ultra96v2, the Frame Per Second can reach 23.8fps for 240x320 size image.

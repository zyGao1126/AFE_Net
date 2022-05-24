### This repo is about AFENet accelerator implemented in Vivado HLS 2019.2

* for golden repo, I did not add any HLS pragma optimization and just want to test the functional correctness. you can easily excute the following command to test the code.
```
$ cd golden
$ make
$ GFENet_FPGA ./test.jpg
```

* for float32 repo, I use float32 data to build FPGA accelerator. You can add files on Vivado HLS and generator corresponding IP.
* for int8 repo, I use data which have been quantized to build FPGA accelerator. 

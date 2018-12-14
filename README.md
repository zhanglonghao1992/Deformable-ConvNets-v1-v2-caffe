# Deformable-ConvNets(v1&v2)-caffe
  
Experiment Results：
--------  
**Model： Faster Rcnn （ResNet-50 backbone） without OHEM and Deformable Roi Pooling**    
**Dataset：train with voc 07+12 test on voc 07**    
  
**Deformable-V1:** 
  
 mAP@0.5 | aeroplane | bicycle |  bird  |  boat  | bottle |  bus  |  car  |  cat  | chair |  cow  |  
 ------- |-----------|-------- |--------|--------|--------|-------|-------|-------|-------|-------|
  0.7836 |   0.8004  |  0.8071 | 0.7909 | 0.7092 | 0.6297 | 0.8582| 0.8697| 0.8951| 0.6366| 0.8516|  
  
 diningtable |  dog  | horse | motorbike | person | pottedplant | sheep |  sofa  |  train  | tvmonitor |
 ----------- |-------|------ |-----------|------- |-------------|-------|------- |---------|-----------|
   0.7121    | 0.8822| 0.8837|  0.8162   | 0.7965 |    0.5449   | 0.7787| 0.7764 | 0.8725  |  0.7613   |  
     
Add the code to your caffe:  
--------
```
move deformable_conv_layer.cpp and deformable_conv_layer.cu to yourcaffepath/src/caffe/layers/
move modulated_deformable_conv_layer.cpp and modulated_deformable_conv_layer.cu to yourcaffepath/src/caffe/layers/
move deformable_conv_layer.hpp and modulated_deformable_conv_layer.hpp to yourcaffepath/include/caffe/layers/
move deformable_im2col.hpp and modulated_deformable_im2col.hpp to yourcaffepath/include/caffe/util/
move deformable_im2col.cu and modulated_deformable_im2col.cu to yourcaffepath/src/caffe/util/
```
  
  
edit caffe.proto:
```
optional DeformableConvolutionParameter deformable_convolution_param = 999999;  
optional ModulatedDeformableConvolutionParameter modulated_deformable_convolution_param = 9999999;  


message DeformableConvolutionParameter {
  optional uint32 num_output = 1; 
  optional bool bias_term = 2 [default = true]; 
  repeated uint32 pad = 3; // The padding size; defaults to 0
  repeated uint32 kernel_size = 4; // The kernel size
  repeated uint32 stride = 6; // The stride; defaults to 1
  repeated uint32 dilation = 18; // The dilation; defaults to 1
  optional uint32 pad_h = 9 [default = 0]; // The padding height (2D only)
  optional uint32 pad_w = 10 [default = 0]; // The padding width (2D only)
  optional uint32 kernel_h = 11; // The kernel height (2D only)
  optional uint32 kernel_w = 12; // The kernel width (2D only)
  optional uint32 stride_h = 13; // The stride height (2D only)
  optional uint32 stride_w = 14; // The stride width (2D only)
  optional uint32 group = 5 [default = 1]; 
  optional uint32 deformable_group = 25 [default = 1]; 
  optional FillerParameter weight_filler = 7; // The filler for the weight
  optional FillerParameter bias_filler = 8; // The filler for the bias
  enum Engine {
    DEFAULT = 0;
    CAFFE = 1;
    CUDNN = 2;
  }
  optional Engine engine = 15 [default = DEFAULT];
  optional int32 axis = 16 [default = 1];
  optional bool force_nd_im2col = 17 [default = false];
}


message ModulatedDeformableConvolutionParameter {
  optional uint32 num_output = 1; 
  optional bool bias_term = 2 [default = true]; 
  repeated uint32 pad = 3; // The padding size; defaults to 0
  repeated uint32 kernel_size = 4; // The kernel size
  repeated uint32 stride = 6; // The stride; defaults to 1
  repeated uint32 dilation = 18; // The dilation; defaults to 1
  optional uint32 pad_h = 9 [default = 0]; // The padding height (2D only)
  optional uint32 pad_w = 10 [default = 0]; // The padding width (2D only)
  optional uint32 kernel_h = 11; // The kernel height (2D only)
  optional uint32 kernel_w = 12; // The kernel width (2D only)
  optional uint32 stride_h = 13; // The stride height (2D only)
  optional uint32 stride_w = 14; // The stride width (2D only)
  optional uint32 group = 5 [default = 1]; 
  optional uint32 deformable_group = 25 [default = 1]; 
  optional FillerParameter weight_filler = 7; // The filler for the weight
  optional FillerParameter bias_filler = 8; // The filler for the bias
  enum Engine {
    DEFAULT = 0;
    CAFFE = 1;
    CUDNN = 2;
  }
  optional Engine engine = 15 [default = DEFAULT];
  optional int32 axis = 16 [default = 1];
  optional bool force_nd_im2col = 17 [default = false];
}
```
Model structure:  
--------
Deformable_ConvNet_V1 in ResNet:    
![Deformable_ConvNet_V1](https://github.com/zhanglonghao1992/ReadmeImages/blob/master/images/WFOB%60M_%24AD9I4BHW3L4JV5F.png)    
    
    
Deformable_ConvNet_V2 in Resnet:      
![Deformable_ConvNet_V2](https://github.com/zhanglonghao1992/ReadmeImages/blob/master/images/ZHR5PSZBMDJS48%605YZY.png)      
  
Acknowlegement:  
---------
Thanks to [offical mxnet code](https://github.com/msracver/Deformable-ConvNets)    
Thanks to [unsky](https://github.com/unsky/Deformable-ConvNets-caffe)    

# Deformable-ConvNets(v1&v2)-caffe
Thanks to [offical mxnet code](https://github.com/msracver/Deformable-ConvNets)  
Thanks to [unsky](https://github.com/unsky/Deformable-ConvNets-caffe)  
  
Deformable_ConvNet_V1:  
![Deformable_ConvNet_V1](https://github.com/zhanglonghao1992/ReadmeImages/blob/master/images/WFOB%60M_%24AD9I4BHW3L4JV5F.png)  
  
  
Deformable_ConvNet_V2:  
![Deformable_ConvNet_V2](https://github.com/zhanglonghao1992/ReadmeImages/blob/master/images/ZHR5PSZBMDJS48%605YZY.png)  
  
  
caffe.proto:  
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

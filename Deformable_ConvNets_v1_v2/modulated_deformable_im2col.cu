#include <algorithm>
#include <iostream>

#include "caffe/common.hpp"
#include "caffe/util/gpu_util.cuh"
#include "caffe/util/modulated_deformable_im2col.hpp"
using namespace std;

namespace caffe {

template <typename Dtype>
__device__ Dtype dmcn_im2col_bilinear(const Dtype* bottom_data, const int data_width, 
  const int height, const int width, Dtype h, Dtype w) {

  int h_low = floor(h);
  int w_low = floor(w);
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  Dtype lh = h - h_low;
  Dtype lw = w - w_low;
  Dtype hh = 1 - lh, hw = 1 - lw;

  Dtype v1 = 0;
  if (h_low >= 0 && w_low >= 0)
    v1 = bottom_data[h_low * data_width + w_low];
  Dtype v2 = 0;
  if (h_low >=0 && w_high <= width - 1)
    v2 = bottom_data[h_low * data_width + w_high];
  Dtype v3 = 0;
  if (h_high <= height - 1 && w_low >= 0)
    v3 = bottom_data[h_high * data_width + w_low];
  Dtype v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1)
    v4 = bottom_data[h_high * data_width + w_high];
  
  Dtype w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  Dtype val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}


template <typename Dtype>
__device__ Dtype dmcn_get_gradient_weight(Dtype argmax_h, Dtype argmax_w, 
  const int h, const int w, const int height, const int width) {

  if (argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 || argmax_w >= width) {
    //empty
    return 0;
  }

  int argmax_h_low = floor(argmax_h);
  int argmax_w_low = floor(argmax_w);
  int argmax_h_high = argmax_h_low + 1;
  int argmax_w_high = argmax_w_low + 1;

  Dtype weight = 0;
  if (h == argmax_h_low && w == argmax_w_low)
      weight = (h + 1 - argmax_h) * (w + 1 - argmax_w);
  if (h == argmax_h_low && w == argmax_w_high)
      weight = (h + 1 - argmax_h) * (argmax_w + 1 - w);
  if (h == argmax_h_high && w == argmax_w_low)
      weight = (argmax_h + 1 - h) * (w + 1 - argmax_w);
  if (h == argmax_h_high && w == argmax_w_high)
      weight = (argmax_h + 1 - h) * (argmax_w + 1 - w);
  return weight;
}


template <typename Dtype>
__device__ Dtype dmcn_get_coordinate_weight(Dtype argmax_h, Dtype argmax_w,
  const int height, const int width, const Dtype* im_data,
  const int data_width, const int bp_dir) {

  if (argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 || argmax_w >= width)
  {
    //empty
    return 0;
  }

  int argmax_h_low = floor(argmax_h);
  int argmax_w_low = floor(argmax_w);
  int argmax_h_high = argmax_h_low + 1;
  int argmax_w_high = argmax_w_low + 1;
  
  Dtype weight = 0;

  if (bp_dir == 0) {
    if (argmax_h_low >= 0 && argmax_w_low >= 0)
        weight += -1 * (argmax_w_low + 1 - argmax_w) * im_data[argmax_h_low * data_width + argmax_w_low];
    if (argmax_h_low >= 0 && argmax_w_high <= width - 1)
        weight += -1 * (argmax_w - argmax_w_low) * im_data[argmax_h_low * data_width + argmax_w_high];
    if (argmax_h_high <= height - 1 && argmax_w_low >= 0)
        weight += (argmax_w_low + 1 - argmax_w) * im_data[argmax_h_high * data_width + argmax_w_low];
    if (argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
        weight += (argmax_w - argmax_w_low) * im_data[argmax_h_high * data_width + argmax_w_high];
  } else if (bp_dir == 1) {
    if (argmax_h_low >= 0 && argmax_w_low >= 0)
        weight += -1 * (argmax_h_low + 1 - argmax_h) * im_data[argmax_h_low * data_width + argmax_w_low];
    if (argmax_h_low >= 0 && argmax_w_high <= width - 1)
        weight += (argmax_h_low + 1 - argmax_h) * im_data[argmax_h_low * data_width + argmax_w_high];
    if (argmax_h_high <= height - 1 && argmax_w_low >= 0)
        weight += -1 * (argmax_h - argmax_h_low) * im_data[argmax_h_high * data_width + argmax_w_low];
    if (argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
        weight += (argmax_h - argmax_h_low) * im_data[argmax_h_high * data_width + argmax_w_high];
  }

  return weight;
}


/*!
 * \brief deformable_im2col gpu kernel.
 * DO NOT call this directly. Use wrapper function im2col() instead;
 */
template <typename Dtype>
__global__ void modulated_deformable_im2col_gpu_kernel(const int n, const Dtype* data_im, const Dtype* data_offset, const Dtype* data_mask,
  const int height, const int width, const int kernel_h, const int kernel_w,
  const int pad_h, const int pad_w,
  const int stride_h, const int stride_w,
  const int dilation_h, const int dilation_w,
  const int channel_per_deformable_group,
  const int height_col, const int width_col,
  Dtype* data_col) {
  CUDA_KERNEL_LOOP(index, n) {
	// index index of output matrix
    const int w_col = index % width_col;
    const int h_col = (index / width_col) % height_col;
    const int c_im = (index / width_col) / height_col;
    const int c_col = c_im * kernel_h * kernel_w;

    // compute deformable group index
    const int deformable_group_index = c_im / channel_per_deformable_group;

    const int h_in = h_col * stride_h - pad_h;
    const int w_in = w_col * stride_w - pad_w;
    Dtype* data_col_ptr = data_col + (c_col * height_col + h_col) * width_col + w_col;

    //const Dtype* data_im_ptr = data_im + (c_im * height + h_in) * width + w_in;
	const Dtype* data_im_ptr = data_im + (c_im * height) * width;
    const Dtype* data_offset_ptr = data_offset + deformable_group_index * 2 * kernel_h * kernel_w * height_col * width_col;
	const Dtype* data_mask_ptr = data_mask + deformable_group_index * kernel_h * kernel_w * height_col * width_col;


    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        const int data_offset_h_ptr = ((2 * (i * kernel_w + j)) * height_col + h_col) * width_col + w_col;
        const int data_offset_w_ptr = ((2 * (i * kernel_w + j) + 1) * height_col + h_col) * width_col + w_col;
		const int data_mask_hw_ptr = ((i * kernel_w + j) * height_col + h_col) * width_col + w_col;
        const Dtype offset_h = data_offset_ptr[data_offset_h_ptr];
        const Dtype offset_w = data_offset_ptr[data_offset_w_ptr];
		const Dtype mask = data_mask_ptr[data_mask_hw_ptr];
        Dtype val = static_cast<Dtype>(0);
        const Dtype h_im = h_in + i * dilation_h + offset_h;
        const Dtype w_im = w_in + j * dilation_w + offset_w;
        //if (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) {
		if (h_im > -1 && w_im > -1 && h_im < height && w_im < width) {
          //const Dtype map_h = i * dilation_h + offset_h;
          //const Dtype map_w = j * dilation_w + offset_w;
          //const int cur_height = height - h_in;
          //const int cur_width = width - w_in;
          //val = deformable_im2col_bilinear(data_im_ptr, width, cur_height, cur_width, map_h, map_w);
		  val = dmcn_im2col_bilinear(data_im_ptr, width, height, width, h_im, w_im);
        }
        *data_col_ptr = val * mask;
        data_col_ptr += height_col * width_col;
      }
    }
  }
}
template <typename Dtype>
void modulated_deformable_im2col_gpu(const Dtype* data_im, const Dtype* data_offset, const Dtype* data_mask, const int channels,
	const int height, const int width, const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w,
	const int stride_h, const int stride_w,
	const int dilation_h, const int dilation_w,
	const int deformable_group,
	Dtype* data_col) {
  // We are going to launch channels * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
  int height_col = (height + 2 * pad_h -
	  (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  int width_col = (width + 2 * pad_w -
	  (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  int num_kernels = channels * height_col * width_col;
  int channel_per_deformable_group =  channels/ deformable_group;
  modulated_deformable_im2col_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
							 CAFFE_CUDA_NUM_THREADS>>>(
	  num_kernels, data_im, data_offset, data_mask, height, width, kernel_h, kernel_w, pad_h,
	  pad_w, stride_h, stride_w, dilation_h, dilation_w, channel_per_deformable_group, height_col,
    width_col, data_col);

  CUDA_POST_KERNEL_CHECK;
}

template void modulated_deformable_im2col_gpu<float>(const float* data_im, const float* data_offset, const float* data_mask, const int channels,
	const int height, const int width, const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w,
	const int stride_h, const int stride_w,
	const int dilation_h, const int dilation_w,
	const int deformable_group,
	float* data_col);
template void modulated_deformable_im2col_gpu<double>(const double* data_im, const double* data_offset, const double* data_mask, const int channels,
	const int height, const int width, const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w,
	const int stride_h, const int stride_w,
	const int dilation_h, const int dilation_w,
	const int deformable_group,
	double* data_col);



template <typename Dtype>
__global__ void modulated_deformable_col2im_gpu_kernel(const int n, const Dtype* data_col, 
  const Dtype* data_offset, const Dtype* data_mask,
  const int channels, const int height, const int width,
  const int kernel_h, const int kernel_w,
  const int pad_h, const int pad_w,
  const int stride_h, const int stride_w,
  const int dilation_h, const int dilation_w,
  int channel_per_deformable_group,
  int height_col, int width_col,
  Dtype* grad_im) {
  CUDA_KERNEL_LOOP(index, n) {
    const int j = (index / width_col / height_col) % kernel_w;
    const int i = (index / width_col / height_col / kernel_w) % kernel_h;
    const int c = index / width_col / height_col / kernel_w / kernel_h;
    // compute the start and end of the output

    const int deformable_group_index = c / channel_per_deformable_group;

    int w_out = index % width_col;
    int h_out = (index / width_col) % height_col;
    int w_in = w_out * stride_w - pad_w;
    int h_in = h_out * stride_h - pad_h;

    const Dtype* data_offset_ptr = data_offset + deformable_group_index * 2 * kernel_h * kernel_w * height_col * width_col;
	const Dtype* data_mask_ptr = data_mask + deformable_group_index * kernel_h * kernel_w * height_col * width_col;
    const int data_offset_h_ptr = ((2 * (i * kernel_w + j)) * height_col + h_out) * width_col + w_out;
    const int data_offset_w_ptr = ((2 * (i * kernel_w + j) + 1) * height_col + h_out) * width_col + w_out;
	const int data_mask_hw_ptr = ((i * kernel_w + j) * height_col + h_out) * width_col + w_out;
    const Dtype offset_h = data_offset_ptr[data_offset_h_ptr];
    const Dtype offset_w = data_offset_ptr[data_offset_w_ptr];
	const Dtype mask = data_mask_ptr[data_mask_hw_ptr];
    const Dtype cur_inv_h_data = h_in + i * dilation_h + offset_h;
    const Dtype cur_inv_w_data = w_in + j * dilation_w + offset_w;

    const Dtype cur_top_grad = data_col[index] * mask;
    const int cur_h = (int)cur_inv_h_data;
    const int cur_w = (int)cur_inv_w_data;
    for (int dy = -2; dy <= 2; dy++) {
      for (int dx = -2; dx <= 2; dx++) {
        if (cur_h + dy >= 0 && cur_h + dy < height &&
          cur_w + dx >= 0 && cur_w + dx < width &&
          abs(cur_inv_h_data - (cur_h + dy)) < 1 &&
          abs(cur_inv_w_data - (cur_w + dx)) < 1
          ) {
          int cur_bottom_grad_pos = (c * height + cur_h + dy) * width + cur_w + dx;
          Dtype weight = dmcn_get_gradient_weight(cur_inv_h_data, cur_inv_w_data, cur_h + dy, cur_w + dx, height, width);
          caffe_gpu_atomic_add(weight * cur_top_grad, grad_im + cur_bottom_grad_pos);
        }
      }
    }
  }
}
template <typename Dtype>
void modulated_deformable_col2im_gpu(const Dtype* data_col, const Dtype* data_offset, const Dtype* data_mask,
   const int channels,const int height, const int width, const int num_kernels,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    const int deformable_group, Dtype* grad_im) {
  int height_col = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) /
      stride_h + 1;
  int width_col = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) /
      stride_w + 1;
  int channel_per_deformable_group = channels / deformable_group;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  // NOLINT_NEXT_LINE(whitespace/operators)
  modulated_deformable_col2im_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                             CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, data_col, data_offset, data_mask, channels, height, width,  kernel_h, kernel_w,
      pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
      channel_per_deformable_group, height_col, width_col, grad_im);
  CUDA_POST_KERNEL_CHECK;
}

template void modulated_deformable_col2im_gpu<float>(const float* data_col, const float* data_offset, const float* data_mask, const int channels,
    const int height, const int width,const int num_kernels, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    const int deformable_group, float* grad_im);
template void modulated_deformable_col2im_gpu<double>(const double* data_col, const double* data_offset, const double* data_mask, const int channels,
    const int height, const int width, const int num_kernels,const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    const int deformable_group, double* grad_im);


template <typename Dtype>
__global__ void modulated_deformable_col2im_coord_gpu_kernel(const int n, const Dtype* data_col, 
  const Dtype* data_im, const Dtype* data_offset, const Dtype* data_mask,
  const int channels, const int height, const int width,
  const int kernel_h, const int kernel_w,
  const int pad_h, const int pad_w,
  const int stride_h, const int stride_w,
  const int dilation_h, const int dilation_w,
  const int channel_per_deformable_group,
  const int height_col, const int width_col,
  Dtype* grad_offset, Dtype* grad_mask) {
  CUDA_KERNEL_LOOP(index, n) {
    Dtype val = 0, mval = 0;
    int w = index % width_col;
    int h = (index / width_col) % height_col;
    int c = index / width_col / height_col;
    // compute the start and end of the output

    const int deformable_group_index = c / (2 * kernel_h * kernel_w);
    const int col_step = kernel_h * kernel_w;
    int cnt = 0;
    const Dtype* data_col_ptr = data_col + deformable_group_index * channel_per_deformable_group * width_col * height_col;
    const Dtype* data_im_ptr = data_im + deformable_group_index * channel_per_deformable_group / kernel_h / kernel_w * height * width;
    const Dtype* data_offset_ptr = data_offset + deformable_group_index * 2 * kernel_h * kernel_w * height_col * width_col;
	const Dtype* data_mask_ptr = data_mask + deformable_group_index * kernel_h * kernel_w * height_col * width_col;

    const int offset_c = c - deformable_group_index * 2 * kernel_h * kernel_w;

    for (int col_c = (offset_c / 2); col_c < channel_per_deformable_group; col_c += col_step) {
      const int col_pos = ((col_c * height_col) + h) * width_col + w;
      const int bp_dir = offset_c % 2;

      int j = (col_pos / width_col / height_col) % kernel_w;
      int i = (col_pos / width_col / height_col / kernel_w) % kernel_h;
      int w_out = col_pos % width_col;
      int h_out = (col_pos / width_col) % height_col;
      int w_in = w_out * stride_w - pad_w;
      int h_in = h_out * stride_h - pad_h;
      const int data_offset_h_ptr = (((2 * (i * kernel_w + j)) * height_col + h_out) * width_col + w_out);
      const int data_offset_w_ptr = (((2 * (i * kernel_w + j) + 1) * height_col + h_out) * width_col + w_out);
	  const int data_mask_hw_ptr = (((i * kernel_w + j) * height_col + h_out) * width_col + w_out);
      const Dtype offset_h = data_offset_ptr[data_offset_h_ptr];
      const Dtype offset_w = data_offset_ptr[data_offset_w_ptr];
	  const Dtype mask = data_mask_ptr[data_mask_hw_ptr];
      Dtype inv_h = h_in + i * dilation_h + offset_h;
      Dtype inv_w = w_in + j * dilation_w + offset_w;
      if (inv_h < -1 || inv_w < -1 || inv_h >= height || inv_w >= width) {
        inv_h = inv_w = -2;
      }else {
        mval += data_col_ptr[col_pos] * dmcn_im2col_bilinear(data_im_ptr + cnt * height * width, width, height, width, inv_h, inv_w);
      }
      const Dtype weight = dmcn_get_coordinate_weight(
        inv_h, inv_w,
        height, width, data_im_ptr + cnt * height * width, width, bp_dir);
      val += weight * data_col_ptr[col_pos] * mask;
      cnt += 1;
    }

    grad_offset[index] = val;
	if (offset_c % 2 == 0){
	    grad_mask[((deformable_group_index * kernel_h * kernel_w + offset_c / 2)* height_col + h) * width_col + w] = mval;
	}


  }
}
template <typename Dtype>
void modulated_deformable_col2im_coord_gpu(const Dtype* data_col, const Dtype* data_im, const Dtype* data_offset, const Dtype* data_mask, const int channels,
    const int height, const int width, 
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    const int deformable_group, Dtype* grad_offset, Dtype* grad_mask) {
  int height_col = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) /
      stride_h + 1;
  int width_col = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) /
      stride_w + 1;
  int num_kernels = height_col * width_col * 2 * kernel_h * kernel_h * deformable_group;

  int channel_per_deformable_group = channels/ deformable_group;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  // NOLINT_NEXT_LINE(whitespace/operators)
  modulated_deformable_col2im_coord_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                             CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, data_col, data_im,data_offset, data_mask, channels,height, width,  kernel_h, kernel_w,
      pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
      channel_per_deformable_group, height_col, width_col, grad_offset, grad_mask);
  CUDA_POST_KERNEL_CHECK;
}

template void modulated_deformable_col2im_coord_gpu<float>(const float* data_col, const float* data_im,const float* data_offset, const float* data_mask, const int channels,
const int height, const int width, const int kernel_h, const int kernel_w,
const int pad_h, const int pad_w, const int stride_h,
const int stride_w, const int dilation_h, const int dilation_w,
const int deformable_group, float* grad_offset, float* grad_mask);

template void modulated_deformable_col2im_coord_gpu<double>(const double* data_col, const double* data_im,const double* data_offset, const double* data_mask, const int channels,
const int height, const int width, const int kernel_h, const int kernel_w,
const int pad_h, const int pad_w, const int stride_h,
const int stride_w, const int dilation_h, const int dilation_w,
const int deformable_group, double* grad_offset, double* grad_mask);
}

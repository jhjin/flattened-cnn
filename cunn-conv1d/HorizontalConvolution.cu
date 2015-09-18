#include "utils.h"
#include "common.h"

// Kernel for fast unfold+copy
// (borrowed from Caffe: https://github.com/BVLC/caffe/blob/master/src/caffe/layers/conv_layer.cu)
__global__ void im2col_kernel_h(const int n, const float* data_im,
    const int height, const int width, const int ksize_h, const int ksize_w, const int pad_h,
    const int pad_w, const int stride_h, const int stride_w, const int height_col, const int width_col,
    float* data_col) {
  CUDA_KERNEL_LOOP(index, n) {
    int w_out = index % width_col;
    index /= width_col;
    int h_out = index % height_col;
    int channel_in = index / height_col;
    int channel_out = channel_in * ksize_h * ksize_w;
    int h_in = h_out * stride_h - pad_h;
    int w_in = w_out * stride_w - pad_w;
    data_col += (channel_out * height_col + h_out) * width_col + w_out;
    data_im += (channel_in * height + h_in) * width + w_in;
    for (int i = 0; i < ksize_h; ++i) {
      for (int j = 0; j < ksize_w; ++j) {
        int h = h_in + i;
        int w = w_in + j;
        *data_col = (h >= 0 && w >= 0 && h < height && w < width) ?
          data_im[i * width + j] : 0;
        data_col += height_col * width_col;
      }
    }
  }
}


__global__ void conv_horizontal_naive_output(const int n, float *y,
                                             const float *x, const float *w,
                                             const int iH, const int iW, const int kL)
{
   for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < n; i += blockDim.x*gridDim.x) {
      int oW = iW - kL + 1;
      int x_offset = (i/oW)*iW + i%oW;
      int w_offset = (i/(oW*iH))*kL;

      for (int k = 0; k < kL; k++) {
         y[i] += w[w_offset + k]*x[x_offset + k];
      }
   }
}


__global__ void conv_horizontal_naive_gradInput(const int n, float *dx,
                                                const float *dy, const float *w,
                                                const int oH, const int oW, const int kL)
{
   for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < n; i += blockDim.x*gridDim.x) {
      int iW = oW + kL - 1;
      int col = i%iW;
      int dy_offset = (i/iW)*oW + i%iW;
      int w_offset = (i/(iW*oH))*kL;

      int k_begin = max(0, col-oW+1);
      int k_end = min(kL, col+1);

      dx[i] = 0.0f;
      for (int k = k_begin; k < k_end; k++) {
         dx[i] += w[w_offset + k]*dy[dy_offset - k];
      }
   }
}


__global__ void conv_horizontal_naive_gradParam(const int n, float *dw,
                                                const float *x, const float *dy,
                                                const int kL, const int oH, const int oW)
{
   for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < n; i += blockDim.x*gridDim.x) {
      int iW = oW + kL - 1;
      int dy_offset = (i/kL)*oH*oW;
      int x_offset = (i/kL)*oH*oW + i%kL;

      for (int j = 0; j < oH; j++) {
         for (int k = 0; k < oW; k++) {
            dw[i] += dy[dy_offset + j*oW + k]*x[x_offset + j*iW + k];
         }
      }
   }
}


__global__ void conv_horizontal_naive_gradWeight(const int n, float *y,
                                                 const float *x, const int kL,
                                                 const int iC)
{
   for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < n; i += blockDim.x*gridDim.x) {
      y[i] = x[(i/kL)*kL*iC + i];
   }
}


static int cunnconv1d_HorizontalConvolution_updateOutput(lua_State *L) {
   THCState *state = getCutorchState(L);
   THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");

   int nInputPlane = luaT_getfieldcheckint(L, 1, "nInputPlane");
   int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");

   THCudaTensor *weight = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "weight", "torch.CudaTensor");
   THCudaTensor *bias = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "bias", "torch.CudaTensor");
   THCudaTensor *ones = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "ones", "torch.CudaTensor");
   THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");

   const int device = THCudaTensor_getDevice(state, weight);
   luaL_argcheck(L, THCudaTensor_getDevice(state, bias) == device, 1,
                 "weight and bias need to be on the same device");
   luaL_argcheck(L, THCudaTensor_getDevice(state, output) == device ||
                 THCudaTensor_getDevice(state, output) == -1, 1,
                 "weight and output need to be on the same device");
   luaL_argcheck(L, THCudaTensor_getDevice(state, input) == device, 2,
                 "weight and input need to be on the same device");
   luaL_argcheck(L, input->nDimension == 3 || input->nDimension == 4, 2,
                 "3D or 4D (batch mode) tensor is expected");

   // change to batch mode
   int batch = 1;
   if (input->nDimension == 3) {
      luaL_argcheck(L, input->size[0] == nInputPlane, 2, "input channels and nInputPlane dont match");
      batch = 0;
      THCudaTensor_resize4d(state, input, 1, input->size[0], input->size[1], input->size[2]);
   } else {
      luaL_argcheck(L, input->size[1] == nInputPlane, 2, "input channels and nInputPlane dont match");
   }

   long batchSize    = input->size[0];
   long inputHeight  = input->size[2];
   long inputWidth   = input->size[3];
   long outputHeight = inputHeight;
   long outputWidth  = inputWidth - weight->size[1] + 1;

   THCudaTensor_resize4d(state, output, batchSize, nOutputPlane, outputHeight, outputWidth);

   if (ones->nDimension != 2 || ones->size[0]*ones->size[1] < outputHeight*outputWidth) {
      THCudaTensor_resize2d(state, ones, outputHeight, outputWidth);
      THCudaTensor_fill(state, ones, 1);
   }

   THCudaTensor *input_n = THCudaTensor_new(state);
   THCudaTensor *output_n = THCudaTensor_new(state);

   for (int elt = 0; elt < batchSize; elt ++) {

      // select each batch
      THCudaTensor_select(state, input_n, input, 0, elt);
      THCudaTensor_select(state, output_n, output, 0, elt);

      // fill biases
      THCudaBlas_gemm(
         state, 't', 'n',
         outputHeight*outputWidth, nOutputPlane, 1,
         1,
         THCudaTensor_data(state, ones), 1,
         THCudaTensor_data(state, bias), 1,
         0,
         THCudaTensor_data(state, output_n), outputHeight*outputWidth
      );

      // convolve
      long num_threads = nOutputPlane*outputHeight*outputWidth;
      conv_horizontal_naive_output <<<GET_BLOCKS(num_threads), CUDA_NUM_THREADS>>>
         (num_threads,
          THCudaTensor_data(state, output_n),
          THCudaTensor_data(state, input_n),
          THCudaTensor_data(state, weight),
          inputHeight, inputWidth, weight->size[1]);
   }

   THCudaTensor_free(state, input_n);
   THCudaTensor_free(state, output_n);

   // revert to single batch
   if (batch == 0) {
      THCudaTensor_resize3d(state, output, nOutputPlane, outputHeight, outputWidth);
      THCudaTensor_resize3d(state, input, nInputPlane, inputHeight, inputWidth);
   }

   return 1;
}


static int cunnconv1d_HorizontalConvolution_updateGradInput(lua_State *L) {
   THCState *state = getCutorchState(L);
   THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
   THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");

   int nInputPlane = luaT_getfieldcheckint(L, 1, "nInputPlane");
   int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");

   THCudaTensor *weight = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "weight", "torch.CudaTensor");
   THCudaTensor *gradInput = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");

   const int device = THCudaTensor_getDevice(state, weight);
   luaL_argcheck(L, THCudaTensor_getDevice(state, input) == device, 2,
                 "weight and input need to be on the same device");
   luaL_argcheck(L, THCudaTensor_getDevice(state, gradInput) == device
                 || THCudaTensor_getDevice(state, gradInput) == -1, 2,
                 "weight and gradInput need to be on the same device");
   luaL_argcheck(L, THCudaTensor_getDevice(state, gradOutput) == device
                 || THCudaTensor_getDevice(state, gradOutput) == -1, 2,
                 "weight and gradOutput need to be on the same device");
   luaL_argcheck(L, input->nDimension == 3 || input->nDimension == 4, 2,
                 "3D or 4D (batch mode) tensor is expected");

   int batch = 1;
   if (input->nDimension == 3) {
      batch = 0;
      THCudaTensor_resize4d(state, input, 1, input->size[0], input->size[1], input->size[2]);
      THCudaTensor_resize4d(state, gradOutput, 1, gradOutput->size[0], gradOutput->size[1], gradOutput->size[2]);
   }

   long batchSize    = input->size[0];
   long inputHeight  = input->size[2];
   long inputWidth   = input->size[3];
   long outputHeight = inputHeight;
   long outputWidth  = inputWidth - weight->size[1] + 1;

   THCudaTensor_resize4d(state, gradInput, batchSize, nInputPlane, inputHeight, inputWidth);

   THCudaTensor *gradInput_n = THCudaTensor_new(state);
   THCudaTensor *gradOutput_n = THCudaTensor_new(state);

   for (int elt = 0; elt < batchSize; elt ++) {

      // select each batch in 2D
      THCudaTensor_select(state, gradInput_n, gradInput, 0, elt);
      THCudaTensor_select(state, gradOutput_n, gradOutput, 0, elt);

      // convolve
      long num_threads = nInputPlane*inputHeight*inputWidth;
      conv_horizontal_naive_gradInput <<<GET_BLOCKS(num_threads), CUDA_NUM_THREADS>>>
         (num_threads,
          THCudaTensor_data(state, gradInput_n),
          THCudaTensor_data(state, gradOutput_n),
          THCudaTensor_data(state, weight),
          outputHeight, outputWidth, weight->size[1]);
   }

   THCudaTensor_free(state, gradInput_n);
   THCudaTensor_free(state, gradOutput_n);

   // revert to single batch
   if (batch == 0) {
      THCudaTensor_resize3d(state, input, nInputPlane, inputHeight, inputWidth);
      THCudaTensor_resize3d(state, gradInput, nInputPlane, inputHeight, inputWidth);
      THCudaTensor_resize3d(state, gradOutput, nOutputPlane, outputHeight, outputWidth);
   }

   return 1;
}


static int cunnconv1d_HorizontalConvolution_accGradParameters(lua_State *L) {
   THCState *state = getCutorchState(L);
   THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
   THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");

   float scale = luaL_optnumber(L, 4, 1);
   int nInputPlane = luaT_getfieldcheckint(L, 1, "nInputPlane");
   int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");
   int kL = luaT_getfieldcheckint(L, 1, "kL");

   THCudaTensor *gradWeight = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "gradWeight", "torch.CudaTensor");
   THCudaTensor *gradBias = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "gradBias", "torch.CudaTensor");
   THCudaTensor *ones = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "ones", "torch.CudaTensor");
   THCudaTensor *finput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "finput", "torch.CudaTensor");
   THCudaTensor *fgradWeight = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "fgradWeight", "torch.CudaTensor");

   const int device = THCudaTensor_getDevice(state, gradWeight);
   luaL_argcheck(L, THCudaTensor_getDevice(state, gradBias) == device, 1,
                 "gradWeight and gradBias need to be on the same device");
   luaL_argcheck(L, THCudaTensor_getDevice(state, input) == device, 1,
                 "gradWeight and input need to be on the same device");
   luaL_argcheck(L, THCudaTensor_getDevice(state, gradOutput) == device, 1,
                 "gradWeight and gradOutput need to be on the same device");
   luaL_argcheck(L, input->nDimension == 3 || input->nDimension == 4, 2,
                 "3D or 4D (batch mode) tensor is expected");

   // change to batch mode
   int batch = 1;
   if (input->nDimension == 3) {
      batch = 0;
      THCudaTensor_resize4d(state, input, 1, input->size[0], input->size[1], input->size[2]);
      THCudaTensor_resize4d(state, gradOutput, 1, gradOutput->size[0], gradOutput->size[1], gradOutput->size[2]);
   }

   long batchSize    = input->size[0];
   long inputHeight  = input->size[2];
   long inputWidth   = input->size[3];
   long outputHeight = inputHeight;
   long outputWidth  = inputWidth - kL + 1;

   if (ones->nDimension != 2 || ones->size[0]*ones->size[1] < outputHeight*outputWidth) {
      THCudaTensor_resize2d(state, ones, outputHeight, outputWidth);
      THCudaTensor_fill(state, ones, 1);
   }

   THCudaTensor_resize2d(state, finput, kL*nInputPlane, outputHeight*outputWidth);
   THCudaTensor_resize2d(state, fgradWeight, nOutputPlane, kL*nInputPlane);

   THCudaTensor *input_n = THCudaTensor_new(state);
   THCudaTensor *gradOutput_n = THCudaTensor_new(state);

   for (int elt = 0; elt < batchSize; elt ++) {

      // select each batch
      THCudaTensor_select(state, input_n, input, 0, elt);
      THCudaTensor_select(state, gradOutput_n, gradOutput, 0, elt);

      // unroll
      long num_threads = nInputPlane*outputHeight*outputWidth;
      im2col_kernel_h <<<GET_BLOCKS(num_threads), CUDA_NUM_THREADS>>> (
         num_threads,
         THCudaTensor_data(state, input_n),
         inputHeight, inputWidth, 1, kL, 0, 0, 1, 1,
         outputHeight, outputWidth,
         THCudaTensor_data(state, finput)
      );

      // convolve
      THCudaBlas_gemm(
         state, 't', 'n',
         kL*nInputPlane, nOutputPlane, outputHeight*outputWidth,
         scale,
         THCudaTensor_data(state, finput), outputHeight*outputWidth,
         THCudaTensor_data(state, gradOutput_n), outputHeight*outputWidth,
         (elt > 0),
         THCudaTensor_data(state, fgradWeight), kL*nInputPlane
      );

      // fill biases
      THCudaBlas_gemv(
         state,
         't',
         outputHeight*outputWidth, nOutputPlane,
         scale,
         THCudaTensor_data(state, gradOutput_n), outputHeight*outputWidth,
         THCudaTensor_data(state, ones), 1,
         1,
         THCudaTensor_data(state, gradBias), 1
      );
   }

   // extract gradWeight
   long num_threads_ = kL*nInputPlane;
   conv_horizontal_naive_gradWeight <<<GET_BLOCKS(num_threads_), CUDA_NUM_THREADS>>> (
      num_threads_,
      THCudaTensor_data(state, gradWeight),
      THCudaTensor_data(state, fgradWeight),
      kL, nInputPlane
   );

   THCudaTensor_free(state, input_n);
   THCudaTensor_free(state, gradOutput_n);

   if (batch == 0) {
      THCudaTensor_resize3d(state, gradOutput, nOutputPlane, outputHeight, outputWidth);
      THCudaTensor_resize3d(state, input, nInputPlane, inputHeight, inputWidth);
   }

   return 0;
}


static const struct luaL_Reg cunnconv1d_HorizontalConvolution__ [] = {
   {"HorizontalConvolution_updateOutput", cunnconv1d_HorizontalConvolution_updateOutput},
   {"HorizontalConvolution_updateGradInput", cunnconv1d_HorizontalConvolution_updateGradInput},
   {"HorizontalConvolution_accGradParameters", cunnconv1d_HorizontalConvolution_accGradParameters},
   {NULL, NULL}
};


void cunnconv1d_HorizontalConvolution_init(lua_State *L)
{
   luaT_pushmetatable(L, "torch.CudaTensor");
   luaT_registeratname(L, cunnconv1d_HorizontalConvolution__, "nn");
   lua_pop(L,1);
}

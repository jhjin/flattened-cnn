#include "utils.h"


static int cunnconv1d_LateralConvolution_updateOutput(lua_State *L) {
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
   long outputWidth  = inputWidth;

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
      THCudaBlas_gemm(
         state,
         'n', 'n',
         outputHeight*outputWidth, nOutputPlane, nInputPlane,
         1,
         THCudaTensor_data(state, input_n), outputHeight*outputWidth,
         THCudaTensor_data(state, weight), nInputPlane,
         1,
         THCudaTensor_data(state, output_n), outputHeight*outputWidth
      );
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


static int cunnconv1d_LateralConvolution_updateGradInput(lua_State *L) {
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
   long outputWidth  = inputWidth;

   THCudaTensor_resize4d(state, gradInput, batchSize, nInputPlane, inputHeight, inputWidth);

   THCudaTensor *gradInput_n = THCudaTensor_new(state);
   THCudaTensor *gradOutput_n = THCudaTensor_new(state);

   for (int elt = 0; elt < batchSize; elt ++) {

      // select each batch in 2D
      THCudaTensor_select(state, gradInput_n, gradInput, 0, elt);
      THCudaTensor_select(state, gradOutput_n, gradOutput, 0, elt);

      // convolve
      THCudaBlas_gemm(
         state,
         'n', 't',
         outputHeight*outputWidth, nInputPlane, nOutputPlane,
         1,
         THCudaTensor_data(state, gradOutput_n), outputHeight*outputWidth,
         THCudaTensor_data(state, weight), nInputPlane,
         0,
         THCudaTensor_data(state, gradInput_n), outputHeight*outputWidth
      );
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


static int cunnconv1d_LateralConvolution_accGradParameters(lua_State *L) {
   THCState *state = getCutorchState(L);
   THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
   THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");

   float scale = luaL_optnumber(L, 4, 1);
   int nInputPlane = luaT_getfieldcheckint(L, 1, "nInputPlane");
   int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");

   THCudaTensor *gradWeight = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "gradWeight", "torch.CudaTensor");
   THCudaTensor *gradBias = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "gradBias", "torch.CudaTensor");
   THCudaTensor *ones = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "ones", "torch.CudaTensor");

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
   long outputWidth  = inputWidth;

   if (ones->nDimension != 2 || ones->size[0]*ones->size[1] < outputHeight*outputWidth) {
      THCudaTensor_resize2d(state, ones, outputHeight, outputWidth);
      THCudaTensor_fill(state, ones, 1);
   }

   THCudaTensor *input_n = THCudaTensor_new(state);
   THCudaTensor *gradOutput_n = THCudaTensor_new(state);

   for (int elt = 0; elt < batchSize; elt ++) {

      // select each batch
      THCudaTensor_select(state, input_n, input, 0, elt);
      THCudaTensor_select(state, gradOutput_n, gradOutput, 0, elt);

      // convolve
      THCudaBlas_gemm(
         state,
         't', 'n',
         nInputPlane, nOutputPlane, outputHeight*outputWidth,
         scale,
         THCudaTensor_data(state, input_n), inputHeight*inputWidth,
         THCudaTensor_data(state, gradOutput_n), outputHeight*outputWidth,
         1,
         THCudaTensor_data(state, gradWeight), nInputPlane
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

   THCudaTensor_free(state, input_n);
   THCudaTensor_free(state, gradOutput_n);

   if (batch == 0) {
      THCudaTensor_resize3d(state, gradOutput, nOutputPlane, outputHeight, outputWidth);
      THCudaTensor_resize3d(state, input, nInputPlane, inputHeight, inputWidth);
   }

   return 0;
}


static const struct luaL_Reg cunnconv1d_LateralConvolution__ [] = {
   {"LateralConvolution_updateOutput", cunnconv1d_LateralConvolution_updateOutput},
   {"LateralConvolution_updateGradInput", cunnconv1d_LateralConvolution_updateGradInput},
   {"LateralConvolution_accGradParameters", cunnconv1d_LateralConvolution_accGradParameters},
   {NULL, NULL}
};


void cunnconv1d_LateralConvolution_init(lua_State *L)
{
   luaT_pushmetatable(L, "torch.CudaTensor");
   luaT_registeratname(L, cunnconv1d_LateralConvolution__, "nn");
   lua_pop(L,1);
}

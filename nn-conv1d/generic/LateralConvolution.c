#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/LateralConvolution.c"
#else


static int nnconv1d_(LateralConvolution_updateOutput)(lua_State *L)
{
   THTensor *input = luaT_checkudata(L, 2, torch_Tensor);

   int nInputPlane = luaT_getfieldcheckint(L, 1, "nInputPlane");
   int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");

   THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor);
   THTensor *bias = luaT_getfieldcheckudata(L, 1, "bias", torch_Tensor);
   THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);

   luaL_argcheck(L, input->nDimension == 3 ||
                    input->nDimension == 4, 2, "3D or 4D (batch mode) tensor expected");

   // change to batch mode
   int batch = 1;
   if (input->nDimension == 3) {
      batch = 0;
      THTensor_(resize4d)(input, 1, nInputPlane, input->size[1], input->size[2]);
   }

   long batchSize    = input->size[0];
   long inputHeight  = input->size[2];
   long inputWidth   = input->size[3];
   long outputHeight = inputHeight;
   long outputWidth  = inputWidth;

   THTensor_(resize4d)(output, batchSize, nOutputPlane, outputHeight, outputWidth);

   int elt;
#pragma omp parallel for private(elt)
   for (elt = 0; elt < batchSize; elt++) {

      // select each batch in 2D
      THTensor *input_t  = THTensor_(newSelect)(input, 0, elt);
      THTensor *output_t = THTensor_(newSelect)(output, 0, elt);
      THTensor *input2d  = THTensor_(newWithStorage2d)
                              (input_t->storage, input_t->storageOffset,
                               nInputPlane, -1, inputHeight*inputWidth, -1);
      THTensor *output2d = THTensor_(newWithStorage2d)
                              (output_t->storage, output_t->storageOffset,
                               nOutputPlane, -1, outputHeight*outputWidth, -1);

      // fill biases
      int i;
      for (i = 0; i < nOutputPlane; i++)
         THVector_(fill)(output_t->storage->data+output->storageOffset+output_t->stride[0]*i,
                         THTensor_(get1d)(bias, i), outputHeight*outputWidth);

      // convolve
      THTensor_(addmm)(output2d, 1, output2d, 1, weight, input2d);

      // release temp tensors
      THTensor_(free)(input2d);
      THTensor_(free)(output2d);
      THTensor_(free)(input_t);
      THTensor_(free)(output_t);
   }

   // revert to single batch
   if (batch == 0) {
      THTensor_(resize3d)(input, nInputPlane, inputHeight, inputWidth);
      THTensor_(resize3d)(output, nOutputPlane, outputHeight, outputWidth);
   }

   return 1;
}


static int nnconv1d_(LateralConvolution_updateGradInput)(lua_State *L)
{
   THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
   THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);

   int nInputPlane = luaT_getfieldcheckint(L, 1, "nInputPlane");
   int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");

   THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor);
   THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);

   THArgCheck(nOutputPlane == gradOutput->size[input->nDimension == 4 ? 1 : 0], 1,
              "Number of output features is not equal to nOutputPlane" );

   // change to batch mode
   int batch = 1;
   if (input->nDimension == 3) {
      batch = 0;
      THTensor_(resize4d)(input, 1, input->size[0], input->size[1], input->size[2]);
      THTensor_(resize4d)(gradOutput, 1, nOutputPlane, gradOutput->size[1], gradOutput->size[2]);
   }

   long batchSize    = input->size[0];
   long inputWidth   = input->size[3];
   long inputHeight  = input->size[2];
   long outputWidth  = inputWidth;
   long outputHeight = inputHeight;

   THTensor_(resizeAs)(gradInput, input);
   THTensor_(transpose)(weight, weight, 0, 1);

   int elt;
#pragma omp parallel for private(elt)
   for (elt = 0; elt < batchSize; elt++) {

      // select each batch in 2D
      THTensor *gradInput_t  = THTensor_(newSelect)(gradInput, 0, elt);
      THTensor *gradOutput_t = THTensor_(newSelect)(gradOutput, 0, elt);
      THTensor *gradInput2d  = THTensor_(newWithStorage2d)
                                  (gradInput_t->storage, gradInput_t->storageOffset,
                                   nInputPlane, -1, inputWidth*inputHeight, -1);
      THTensor *gradOutput2d = THTensor_(newWithStorage2d)
                                  (gradOutput_t->storage, gradOutput_t->storageOffset,
                                   nOutputPlane, -1, outputWidth*outputHeight, -1);

      // convolve
      THTensor_(addmm)(gradInput2d, 0, gradInput2d, 1, weight, gradOutput2d);

      // release temp tensors
      THTensor_(free)(gradInput2d);
      THTensor_(free)(gradOutput2d);
      THTensor_(free)(gradInput_t);
      THTensor_(free)(gradOutput_t);
   }

   THTensor_(transpose)(weight, weight, 0, 1);

   // revert to single batch
   if (batch == 0) {
      THTensor_(resize3d)(input, nInputPlane, inputHeight, inputWidth);
      THTensor_(resize3d)(gradInput, nInputPlane, inputHeight, inputWidth);
      THTensor_(resize3d)(gradOutput, nOutputPlane, outputHeight, outputWidth);
   }

   return 1;
}


static int nnconv1d_(LateralConvolution_accGradParameters)(lua_State *L)
{
   THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
   THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);
   real scale = luaL_optnumber(L, 4, 1);
   int nInputPlane = luaT_getfieldcheckint(L, 1, "nInputPlane");
   int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");

   THTensor *ones = luaT_getfieldcheckudata(L, 1, "ones", torch_Tensor);
   THTensor *gradWeight = luaT_getfieldcheckudata(L, 1, "gradWeight", torch_Tensor);
   THTensor *gradBias = luaT_getfieldcheckudata(L, 1, "gradBias", torch_Tensor);

   THArgCheck(nOutputPlane == gradOutput->size[input->nDimension == 4 ? 1 : 0], 1,
              "Number of output features is not equal to nOutputPlane" );

   // change to batch mode
   int batch = 1;
   if (input->nDimension == 3) {
      batch = 0;
      THTensor_(resize4d)(input, 1, input->size[0], input->size[1], input->size[2]);
      THTensor_(resize4d)(gradOutput, 1, gradOutput->size[0], gradOutput->size[1], gradOutput->size[2]);
   }

   long batchSize    = input->size[0];
   long inputWidth   = input->size[3];
   long inputHeight  = input->size[2];
   long outputWidth  = inputWidth;
   long outputHeight = inputHeight;

   if (ones->nDimension != 1 || ones->size[0] < outputHeight*outputWidth) {
      THTensor_(resize1d)(ones, outputHeight*outputWidth);
      THTensor_(fill)(ones, 1);
   }

   int elt;
   for (elt = 0; elt < batchSize; elt++) {

      // select each batch in 2D
      THTensor *input_t      = THTensor_(newSelect)(input, 0, elt);
      THTensor *gradOutput_t = THTensor_(newSelect)(gradOutput, 0, elt);
      THTensor *input2d      = THTensor_(newWithStorage2d)
                                  (input_t->storage, input_t->storageOffset,
                                   nInputPlane, -1, inputWidth*inputHeight, -1);
      THTensor *gradOutput2d = THTensor_(newWithStorage2d)(gradOutput->storage, gradOutput->storageOffset,
                                   nOutputPlane, -1, outputWidth*outputHeight, -1);

      // convolve
      THTensor_(transpose)(input2d, input2d, 0, 1);
      THTensor_(addmm)(gradWeight, 1, gradWeight, scale, gradOutput2d, input2d);
      THTensor_(transpose)(input2d, input2d, 0, 1);

      // fill biases
      THTensor_(addmv)(gradBias, 1, gradBias, scale, gradOutput2d, ones);

      THTensor_(free)(input2d);
      THTensor_(free)(gradOutput2d);
      THTensor_(free)(input_t);
      THTensor_(free)(gradOutput_t);
   }

   // revert to single batch
   if (batch == 0) {
      THTensor_(resize3d)(input, nInputPlane, inputHeight, inputWidth);
      THTensor_(resize3d)(gradOutput, nOutputPlane, outputHeight, outputWidth);
   }

   return 0;
}

static const struct luaL_Reg nnconv1d_(LateralConvolution__) [] = {
  {"LateralConvolution_updateOutput", nnconv1d_(LateralConvolution_updateOutput)},
  {"LateralConvolution_updateGradInput", nnconv1d_(LateralConvolution_updateGradInput)},
  {"LateralConvolution_accGradParameters", nnconv1d_(LateralConvolution_accGradParameters)},
  {NULL, NULL}
};

static void nnconv1d_(LateralConvolution_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nnconv1d_(LateralConvolution__), "nn");
  lua_pop(L,1);
}

#endif

#!/usr/bin/env th
--
--  Rank-1 3D filter decomposition test
--
require('torch')
require('nnconv1d')
torch.setdefaulttensortype('torch.FloatTensor')


local function check_error(msg, a, b)
   local diff = torch.add(a, -b):abs()
   print('==> '..msg..' error (max/mean): ', diff:max(), diff:mean())
end

local function compose_filter(z, y, x)
   local zyx = torch.Tensor(z:size(1), z:size(2), y:size(2)*x:size(2))
   for i = 1, z:size(1) do
      local yx = torch.ger(y[i], x[i])
      for j = 1, z:size(2) do
         zyx[i][j]:copy(yx):mul(z[i][j])
      end
   end
   return zyx
end


-- set parameters
local batch = 3
local nInputPlanes = 4
local nOutputPlanes = 5
local iH = 5
local iW = 5
local kW = 3
local kH = 3
local use_cuda = false


-- pick an input
local input = torch.randn(batch, nInputPlanes, iH, iW)

-- get rank-1 filters
local z = torch.randn(nOutputPlanes, nInputPlanes) -- over feature
local y = torch.randn(nOutputPlanes, kH)           -- in vertical
local x = torch.randn(nOutputPlanes, kW)           -- in horizontal
local b = torch.randn(nOutputPlanes)               -- bias

-- reconstruct 3d filter
local zyx = compose_filter(z, y, x)


-- define models
local model_full = nn.Sequential()
model_full:add(nn.SpatialConvolutionMM(nInputPlanes, nOutputPlanes, kW, kH))

local model_low = nn.Sequential()
model_low:add(nn.LateralConvolution(nInputPlanes, nOutputPlanes))
model_low:add(nn.VerticalConvolution(nOutputPlanes, nOutputPlanes, kH))
model_low:add(nn.HorizontalConvolution(nOutputPlanes, nOutputPlanes, kW))


-- overwrite parameters
model_full.modules[1].weight:copy(zyx)
model_full.modules[1].bias:copy(b)

model_low.modules[1].weight:copy(z)
model_low.modules[2].weight:copy(y)
model_low.modules[3].weight:copy(x)
model_low.modules[1].bias:zero()
model_low.modules[2].bias:zero()
model_low.modules[3].bias:copy(b)


-- enable GPU
if use_cuda then
   require('cunnconv1d')
   model_full = model_full:cuda()
   model_low = model_low:cuda()
   input = input:cuda()
end


-- test
local output_full = model_full:updateOutput(input)
local output_low = model_low:updateOutput(input)
check_error('output   ', output_full, output_low)

local gradOutput_full = output_full:clone():add(0.1)
local gradOutput_low = output_low:clone():add(0.1)
local gradInput_full = model_full:updateGradInput(input, gradOutput_full)
local gradInput_low = model_low:updateGradInput(input, gradOutput_low)
check_error('gradInput', gradInput_full, gradInput_low)

model_full:zeroGradParameters()
model_low:zeroGradParameters()
model_full:accGradParameters(input, gradOutput_full, 1)
model_low:accGradParameters(input, gradOutput_low, 1)
local w_full, dw_full = model_full:getParameters()
local w_low,  dw_low = model_low:getParameters()

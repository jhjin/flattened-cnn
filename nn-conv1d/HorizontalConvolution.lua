local HorizontalConvolution, parent = torch.class('nn.HorizontalConvolution', 'nn.Module')

function HorizontalConvolution:__init(nInputPlane, nOutputPlane, kL)
   parent.__init(self)

   assert(nInputPlane == nOutputPlane)
   self.nInputPlane = nInputPlane
   self.nOutputPlane = nOutputPlane

   self.kL = kL

   self.weight = torch.Tensor(nInputPlane, kL)
   self.bias = torch.Tensor(nOutputPlane)
   self.gradWeight = torch.Tensor(nInputPlane, kL)
   self.gradBias = torch.Tensor(nOutputPlane)

   self.ones = torch.Tensor()
   self.finput = torch.Tensor()
   self.fgradWeight = torch.Tensor()

   self:reset()
end

function HorizontalConvolution:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1/math.sqrt(self.nInputPlane)
   end
   if nn.oldSeed then
      self.weight:apply(function()
         return torch.uniform(-stdv, stdv)
      end)
      self.bias:apply(function()
         return torch.uniform(-stdv, stdv)
      end)
   else
      self.weight:uniform(-stdv, stdv)
      self.bias:uniform(-stdv, stdv)
   end
end

local function makeContiguous(self, input, gradOutput)
   if not input:isContiguous() then
      self._input = self._input or input.new()
      self._input:resizeAs(input):copy(input)
      input = self._input
   end
   if gradOutput then
      if not gradOutput:isContiguous() then
         self._gradOutput = self._gradOutput or gradOutput.new()
         self._gradOutput:resizeAs(gradOutput):copy(gradOutput)
         gradOutput = self._gradOutput
      end
   end
   return input, gradOutput
end

function HorizontalConvolution:updateOutput(input)
   input = makeContiguous(self, input)
   return input.nn.HorizontalConvolution_updateOutput(self, input)
end

function HorizontalConvolution:updateGradInput(input, gradOutput)
   if self.gradInput then
      input, gradOutput = makeContiguous(self, input, gradOutput)
      return input.nn.HorizontalConvolution_updateGradInput(self, input, gradOutput)
   end
end

function HorizontalConvolution:accGradParameters(input, gradOutput, scale)
   input, gradOutput = makeContiguous(self, input, gradOutput)
   return input.nn.HorizontalConvolution_accGradParameters(self, input, gradOutput, scale)
end

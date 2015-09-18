## Flattened convolutional neural networks

This package has 1D convolution modules (over channel, in vertical, in horizontal) used in
[Flattened Convolutional Neural Networks for Feedforward Acceleration] (http://arxiv.org/abs/1412.5474)
where we denote the flattened convolution layer as a sequence of one-dimensional filters across all 3D directions.


### Install

Choose both or either of `nn`/`cunn` backend packages depending on your computing environment.

```bash
luarocks install https://raw.githubusercontent.com/jhjin/flattened-cnn/master/nnconv1d-scm-1.rockspec    # cpu
luarocks install https://raw.githubusercontent.com/jhjin/flattened-cnn/master/cunnconv1d-scm-1.rockspec  # cuda
```

or use this command if you already cloned this repo.

```bash
cd nn-conv1d
luarocks make rocks/nnconv1d-scm-1.rockspec
cd ../cunn-conv1d
luarocks make rocks/cunnconv1d-scm-1.rockspec
```


### Available modules

This is a list of available modules.

```lua
nn.LateralConvolution(nInputPlane, nOutputPlane)        -- 1d conv over feature
nn.HorizontalConvolution(nInputPlane, nOutputPlane, kL) -- 1d conv in horizontal
nn.VerticalConvolution(nInputPlane, nOutputPlane, kL)   -- 1d conv in vertical
```


### Example

Run the command below.

```bash
th example.lua
```

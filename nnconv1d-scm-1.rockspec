package = "nnconv1d"
version = "scm-1"

source = {
   url = "git://github.com/jhjin/nn-conv1d.git",
}

description = {
   summary = "1D Convolutions for Torch nn",
   detailed = [[
   ]],
   homepage = "https://github.com/jhjin/nn-conv1d",
   license = "MIT"
}

dependencies = {
   "torch >= 7.0",
}

build = {
   type = "command",
   build_command = [[
cd nn-conv1d && cmake -E make_directory build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)" && $(MAKE)
]],
   install_command = "cd nn-conv1d && cd build && $(MAKE) install"
}

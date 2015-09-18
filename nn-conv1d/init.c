#include "TH.h"
#include "luaT.h"

#define torch_(NAME) TH_CONCAT_3(torch_, Real, NAME)
#define torch_Tensor TH_CONCAT_STRING_3(torch.,Real,Tensor)
#define nnconv1d_(NAME) TH_CONCAT_3(nnconv1d_, Real, NAME)

#include "generic/LateralConvolution.c"
#include "THGenerateFloatTypes.h"

#include "generic/VerticalConvolution.c"
#include "THGenerateFloatTypes.h"

#include "generic/HorizontalConvolution.c"
#include "THGenerateFloatTypes.h"

LUA_EXTERNC DLL_EXPORT int luaopen_libnnconv1d(lua_State *L);

int luaopen_libnnconv1d(lua_State *L)
{
  lua_newtable(L);

  nnconv1d_FloatLateralConvolution_init(L);
  nnconv1d_FloatVerticalConvolution_init(L);
  nnconv1d_FloatHorizontalConvolution_init(L);

  nnconv1d_DoubleLateralConvolution_init(L);
  nnconv1d_DoubleVerticalConvolution_init(L);
  nnconv1d_DoubleHorizontalConvolution_init(L);

  return 1;
}

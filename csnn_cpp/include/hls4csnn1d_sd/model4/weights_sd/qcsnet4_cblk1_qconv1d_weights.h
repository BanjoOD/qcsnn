#ifndef QCSNET4_CBLK1_QCONV1D_WEIGHTS_H
#define QCSNET4_CBLK1_QCONV1D_WEIGHTS_H

#include <ap_int.h>

namespace hls4csnn1d_cblk_sd {

const int qcsnet4_cblk1_qconv1d_OUT_CH = 16;
const int qcsnet4_cblk1_qconv1d_IN_CH  = 1;
const int qcsnet4_cblk1_qconv1d_KERNEL_SIZE = 3;
const int qcsnet4_cblk1_qconv1d_STRIDE = 1;

const ap_int<8> qcsnet4_cblk1_qconv1d_input_zero_point = 0;

const ap_int<32> qcsnet4_cblk1_qconv1d_scale_multiplier[16] = {
  1340767788, 1340767788, 1340767788, 1340767788, 1340767788, 1340767788, 1340767788, 1340767788, 1340767788, 1340767788, 1340767788, 1340767788, 1340767788, 1340767788, 1340767788, 1340767788
};

const int qcsnet4_cblk1_qconv1d_right_shift[16] = {
  38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38
};

const acc32_t qcsnet4_cblk1_qconv1d_bias[16] = {
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

const acc32_t qcsnet4_cblk1_qconv1d_weight_sum[16] = {
  109, 84, 135, 18, 166, 113, -95, 18, -162, -183, 57, 43, 108, -58, 217, 51
};

const ap_int<8> qcsnet4_cblk1_qconv1d_weights[16][1][3] = {
{
  { 66, 79, -36 }
},
{
  { 105, -31, 10 }
},
{
  { -60, 75, 120 }
},
{
  { -97, 92, 23 }
},
{
  { 90, 18, 58 }
},
{
  { -16, 96, 33 }
},
{
  { -57, 22, -60 }
},
{
  { -20, -45, 83 }
},
{
  { -83, -49, -30 }
},
{
  { -61, 5, -127 }
},
{
  { 102, -109, 64 }
},
{
  { 17, -40, 66 }
},
{
  { 11, 88, 9 }
},
{
  { -42, 23, -39 }
},
{
  { 43, 102, 72 }
},
{
  { -45, 72, 24 }
}
};

} // namespace
#endif // QCSNET4_CBLK1_QCONV1D_WEIGHTS_H

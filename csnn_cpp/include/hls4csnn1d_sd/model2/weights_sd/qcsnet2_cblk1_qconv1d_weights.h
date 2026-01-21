#ifndef QCSNET2_CBLK1_QCONV1D_WEIGHTS_H
#define QCSNET2_CBLK1_QCONV1D_WEIGHTS_H

#include <ap_int.h>

namespace hls4csnn1d_cblk_sd {

const int qcsnet2_cblk1_qconv1d_OUT_CH = 16;
const int qcsnet2_cblk1_qconv1d_IN_CH  = 1;
const int qcsnet2_cblk1_qconv1d_KERNEL_SIZE = 3;
const int qcsnet2_cblk1_qconv1d_STRIDE = 1;

const ap_int<8> qcsnet2_cblk1_qconv1d_input_zero_point = 0;

const ap_int<32> qcsnet2_cblk1_qconv1d_scale_multiplier[16] = {
  1800165518, 1800165518, 1800165518, 1800165518, 1800165518, 1800165518, 1800165518, 1800165518, 1800165518, 1800165518, 1800165518, 1800165518, 1800165518, 1800165518, 1800165518, 1800165518
};

const int qcsnet2_cblk1_qconv1d_right_shift[16] = {
  38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38
};

const acc32_t qcsnet2_cblk1_qconv1d_bias[16] = {
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

const acc32_t qcsnet2_cblk1_qconv1d_weight_sum[16] = {
  180, 107, 71, 20, 186, 102, -91, -2, -158, -214, 71, 36, 89, -50, 280, 38
};

const ap_int<8> qcsnet2_cblk1_qconv1d_weights[16][1][3] = {
{
  { 115, 97, -32 }
},
{
  { 103, -15, 19 }
},
{
  { -58, 49, 80 }
},
{
  { -80, 82, 18 }
},
{
  { 95, 33, 58 }
},
{
  { -17, 86, 33 }
},
{
  { -54, 17, -54 }
},
{
  { -29, -45, 72 }
},
{
  { -80, -49, -29 }
},
{
  { -69, -18, -127 }
},
{
  { 98, -88, 61 }
},
{
  { 12, -35, 59 }
},
{
  { 10, 75, 4 }
},
{
  { -31, 20, -39 }
},
{
  { 58, 125, 97 }
},
{
  { -44, 60, 22 }
}
};

} // namespace
#endif // QCSNET2_CBLK1_QCONV1D_WEIGHTS_H

#ifndef QPARAMS_QCSNET4_CBLK1_INPUT_H
#define QPARAMS_QCSNET4_CBLK1_INPUT_H

#include <ap_int.h>

namespace hls4csnn1d_cblk_sd {

// Activation quantization parameters (optional for kernels)
const int   qcsnet4_cblk1_input_bit_width = 8;
const float qcsnet4_cblk1_input_scale     = 0.03901852295;  // kept for reference only
const ap_int<16> qcsnet4_cblk1_input_act_scale_int = 160;
const int   qcsnet4_cblk1_input_zero_point= 0;

} // namespace
#endif // QPARAMS_QCSNET4_CBLK1_INPUT_H

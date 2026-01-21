#ifndef QPARAMS_QCSNET4_CBLK2_INPUT_H
#define QPARAMS_QCSNET4_CBLK2_INPUT_H

#include <ap_int.h>

namespace hls4csnn1d_cblk_sd {

// Activation quantization parameters (optional for kernels)
const int   qcsnet4_cblk2_input_bit_width = 8;
// const float qcsnet4_cblk2_input_scale     = 0.007874004543;  // kept for reference only
const ap_int<16> qcsnet4_cblk2_input_act_scale_int = 32;
const int   qcsnet4_cblk2_input_zero_point= 0;

} // namespace
#endif // QPARAMS_QCSNET4_CBLK2_INPUT_H

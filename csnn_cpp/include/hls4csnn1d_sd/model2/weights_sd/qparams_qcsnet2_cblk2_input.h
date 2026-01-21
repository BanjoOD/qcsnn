#ifndef QPARAMS_QCSNET2_CBLK2_INPUT_H
#define QPARAMS_QCSNET2_CBLK2_INPUT_H

#include <ap_int.h>

namespace hls4csnn1d_cblk_sd {

// Activation quantization parameters (optional for kernels)
const int   qcsnet2_cblk2_input_bit_width = 8;
// const float qcsnet2_cblk2_input_scale     = 0.00748090772;  // kept for reference only
const ap_int<16> qcsnet2_cblk2_input_act_scale_int = 31;
const int   qcsnet2_cblk2_input_zero_point= 0;

} // namespace
#endif // QPARAMS_QCSNET2_CBLK2_INPUT_H

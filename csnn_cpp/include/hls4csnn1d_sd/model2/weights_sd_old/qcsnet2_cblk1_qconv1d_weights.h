#ifndef QCSNET2_CBLK1_QCONV1D_WEIGHTS_H
#define QCSNET2_CBLK1_QCONV1D_WEIGHTS_H

#include <ap_fixed.h>
typedef ap_fixed<8, 4, AP_RND, AP_SAT> ap_fixed_c;

namespace hls4csnn1d_cblk_sd {

const int qcsnet2_cblk1_qconv1d_IN_CH  = 1;
const int qcsnet2_cblk1_qconv1d_OUT_CH = 16;
const int qcsnet2_cblk1_qconv1d_K      = 3;

static const ap_fixed_c qcsnet2_cblk1_qconv1d_weights[16][1][3] = { 0.5625, 0.9375, 0.6875, 7.3125, -3.1875, -4.125, 1.875, 0.125, -0.3125, 2.5, -2.0625, 1.3125, 2.125, -2.625, -1.125, -3.9375, -1.8125, 4.5, 1.75, 2.5625, -3, 0.5625, 0.8125, 0.25, 1.25, -0.5625, 1.625, -0.875, -0.8125, 0.0625, 0.625, 0.5625, 0.5, 2.3125, 1.875, -2.125, -5.1875, 0.9375, 5.4375, -3.1875, 3.8125, 0.9375, 2.0625, 0, -0.125, 0.75, -1.125, -1.5625 };

} // namespace hls4csnn1d_cblk_sd

#endif // QCSNET2_CBLK1_QCONV1D_WEIGHTS_H

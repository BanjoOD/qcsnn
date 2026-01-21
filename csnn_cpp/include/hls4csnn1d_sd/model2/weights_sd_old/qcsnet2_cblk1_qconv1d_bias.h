#ifndef QCSNET2_CBLK1_QCONV1D_BIAS_H
#define QCSNET2_CBLK1_QCONV1D_BIAS_H

#include <ap_fixed.h>
typedef ap_fixed<8, 4, AP_RND, AP_SAT> ap_fixed_c;

namespace hls4csnn1d_cblk_sd {

static const ap_fixed_c qcsnet2_cblk1_qconv1d_bias[16] = { -0.875, -0.125, -0.6875, -0.75, 0.5, 0.375, -0.625, -0.75, -0.75, 0.5625, -0.8125, -0.8125, -0.5, -0.625, -0.75, 0.75 };

} // namespace hls4csnn1d_cblk_sd

#endif // QCSNET2_CBLK1_QCONV1D_BIAS_H

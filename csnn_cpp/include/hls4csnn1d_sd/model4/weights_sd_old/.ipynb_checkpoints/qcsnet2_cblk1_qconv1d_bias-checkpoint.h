#ifndef QCSNET2_CBLK1_QCONV1D_BIAS_H
#define QCSNET2_CBLK1_QCONV1D_BIAS_H

#include <ap_fixed.h>
typedef ap_fixed<8, 4, AP_RND, AP_SAT> ap_fixed_c;

namespace hls4csnn1d_cblk_sd {

static const ap_fixed_c qcsnet2_cblk1_qconv1d_bias[16] = { -1.0625, -1.0625, 0.375, -1.3125, -1.1875, 1.1875, 1.4375, 0.6875, 0.0625, -1.3125, 0.1875, 0.6875, -1.1875, 0.8125, 0.625, -1.25 };
} // namespace

#endif

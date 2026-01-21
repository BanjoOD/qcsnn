#ifndef QCSNET2_CBLK2_QCONV1D_BIAS_H
#define QCSNET2_CBLK2_QCONV1D_BIAS_H

#include <ap_fixed.h>
typedef ap_fixed<8, 4, AP_RND, AP_SAT> ap_fixed_c;

namespace hls4csnn1d_cblk_sd {

static const ap_fixed_c qcsnet2_cblk2_qconv1d_bias[24] = { -0.625, -1.125, 0.125, 0.4375, 0.0625, -1, -1.25, 0.0625, -0.1875, -1.0625, -0.5625, -0.125, -0.3125, -0.3125, -1.3125, 0.5, -0.25, -1.25, 0.75, -0.3125, -0.4375, -1.375, 1.875, -0.25 };

} // namespace hls4csnn1d_cblk_sd

#endif // QCSNET2_CBLK2_QCONV1D_BIAS_H

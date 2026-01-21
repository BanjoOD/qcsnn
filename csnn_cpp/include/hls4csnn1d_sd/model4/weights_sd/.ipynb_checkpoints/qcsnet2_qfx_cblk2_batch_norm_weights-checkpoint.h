#ifndef QCSNET2_QFX_CBLK2_BATCH_NORM_WEIGHTS_H
#define QCSNET2_QFX_CBLK2_BATCH_NORM_WEIGHTS_H

#include <ap_fixed.h>
typedef ap_fixed<8, 4, AP_RND, AP_SAT> ap_fixed_c;

namespace hls4csnn1d_cblk_sd_train {

const int qcsnet2_qfx_cblk2_batch_norm_CH = 24;

static const ap_fixed_c qcsnet2_qfx_cblk2_batch_norm_ALPHA[24] = { 7.9375, 7.9375, 7.9375, 7.9375, 7.9375, 7.9375, 7.9375, 7.9375, 7.9375, 7.9375, 7.9375, 7.9375, 7.9375, 7.9375, 7.9375, 7.9375, 7.9375, 7.9375, 7.9375, 7.9375, 7.9375, 7.9375, 7.9375, 7.9375 };

static const ap_fixed_c qcsnet2_qfx_cblk2_batch_norm_ETA[24]   = { 5.875, 7.9375, 4.8125, -1.8125, -1.625, -3.875, 7.9375, 7.9375, -3.125, 4.1875, -0.625, 5.875, 4.375, -1.125, 6, -2.6875, -3.375, -1.0625, 3.3125, -1.625, 2.6875, 1.875, -0.8125, -4.1875 };

} // namespace hls4csnn1d_cblk_sd_train

#endif // QCSNET2_QFX_CBLK2_BATCH_NORM_WEIGHTS_H

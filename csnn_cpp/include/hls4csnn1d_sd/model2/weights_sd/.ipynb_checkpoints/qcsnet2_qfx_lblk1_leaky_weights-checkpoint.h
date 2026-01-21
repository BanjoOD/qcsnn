#ifndef QCSNET2_QFX_LBLK1_LEAKY_WEIGHTS_H
#define QCSNET2_QFX_LBLK1_LEAKY_WEIGHTS_H

#include <ap_fixed.h>
typedef ap_fixed<8, 4, AP_RND, AP_SAT> ap_fixed_layer_t;

namespace hls4csnn1d_cblk_sd_train {

const int qcsnet2_qfx_lblk1_leaky_ARRAY_POSITION = 5;

const int qcsnet2_qfx_lblk1_leaky_CH = 2;

static const ap_fixed_layer_t qcsnet2_qfx_lblk1_leaky_leaky_beta[2]      = { -0.004707828164, -0.0002838242799 };
static const ap_fixed_layer_t qcsnet2_qfx_lblk1_leaky_leaky_threshold[2] = { 1.200227022, 0.8908810019 };

} // namespace hls4csnn1d_cblk_sd_train

#endif // QCSNET2_QFX_LBLK1_LEAKY_WEIGHTS_H

#ifndef QCSNET2_QFX_CBLK1_LEAKY_WEIGHTS_H
#define QCSNET2_QFX_CBLK1_LEAKY_WEIGHTS_H

#include <ap_fixed.h>
typedef ap_fixed<8, 4, AP_RND, AP_SAT> ap_fixed_layer_t;

namespace hls4csnn1d_cblk_sd_train {

const int qcsnet2_qfx_cblk1_leaky_ARRAY_POSITION = 0;

const int qcsnet2_qfx_cblk1_leaky_CH = 16;

static const ap_fixed_layer_t qcsnet2_qfx_cblk1_leaky_leaky_beta[16]      = { 0.5404960513, -0.3199403286, -0.05533212051, 0.3077687025, 0.151379317, 0.7390828729, 0.8901569247, -0.160549894, 0.5919830799, -0.4986776412, -0.2910797298, -0.02403385751, 0.3349105716, 1.033800721, 0.5311086178, 0.8052558303 };
static const ap_fixed_layer_t qcsnet2_qfx_cblk1_leaky_leaky_threshold[16] = { 0.7939206958, 0.8644798398, 1.24006772, 1.657785892, 1.542571902, 1.026837826, 0.3423597217, 1.006223679, 0.5699458122, 0.7121350765, 0.125513956, 0.2062557191, 0.7289070487, 0.3567063212, 1.142229199, 0.07066887617 };

} // namespace hls4csnn1d_cblk_sd_train

#endif // QCSNET2_QFX_CBLK1_LEAKY_WEIGHTS_H

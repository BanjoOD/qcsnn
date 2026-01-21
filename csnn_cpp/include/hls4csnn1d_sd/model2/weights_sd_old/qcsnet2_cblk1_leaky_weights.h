#ifndef QCSNET2_CBLK1_LEAKY_WEIGHTS_H
#define QCSNET2_CBLK1_LEAKY_WEIGHTS_H

#include <ap_fixed.h>
typedef ap_fixed<8, 4, AP_RND, AP_SAT> ap_fixed_c;

namespace hls4csnn1d_cblk_sd {

static const ap_fixed_c qcsnet2_cblk1_leaky_beta      = 0.500000;
static const ap_fixed_c qcsnet2_cblk1_leaky_threshold = 0.437500;

} // namespace hls4csnn1d_cblk_sd

#endif // QCSNET2_CBLK1_LEAKY_WEIGHTS_H

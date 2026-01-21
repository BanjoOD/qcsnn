#ifndef QCSNET2_CBLK2_LEAKY_WEIGHTS_H
#define QCSNET2_CBLK2_LEAKY_WEIGHTS_H

#include <ap_fixed.h>
typedef ap_fixed<8, 4, AP_RND, AP_SAT> ap_fixed_c;

namespace hls4csnn1d_cblk_sd {

static const ap_fixed_c qcsnet2_cblk2_leaky_beta      = 1.000000;
static const ap_fixed_c qcsnet2_cblk2_leaky_threshold = 0.312500;

} // namespace hls4csnn1d_cblk_sd

#endif // QCSNET2_CBLK2_LEAKY_WEIGHTS_H

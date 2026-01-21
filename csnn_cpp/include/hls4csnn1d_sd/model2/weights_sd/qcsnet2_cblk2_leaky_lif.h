#ifndef QCSNET2_CBLK2_LEAKY_LIF_H
#define QCSNET2_CBLK2_LEAKY_LIF_H

#include <hls_stream.h>
#include <ap_int.h>
#include "../constants_sd.h"

namespace hls4csnn1d_cblk_sd {

enum { qcsnet2_cblk2_leaky_FRAC_BITS = 12 };
const ap_int<16> qcsnet2_cblk2_leaky_beta_int   = 3961;
const ap_int<16> qcsnet2_cblk2_leaky_theta_int  = 505;
const ap_int<16> qcsnet2_cblk2_leaky_scale_int  = 31;

} // namespace hls4csnn1d_cblk_sd
#endif // QCSNET2_CBLK2_LEAKY_LIF_H

#ifndef QCSNET2_CBLK1_LEAKY_LIF_H
#define QCSNET2_CBLK1_LEAKY_LIF_H

#include <hls_stream.h>
#include <ap_int.h>
#include "../constants_sd.h"

namespace hls4csnn1d_cblk_sd {

enum { qcsnet2_cblk1_leaky_FRAC_BITS = 12 };
const ap_int<16> qcsnet2_cblk1_leaky_beta_int   = 1657;
const ap_int<16> qcsnet2_cblk1_leaky_theta_int  = 1048;
const ap_int<16> qcsnet2_cblk1_leaky_scale_int  = 158;

} // namespace hls4csnn1d_cblk_sd
#endif // QCSNET2_CBLK1_LEAKY_LIF_H

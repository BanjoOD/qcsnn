#ifndef QCSNET4_CBLK1_LEAKY_LIF_H
#define QCSNET4_CBLK1_LEAKY_LIF_H

#include <hls_stream.h>
#include <ap_int.h>
#include "../constants4_sd.h"

namespace hls4csnn1d_cblk_sd {

enum { qcsnet4_cblk1_leaky_FRAC_BITS = 12 };
const ap_int<16> qcsnet4_cblk1_leaky_beta_int   = 1777;
const ap_int<16> qcsnet4_cblk1_leaky_theta_int  = 1803;
const ap_int<16> qcsnet4_cblk1_leaky_scale_int  = 164;

} // namespace hls4csnn1d_cblk_sd
#endif // QCSNET4_CBLK1_LEAKY_LIF_H

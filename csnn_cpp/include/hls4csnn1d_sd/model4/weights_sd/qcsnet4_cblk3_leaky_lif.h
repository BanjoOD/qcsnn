#ifndef QCSNET4_CBLK3_LEAKY_LIF_H
#define QCSNET4_CBLK3_LEAKY_LIF_H

#include <hls_stream.h>
#include <ap_int.h>
#include "../constants4_sd.h"

namespace hls4csnn1d_cblk_sd {

enum { qcsnet4_cblk3_leaky_FRAC_BITS = 12 };
const ap_int<16> qcsnet4_cblk3_leaky_beta_int   = 4097;
const ap_int<16> qcsnet4_cblk3_leaky_theta_int  = 613;
const ap_int<16> qcsnet4_cblk3_leaky_scale_int  = 21;

} // namespace hls4csnn1d_cblk_sd
#endif // QCSNET4_CBLK3_LEAKY_LIF_H

#ifndef QCSNET4_LBLK2_LEAKY_LIF_H
#define QCSNET4_LBLK2_LEAKY_LIF_H

#include <hls_stream.h>
#include <ap_int.h>
#include "../constants4_sd.h"

namespace hls4csnn1d_cblk_sd {

enum { qcsnet4_lblk2_leaky_FRAC_BITS = 12 };
const ap_int<16> qcsnet4_lblk2_leaky_beta_int   = 4097;
const ap_int<16> qcsnet4_lblk2_leaky_theta_int  = -83;
const ap_int<16> qcsnet4_lblk2_leaky_scale_int  = 154;

} // namespace hls4csnn1d_cblk_sd
#endif // QCSNET4_LBLK2_LEAKY_LIF_H

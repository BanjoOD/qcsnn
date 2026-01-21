#ifndef QCSNET2_LBLK1_LEAKY_LIF_H
#define QCSNET2_LBLK1_LEAKY_LIF_H

#include <hls_stream.h>
#include <ap_int.h>
#include "../constants_sd.h"

namespace hls4csnn1d_cblk_sd {

enum { qcsnet2_lblk1_leaky_FRAC_BITS = 12 };
const ap_int<16> qcsnet2_lblk1_leaky_beta_int   = 4097;
const ap_int<16> qcsnet2_lblk1_leaky_theta_int  = 112;
const ap_int<16> qcsnet2_lblk1_leaky_scale_int  = 81;

} // namespace hls4csnn1d_cblk_sd
#endif // QCSNET2_LBLK1_LEAKY_LIF_H

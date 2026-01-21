#ifndef QCSNET4_CBLK2_BATCH_NORM_BN_H
#define QCSNET4_CBLK2_BATCH_NORM_BN_H

#include <hls_stream.h>
#include <ap_int.h>
#include "../constants4_sd.h"

namespace hls4csnn1d_cblk_sd {

const int qcsnet4_cblk2_batch_norm_C = 16;

const ap_int8_c qcsnet4_cblk2_batch_norm_weight[16] = {
  119, 127, 116, 122, 119, 127, 127, 121, 124, 120, 123, 119, 124, 121, 120, 119
};

const ap_int<32> qcsnet4_cblk2_batch_norm_bias[16] = {
  1131, 2202, -1025, -882, -855, -23, 1937, -872, -403, 352, -981, -506, 388, -1612, -283, -343
};

const ap_int<32> qcsnet4_cblk2_batch_norm_scale_multiplier[16] = {
  1201780458, 1201780458, 1201780458, 1201780458, 1201780458, 1201780458, 1201780458, 1201780458, 1201780458, 1201780458, 1201780458, 1201780458, 1201780458, 1201780458, 1201780458, 1201780458
};

const int qcsnet4_cblk2_batch_norm_right_shift[16] = {
  37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37
};

} // namespace hls4csnn1d_cblk_sd
#endif // QCSNET4_CBLK2_BATCH_NORM_BN_H

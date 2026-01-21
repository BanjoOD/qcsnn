#ifndef QCSNET4_CBLK1_BATCH_NORM_BN_H
#define QCSNET4_CBLK1_BATCH_NORM_BN_H

#include <hls_stream.h>
#include <ap_int.h>
#include "../constants4_sd.h"

namespace hls4csnn1d_cblk_sd {

const int qcsnet4_cblk1_batch_norm_C = 16;

const ap_int8_c qcsnet4_cblk1_batch_norm_weight[16] = {
  104, 115, 126, 127, 117, 121, 116, 127, 112, 118, 111, 112, 112, 119, 116, 118
};

const ap_int<32> qcsnet4_cblk1_batch_norm_bias[16] = {
  -258, -218, 616, 96, -92, -92, -308, -63, -589, -102, -360, -241, -254, -65, -18, -204
};

const ap_int<32> qcsnet4_cblk1_batch_norm_scale_multiplier[16] = {
  1204820817, 1204820817, 1204820817, 1204820817, 1204820817, 1204820817, 1204820817, 1204820817, 1204820817, 1204820817, 1204820817, 1204820817, 1204820817, 1204820817, 1204820817, 1204820817
};

const int qcsnet4_cblk1_batch_norm_right_shift[16] = {
  37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37
};

} // namespace hls4csnn1d_cblk_sd
#endif // QCSNET4_CBLK1_BATCH_NORM_BN_H

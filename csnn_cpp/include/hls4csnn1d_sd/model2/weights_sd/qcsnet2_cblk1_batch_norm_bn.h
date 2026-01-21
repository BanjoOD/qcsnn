#ifndef QCSNET2_CBLK1_BATCH_NORM_BN_H
#define QCSNET2_CBLK1_BATCH_NORM_BN_H

#include <hls_stream.h>
#include <ap_int.h>
#include "../constants_sd.h"

namespace hls4csnn1d_cblk_sd {

const int qcsnet2_cblk1_batch_norm_C = 16;

const ap_int8_c qcsnet2_cblk1_batch_norm_weight[16] = {
  121, 107, 98, 103, 112, 110, 105, 127, 102, 115, 102, 99, 98, 104, 123, 105
};

const ap_int<32> qcsnet2_cblk1_batch_norm_bias[16] = {
  254, -158, -401, -155, -69, -65, -316, 121, -450, -7, -105, -281, -408, -328, 643, -109
};

const ap_int<32> qcsnet2_cblk1_batch_norm_scale_multiplier[16] = {
  1277679275, 1277679275, 1277679275, 1277679275, 1277679275, 1277679275, 1277679275, 1277679275, 1277679275, 1277679275, 1277679275, 1277679275, 1277679275, 1277679275, 1277679275, 1277679275
};

const int qcsnet2_cblk1_batch_norm_right_shift[16] = {
  37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37
};

} // namespace hls4csnn1d_cblk_sd
#endif // QCSNET2_CBLK1_BATCH_NORM_BN_H

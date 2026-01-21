#ifndef QCSNET2_CBLK2_BATCH_NORM_BN_H
#define QCSNET2_CBLK2_BATCH_NORM_BN_H

#include <hls_stream.h>
#include <ap_int.h>
#include "../constants_sd.h"

namespace hls4csnn1d_cblk_sd {

const int qcsnet2_cblk2_batch_norm_C = 24;

const ap_int8_c qcsnet2_cblk2_batch_norm_weight[24] = {
  117, 113, 97, 106, 108, 103, 102, 99, 116, 107, 102, 101, 101, 104, 112, 102,
    107, 104, 113, 127, 114, 98, 103, 111
};

const ap_int<32> qcsnet2_cblk2_batch_norm_bias[24] = {
  -6, 46, -237, -414, -343, -1569, -458, -1388, 693, -1216, -1546, -1290, -1060, -1795, -222, -1002,
    -423, -1126, 174, 291, -325, -1530, -1296, -1528
};

const ap_int<32> qcsnet2_cblk2_batch_norm_scale_multiplier[24] = {
  1237387393, 1237387393, 1237387393, 1237387393, 1237387393, 1237387393, 1237387393, 1237387393, 1237387393, 1237387393, 1237387393, 1237387393, 1237387393, 1237387393, 1237387393, 1237387393,
    1237387393, 1237387393, 1237387393, 1237387393, 1237387393, 1237387393, 1237387393, 1237387393
};

const int qcsnet2_cblk2_batch_norm_right_shift[24] = {
  37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37,
    37, 37, 37, 37, 37, 37, 37, 37
};

} // namespace hls4csnn1d_cblk_sd
#endif // QCSNET2_CBLK2_BATCH_NORM_BN_H

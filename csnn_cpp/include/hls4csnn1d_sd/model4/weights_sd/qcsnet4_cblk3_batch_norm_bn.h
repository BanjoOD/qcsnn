#ifndef QCSNET4_CBLK3_BATCH_NORM_BN_H
#define QCSNET4_CBLK3_BATCH_NORM_BN_H

#include <hls_stream.h>
#include <ap_int.h>
#include "../constants4_sd.h"

namespace hls4csnn1d_cblk_sd {

const int qcsnet4_cblk3_batch_norm_C = 24;

const ap_int8_c qcsnet4_cblk3_batch_norm_weight[24] = {
  105, 105, 126, 111, 117, 110, 123, 123, 102, 109, 109, 108, 110, 115, 100, 113,
    107, 115, 114, 112, 127, 107, 110, 112
};

const ap_int<32> qcsnet4_cblk3_batch_norm_bias[24] = {
  -1636, -2287, 49, -803, -451, -1679, 270, -953, -2422, -2685, -925, -1375, -1798, -968, -2327, -2050,
    -2805, -311, -2119, -695, 49, -1521, -1934, -594
};

const ap_int<32> qcsnet4_cblk3_batch_norm_scale_multiplier[24] = {
  1322243749, 1322243749, 1322243749, 1322243749, 1322243749, 1322243749, 1322243749, 1322243749, 1322243749, 1322243749, 1322243749, 1322243749, 1322243749, 1322243749, 1322243749, 1322243749,
    1322243749, 1322243749, 1322243749, 1322243749, 1322243749, 1322243749, 1322243749, 1322243749
};

const int qcsnet4_cblk3_batch_norm_right_shift[24] = {
  37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37,
    37, 37, 37, 37, 37, 37, 37, 37
};

} // namespace hls4csnn1d_cblk_sd
#endif // QCSNET4_CBLK3_BATCH_NORM_BN_H

#ifndef QCSNET2_QFX_CBLK1_BATCH_NORM_WEIGHTS_H
#define QCSNET2_QFX_CBLK1_BATCH_NORM_WEIGHTS_H

#include "../cblk_sd_train/bn_struct_sd_train.h"

namespace hls4csnn1d_cblk_sd_train {

const int qcsnet2_qfx_cblk1_batch_norm_CH  = 16;    

// Position on array
const int qcsnet2_qfx_cblk1_batch_norm_ARRAY_POSITION = 0;

static const BNParams<16> qcsnet2_qfx_cblk1_batch_norm_PARAMS = {

    // Gamma
    { 0.875, 2.5, 3.4375, 0.75, 2.5625, 0.9375, 0.125, 2.3125, 1.0625, 3.3125, 2.875, 3.3125, 1.1875, -0, 2.3125, -0 },

    // Beta
    { 0.4375, 0.625, -0.4375, 0.1875, -0.25, 0.6875, 0.3125, -0.1875, 0.0625, 0.125, 0.25, -0.6875, 0.25, 0.1875, -0.3125, -0.0625 },

    // Running mean
    { 0, 0, -0.0625, 0.125, -0.0625, -0, -0, 0.0625, -0.0625, -0.1875, 0, -0, 0, -0.125, 0, 0 }, 
    
    // Running var
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } 
    
};

} // namespace hls4csnn1d_cblk_sd_train

#endif // QCSNET2_QFX_CBLK1_BATCH_NORM_WEIGHTS_H

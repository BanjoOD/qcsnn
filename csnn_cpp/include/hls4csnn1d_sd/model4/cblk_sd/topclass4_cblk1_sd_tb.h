#ifndef TOP_CLASS4_CBLK1_TB_H
#define TOP_CLASS4_CBLK1_TB_H


#include "../constants4_sd.h"
extern "C" {
    void topFunctionCblk1(hls::stream<axi_fixed_t> &dataStream, hls::stream<axi_fixed_t> &labelStream);
}
#endif


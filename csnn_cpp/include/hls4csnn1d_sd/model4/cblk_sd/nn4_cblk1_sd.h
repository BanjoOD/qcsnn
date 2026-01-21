#ifndef NEURAL_NETWORK4_CBLK1_SD_H
#define NEURAL_NETWORK4_CBLK1_SD_H

#include <hls_stream.h>
#include <ap_fixed.h>
#include "../constants4_sd.h"
#include "includeheaders4_sd.h"

/* ENABLE debug prints for *this* translation unit */
#define PRINT_FIRST_CONV_ROW
#include "conv1d_sd.h"          // ← debug code compiled in
#undef  PRINT_FIRST_CONV_ROW    // ← other files stay silent

#include "batchnorm1d_sd.h"
#include "lif1d_integer.h"
#include "maxpool1d_sd.h"
#include "linear1d_sd.h"
#include "quantidentity1d_sd.h"
#include "../utils_sd.h"          

namespace hls4csnn1d_cblk_sd {

/* ─────────────────────────────────────────────────────────────── */
template<int NUM_STEPS>
class NeuralNetwork4_Cblk1_sd {
public:
    NeuralNetwork4_Cblk1_sd()  = default;
    ~NeuralNetwork4_Cblk1_sd() = default;

/* ================================================================
 *  1.  INFERENCE FORWARD
 * ================================================================ */
    void forward(hls::stream<ap_int8_c>& input_stream,
                 hls::stream<ap_int8_c>& output_stream) {
#pragma HLS DATAFLOW
#pragma HLS INTERFACE axis      port=input_stream
#pragma HLS INTERFACE axis      port=output_stream
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL

        // --- 0) Buffer ONE record so we can replay it T times ---
        ap_int8_c in_buf[CONV_IN_LENGTH1];
#pragma HLS ARRAY_PARTITION variable=in_buf cyclic factor=8 dim=1
        for (int k = 0; k < CONV_IN_LENGTH1; ++k) {
#pragma HLS PIPELINE II=1
            in_buf[k] = input_stream.read();
        }

        // ───── Internal streams ─────────────────────────────────
        hls::stream<ap_int8_c> s0("conv_to_bn_cblk1");     // conv1 -> bn1
#pragma HLS STREAM variable=s0 depth=FIXED_LENGTH2

        hls::stream<ap_int8_c> s1("bn_to_lif_cblk1");      // bn1 -> lif1
#pragma HLS STREAM variable=s1 depth=FIXED_LENGTH2

        hls::stream<ap_int8_c> s2("lif_to_mp_cblk1");      // lif1 -> mp1
#pragma HLS STREAM variable=s2 depth=FIXED_LENGTH2

        hls::stream<ap_int8_c> s3("mp_to_qi_cblk2");       // mp1 -> qi(cblk2)
#pragma HLS STREAM variable=s3 depth=FIXED_LENGTH3

        hls::stream<ap_int8_c> s4("qi_to_conv_cblk2");     // qi -> conv2
#pragma HLS STREAM variable=s4 depth=FIXED_LENGTH3

        hls::stream<ap_int8_c> s5("conv_to_bn_cblk2");     // conv2 -> bn2
#pragma HLS STREAM variable=s5 depth=FIXED_LENGTH4

        hls::stream<ap_int8_c> s6("bn_to_lif_cblk2");      // bn2 -> lif2
#pragma HLS STREAM variable=s6 depth=FIXED_LENGTH4

        hls::stream<ap_int8_c> s7("lif_to_mp_cblk2");      // lif2 -> mp2
#pragma HLS STREAM variable=s7 depth=FIXED_LENGTH4

        hls::stream<ap_int8_c> s8("mp_to_qi_cblk3");       // mp2 -> qi(cblk3)
#pragma HLS STREAM variable=s8 depth=FIXED_LENGTH5

        hls::stream<ap_int8_c> s9("qi_to_conv_cblk3");     // qi -> conv3
#pragma HLS STREAM variable=s9 depth=FIXED_LENGTH5

        hls::stream<ap_int8_c> s10("conv_to_bn_cblk3");    // conv3 -> bn3
#pragma HLS STREAM variable=s10 depth=FIXED_LENGTH6

        hls::stream<ap_int8_c> s11("bn_to_lif_cblk3");     // bn3 -> lif3
#pragma HLS STREAM variable=s11 depth=FIXED_LENGTH6

        hls::stream<ap_int8_c> s12("lif_to_mp_cblk3");     // lif3 -> mp3
#pragma HLS STREAM variable=s12 depth=FIXED_LENGTH6

        hls::stream<ap_int8_c> s13("mp_to_qi_lblk1");      // mp3 -> qi(lblk1)
#pragma HLS STREAM variable=s13 depth=FIXED_LENGTH7     // 24 * 20 = 480

        hls::stream<ap_int8_c> s14("qi_to_fc_lblk1");      // qi -> fc1
#pragma HLS STREAM variable=s14 depth=FIXED_LENGTH7     // 480

        hls::stream<ap_int8_c> s15("fc_to_lif_lblk1");     // fc1 -> lif(lblk1)
#pragma HLS STREAM variable=s15 depth=FIXED_LENGTH8     // 128

        hls::stream<ap_int8_c> s16("lif_to_qi_lblk2");     // lif -> qi(head)
#pragma HLS STREAM variable=s16 depth=FIXED_LENGTH8     // 128

        hls::stream<ap_int8_c> s17("qi_to_fc_lblk2");      // qi -> fc(head)
#pragma HLS STREAM variable=s17 depth=FIXED_LENGTH8     // 128

        hls::stream<ap_int8_c> s18("fc_to_lif_lblk2");     // fc -> lif(head)
#pragma HLS STREAM variable=s18 depth=FIXED_LENGTH9     // 4

        // internal stream for THIS step’s head-LIF output
        hls::stream<ap_int8_c> s_out_step("lif_out_step");
#pragma HLS STREAM variable=s_out_step depth=4

        // reset LIF states ONCE per record (match utils.reset(net))
        qcsnet4_cblk1_lif1d_int.reset();
        qcsnet4_cblk2_lif1d_int.reset();
        qcsnet4_cblk3_lif1d_int.reset();
        qcsnet4_lblk1_lif1d_int.reset();
        qcsnet4_lblk2_lif1d_int.reset();

        // widen accumulators for spike counts (four output neurons)
        ap_int<16> sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;

REPEAT_STEPS:
        for (int t = 0; t < NUM_STEPS; ++t) {
            // (a) re-stream the SAME buffered input into a small local FIFO
            hls::stream<ap_int8_c> s_in("in_replay");
#pragma HLS STREAM variable=s_in depth=CONV_IN_LENGTH1
            for (int k = 0; k < CONV_IN_LENGTH1; ++k) {
#pragma HLS PIPELINE II=1
                s_in.write(in_buf[k]);
            }

            //-----------------------------------------------------------------------------
            // cblk1: conv → bn → lif → mp
            qcsnet4_cblk1_qconv1d_int.forward(
                s_in, s0,
                qcsnet4_cblk1_qconv1d_weights,
                qcsnet4_cblk1_qconv1d_scale_multiplier,
                qcsnet4_cblk1_qconv1d_right_shift,
                qcsnet4_cblk1_qconv1d_bias,
                qcsnet4_cblk1_qconv1d_input_zero_point,
                qcsnet4_cblk1_qconv1d_weight_sum);

            qcsnet4_cblk1_bn1d_int.forward(
                s0, s1,
                qcsnet4_cblk1_batch_norm_weight,
                qcsnet4_cblk1_batch_norm_bias,
                qcsnet4_cblk1_batch_norm_scale_multiplier,
                qcsnet4_cblk1_batch_norm_right_shift);

            qcsnet4_cblk1_lif1d_int.forward(
                s1, s2,
                qcsnet4_cblk1_leaky_beta_int,
                qcsnet4_cblk1_leaky_theta_int,
                qcsnet4_cblk1_leaky_scale_int);

            qcsnet4_cblk1_maxpool1d.forward(s2, s3);

            //----------------------------------------------------------------------------------
            // quant identity before next conv block (maps {0,1}->{0,q_one})
            qcsnet4_cblk2_qi_int.forward(
                s3, s4,
                qcsnet4_cblk2_input_act_scale_int);

            // cblk2: conv → bn → lif → mp
            qcsnet4_cblk2_qconv1d_int.forward(
                s4, s5,
                qcsnet4_cblk2_qconv1d_weights,
                qcsnet4_cblk2_qconv1d_scale_multiplier,
                qcsnet4_cblk2_qconv1d_right_shift,
                qcsnet4_cblk2_qconv1d_bias,
                qcsnet4_cblk2_qconv1d_input_zero_point,
                qcsnet4_cblk2_qconv1d_weight_sum);

            qcsnet4_cblk2_bn1d_int.forward(
                s5, s6,
                qcsnet4_cblk2_batch_norm_weight,
                qcsnet4_cblk2_batch_norm_bias,
                qcsnet4_cblk2_batch_norm_scale_multiplier,
                qcsnet4_cblk2_batch_norm_right_shift);

            qcsnet4_cblk2_lif1d_int.forward(
                s6, s7,
                qcsnet4_cblk2_leaky_beta_int,
                qcsnet4_cblk2_leaky_theta_int,
                qcsnet4_cblk2_leaky_scale_int);

            qcsnet4_cblk2_maxpool1d.forward(s7, s8);

            //---------------------------------------------------------------------------
            // quant identity before next conv block (maps {0,1}->{0,q_one})
            qcsnet4_cblk3_qi_int.forward(
                s8, s9,
                qcsnet4_cblk3_input_act_scale_int);

            // cblk3: conv → bn → lif → mp
            qcsnet4_cblk3_qconv1d_int.forward(
                s9, s10,
                qcsnet4_cblk3_qconv1d_weights,
                qcsnet4_cblk3_qconv1d_scale_multiplier,
                qcsnet4_cblk3_qconv1d_right_shift,
                qcsnet4_cblk3_qconv1d_bias,
                qcsnet4_cblk3_qconv1d_input_zero_point,
                qcsnet4_cblk3_qconv1d_weight_sum);

            qcsnet4_cblk3_bn1d_int.forward(
                s10, s11,
                qcsnet4_cblk3_batch_norm_weight,
                qcsnet4_cblk3_batch_norm_bias,
                qcsnet4_cblk3_batch_norm_scale_multiplier,
                qcsnet4_cblk3_batch_norm_right_shift);

            qcsnet4_cblk3_lif1d_int.forward(
                s11, s12,
                qcsnet4_cblk3_leaky_beta_int,
                qcsnet4_cblk3_leaky_theta_int,
                qcsnet4_cblk3_leaky_scale_int);

            qcsnet4_cblk3_maxpool1d.forward(s12, s13);  // → 24 × 20 = 480

            //---------------------------------------------------------------------------
            // quant identity before first linear block (24 × 20 → 24 × 20)
            qcsnet4_lblk1_qi_int.forward(
                s13, s14,
                qcsnet4_lblk1_input_act_scale_int);

            // dense head block 1: linear(480→128) → lif(128)
            qcsnet4_lblk1_qlinear1d_int.forward(
                s14, s15,
                qcsnet4_lblk1_qlinear_weights,
                qcsnet4_lblk1_qlinear_scale_multiplier,
                qcsnet4_lblk1_qlinear_right_shift,
                qcsnet4_lblk1_qlinear_bias,
                qcsnet4_lblk1_qlinear_input_zero_point,
                qcsnet4_lblk1_qlinear_weight_sum);

            qcsnet4_lblk1_lif1d_int.forward(
                s15, s16,
                qcsnet4_lblk1_leaky_beta_int,
                qcsnet4_lblk1_leaky_theta_int,
                qcsnet4_lblk1_leaky_scale_int);

            //---------------------------------------------------------------------------
            // quant identity before final linear head (128 → 128)
            qcsnet4_lblk2_qi_int.forward(
                s16, s17,
                qcsnet4_lblk2_input_act_scale_int);

            // dense head block 2: linear(128→4) → lif(4)
            qcsnet4_lblk2_qlinear1d_int.forward(
                s17, s18,
                qcsnet4_lblk2_qlinear_weights,
                qcsnet4_lblk2_qlinear_scale_multiplier,
                qcsnet4_lblk2_qlinear_right_shift,
                qcsnet4_lblk2_qlinear_bias,
                qcsnet4_lblk2_qlinear_input_zero_point,
                qcsnet4_lblk2_qlinear_weight_sum);

            qcsnet4_lblk2_lif1d_int.forward(
                s18, s_out_step,
                qcsnet4_lblk2_leaky_beta_int,
                qcsnet4_lblk2_leaky_theta_int,
                qcsnet4_lblk2_leaky_scale_int);

            // (c) consume this step’s four outputs and accumulate counts
            ap_int8_c y0 = s_out_step.read();
            ap_int8_c y1 = s_out_step.read();
            ap_int8_c y2 = s_out_step.read();
            ap_int8_c y3 = s_out_step.read();

            sum0 = sum0 + (ap_int<16>)y0;
            sum1 = sum1 + (ap_int<16>)y1;
            sum2 = sum2 + (ap_int<16>)y2;
            sum3 = sum3 + (ap_int<16>)y3;
        }

        // (d) write the averaged outputs once (Q12 fixed-point average)
        const int F = 12;
        ap_int<16> invT_q = ((ap_int<16>)(1 << F) + (ap_int<16>)(NUM_STEPS >> 1))
                            / (ap_int<16>)NUM_STEPS;

        auto avg_q12_to_i8 = [&](ap_int<16> s)->ap_int8_c {
            ap_int<32> p = (ap_int<32>)s * (ap_int<32>)invT_q; // Q12 product
            p += (ap_int<32>)(1 << (F - 1));                   // round to nearest
            ap_int<16> a = (ap_int<16>)(p >> F);               // back to int
            if (a > 127)  return ap_int8_c(127);
            if (a < -128) return ap_int8_c(-128);
            return (ap_int8_c)a;
        };

        output_stream.write( avg_q12_to_i8(sum0) );
        output_stream.write( avg_q12_to_i8(sum1) );
        output_stream.write( avg_q12_to_i8(sum2) );
        output_stream.write( avg_q12_to_i8(sum3) );
    }

/* ================================================================
 *  3.  LAYER OBJECTS
 * ================================================================ */
private:
    /* 1st conv block */
    Conv1D_SD<1, 16, 3, 1, CONV_IN_LENGTH1>                   qcsnet4_cblk1_qconv1d_int;
    BatchNorm1D_SD<OUT_CH1, FEATURE_LENGTH1>                  qcsnet4_cblk1_bn1d_int;
    LIF1D_SD_Integer<OUT_CH1, FEATURE_LENGTH1>                qcsnet4_cblk1_lif1d_int;
    MaxPool1D_SD<2, 2, OUT_CH1, FEATURE_LENGTH1>              qcsnet4_cblk1_maxpool1d;

    /* 2nd conv block */
    QuantIdentityPerTensor_Int8<OUT_CH1, CONV_IN_LENGTH2>     qcsnet4_cblk2_qi_int;
    Conv1D_SD<16, 16, 3, 1, CONV_IN_LENGTH2>                  qcsnet4_cblk2_qconv1d_int;
    BatchNorm1D_SD<OUT_CH2, FEATURE_LENGTH2>                  qcsnet4_cblk2_bn1d_int;
    LIF1D_SD_Integer<OUT_CH2, FEATURE_LENGTH2>                qcsnet4_cblk2_lif1d_int;
    MaxPool1D_SD<2, 2, OUT_CH2, FEATURE_LENGTH2>              qcsnet4_cblk2_maxpool1d;
    
    /* 3rd conv block */
    QuantIdentityPerTensor_Int8<OUT_CH2, CONV_IN_LENGTH3>     qcsnet4_cblk3_qi_int;
    Conv1D_SD<16, 24, 3, 1, CONV_IN_LENGTH3>                  qcsnet4_cblk3_qconv1d_int;
    BatchNorm1D_SD<OUT_CH3, FEATURE_LENGTH3>                  qcsnet4_cblk3_bn1d_int;
    LIF1D_SD_Integer<OUT_CH3, FEATURE_LENGTH3>                qcsnet4_cblk3_lif1d_int;
    MaxPool1D_SD<2, 2, OUT_CH3, FEATURE_LENGTH3>              qcsnet4_cblk3_maxpool1d;

    /* dense head block 1: 24×20 → 480 → 128 */
    QuantIdentityPerTensor_Int8<OUT_CH3, POOL3_OUT_LEN>       qcsnet4_lblk1_qi_int;      // 24×20 = 480
    Linear1D_SD<LINEAR_IN_SIZE1, LINEAR_OUT_SIZE1>            qcsnet4_lblk1_qlinear1d_int; // 480→128
    LIF1D_SD_Integer<LINEAR_OUT_SIZE1, 1>                     qcsnet4_lblk1_lif1d_int;     // 128×1 = 128

    /* dense head block 2: 128 → 4 */
    QuantIdentityPerTensor_Int8<LINEAR_OUT_SIZE1, 1>          qcsnet4_lblk2_qi_int;        // 128×1 = 128
    Linear1D_SD<LINEAR_OUT_SIZE1, 4>                          qcsnet4_lblk2_qlinear1d_int; // 128→4
    LIF1D_SD_Integer<4, 1>                                    qcsnet4_lblk2_lif1d_int;      // 4×1 = 4
};

} // namespace hls4csnn1d_cblk_sd
#endif /* NEURAL_NETWORK4_CBLK1_SD_H */

#ifndef CONSTANT_H
#define CONSTANT_H

#include <hls_stream.h>
#include <ap_fixed.h>
#include <ap_axi_sdata.h>
#include <array>
#include <any>
#include <map>

#include <nlohmann/json.hpp>


typedef nlohmann::json json;
//typedef std::any any;
// In both FileReader.h and HydrateNetwork.h (at the top)
typedef std::map<std::string, json> JsonMap;
// typedef ap_fixed<8, 4, AP_RND_CONV, AP_SAT> ap_fixed_c; 
// typedef ap_fixed<8, 3, AP_RND_CONV, AP_SAT> ap_fixed_c; 
// typedef ap_fixed<9, 2, AP_RND_CONV, AP_SAT> ap_fixed_c; 
// typedef ap_fixed<32, 26, AP_RND, AP_SAT> ap_fixed_c; 
typedef ap_int<8> ap_int8_c; 
typedef ap_int<32> acc32_t;


// Upper‑bound for synthesis loop (change to suit your application)
// static const int MAX_SAMPLES = 16384;

// Constants
const int NUM_CLASSES_2 = 2;
// const int NUM_CLASSES_4 = 4;

// If you have these defined elsewhere, remove or adjust accordingly
// Adjust or remove these if you have them defined elsewhere.
// const int MAX_CONV_INPUT_CHANNELS  = 1;
// const int MAX_CONV_OUTPUT_CHANNELS = 16;
// const int MAX_KERNEL_SIZE = 3;

// const int MAX_BN_OUTPUT_CHANNELS = 35;

// const int MAX_LIF_RESET_X = 90;
// const int MAX_LIF_RESET_Y_2D = 130;
// const int MAX_LIF_RESET_Y_3D = 28;
// const int MAX_LIF_RESET_Z = 185;

// const int MAX_LINEAR2_INPUT_CHANNELS = 124;
// const int MAX_LINEAR2_OUTPUT_CHANNELS = 4;
// const int MAX_LINEAR4_INPUT_CHANNELS = 495;
// const int MAX_LINEAR4_OUTPUT_CHANNELS = 135;



// const int MAX_CONV_LAYERS = 1;
// const int MAX_BN_LAYERS = 5;
// const int MAX_LIF_LAYERS = 8;
// const int MAX_LINEAR2_LAYERS = 1;
// const int MAX_LINEAR4_LAYERS = 4;
// const int MAX_NN_LAYERS = 18;

/**
 * Maximum number of layers and max length of layer names.
 * Adjust these based on your design constraints.
 */
const int MAX_LAYERS = 10; // Maximum number of layers per type
static const int MAX_LAYER_NAME_LENGTH = 64;

// Upper‑bound for synthesis loop (change to suit your application)
static const int NUM_SAMPLES = 80; //4; //2489; //51002; //1385; //51013; 
static const int NUM_STEPS = 10;



// Assuming that each row has a fixed length of 180
// const int FIXED_LENGTH1 = 180;
// const int FIXED_LENGTH2 = 2136;
// const int FIXED_LENGTH3 = 1068;
// const int FIXED_LENGTH4 = 1392;
// const int FIXED_LENGTH5 = 1700;
// const int FIXED_LENGTH6 = 840;
// const int FIXED_LENGTH7 = 960;
// const int FIXED_LENGTH8 = 2;


const int FIXED_LENGTH1 = 180;
const int FIXED_LENGTH2 = 2848;
const int FIXED_LENGTH3 = 1424;
const int FIXED_LENGTH4 = 2088;
const int FIXED_LENGTH5 = 1032;
const int FIXED_LENGTH6 = 2;

// const int CONV_IN_LENGTH1 = 180;

// const int CONV_IN_LENGTH2 = 60;

// const int BN_LENGTH1 = 60;
// const int BN_LENGTH2 = 30;

// const int MP_IN_LEN1 = 30;
// const int MP_IN_LEN2 = 10;

// const int FEATURE_LENGTH1 = 60;
// const int FEATURE_LENGTH2 = 10;
// const int OUT_CH1 = 16;
// const int OUT_CH2 = 24;

// // ──────────── Dimensional bookkeeping ────────────
// // Layer‑0 input row ...........................................
// const int CONV_IN_LENGTH1   = 180;     // 1 × 180
// const int STRIDE = 1;

// // Block‑1 .....................................................
// const int OUT_CH1           = 12;
// const int FEATURE_LENGTH1   = (CONV_IN_LENGTH1 - 3) / STRIDE + 1;   //(180-3) + 1 = 178
// const int POOL1_IN_LEN      = FEATURE_LENGTH1;                //
// const int POOL1_OUT_LEN     = (POOL1_IN_LEN - 2) / 2 + 1;     //89

// // Block‑2 .....................................................
// const int CONV_IN_LENGTH2   = POOL1_OUT_LEN;                  //89
// const int OUT_CH2           = 16;
// const int FEATURE_LENGTH2   = (CONV_IN_LENGTH2 - 3) / STRIDE + 1;  //87


// // Block‑3 .....................................................
// const int CONV_IN_LENGTH3   = FEATURE_LENGTH2;   //87
// const int OUT_CH3           = 20;
// const int FEATURE_LENGTH3   = (CONV_IN_LENGTH3 - 3) / STRIDE + 1; //85
// const int POOL3_IN_LEN      = FEATURE_LENGTH3;                
// const int POOL3_OUT_LEN     = (POOL3_IN_LEN - 2) / 2 + 1;     //42

// // Block‑2 .....................................................
// const int CONV_IN_LENGTH4   = POOL3_OUT_LEN;                 
// const int OUT_CH4           = 24;
// const int FEATURE_LENGTH4   = (CONV_IN_LENGTH4 - 3) / STRIDE + 1; //40

// // Dense head ..................................................
// const int LINEAR_IN_SIZE    = OUT_CH4 * FEATURE_LENGTH4;    //960    


// ──────────── Dimensional bookkeeping ────────────
// Layer‑0 input row ...........................................
const int CONV_IN_LENGTH1   = 180;     // 1 × 180
const int STRIDE = 1;

// Block‑1 .....................................................
const int OUT_CH1           = 16;
const int FEATURE_LENGTH1   = (CONV_IN_LENGTH1 - 3) / STRIDE + 1;   // (180‑3)/3+1 = 60
const int POOL1_IN_LEN      = FEATURE_LENGTH1;                // 60
const int POOL1_OUT_LEN     = (POOL1_IN_LEN - 2) / 2 + 1;     // (60‑2)/2+1 = 30

// Block‑2 .....................................................
const int CONV_IN_LENGTH2   = POOL1_OUT_LEN;                  // 30
const int OUT_CH2           = 24;
const int FEATURE_LENGTH2   = (CONV_IN_LENGTH2 - 3) / STRIDE + 1;  // (30‑3)/3+1 = 10
const int POOL2_IN_LEN      = FEATURE_LENGTH2;                // 10
const int POOL2_OUT_LEN     = (POOL2_IN_LEN - 2) / 2 + 1;     // (10‑2)/2+1 = 5

// Dense head ..................................................
const int LINEAR_IN_SIZE    = OUT_CH2 * POOL2_OUT_LEN;        // 24 × 5 = 120


// ──────────── Dimensional bookkeeping nn4 ────────────

// Block‑2 .....................................................
const int OUT4_CH2           = 16;
const int FEAT_LEN2          = (CONV_IN_LENGTH2 - 3) / STRIDE + 1;  // (30‑3)/3+1 = 10
const int POOL2_IN_LEN4      = FEAT_LEN2;                // 10
const int POOL2_OUT_LEN4     = (POOL2_IN_LEN4 - 2) / 2 + 1;     // (10‑2)/2+1 = 5

// Block-3 .....................................................
const int CONV_IN_LENGTH3    = POOL2_OUT_LEN4;
const int OUT4_CH3           = 24;
const int FEAT_LEN3          = (CONV_IN_LENGTH3 - 3) / STRIDE + 1;
const int POOL3_IN_LEN4      = FEAT_LEN3;
const int POOL3_OUT_LEN4     = (POOL3_IN_LEN4 -2) / 2 + 1;

// Dense head ..................................................
const int LIN4_IN_SIZE1    = OUT4_CH3 * POOL3_OUT_LEN4;        // 24 × 5 = 120
const int LIN4_OUT_SIZE1    = 128;





typedef std::array<ap_int8_c, FIXED_LENGTH1> array180_t;// Using std::array for compatibility

typedef std::array<ap_int8_c, FIXED_LENGTH2> array2848_t;
typedef std::array<ap_int8_c, FIXED_LENGTH3> array1424_t;
typedef std::array<ap_int8_c, FIXED_LENGTH4> array2088_t;
typedef std::array<ap_int8_c, FIXED_LENGTH5> array1032_t;
typedef std::array<ap_int8_c, FIXED_LENGTH6> array2_t;

// Custom types
typedef std::array<ap_int8_c, FIXED_LENGTH1> input180_data_type;

typedef std::array<ap_int8_c, FIXED_LENGTH2> input2848_data_type;
typedef std::array<ap_int8_c, FIXED_LENGTH3> input1424_data_type;
typedef std::array<ap_int8_c, FIXED_LENGTH4> input2088_data_type;
typedef std::array<ap_int8_c, FIXED_LENGTH5> input1032_data_type;
typedef std::array<ap_int8_c, FIXED_LENGTH6> input2_data_type;

// typedef std::array<ap_fixed_c, FIXED_LENGTH2> array2136_t;
// typedef std::array<ap_fixed_c, FIXED_LENGTH3> array1068_t;
// typedef std::array<ap_fixed_c, FIXED_LENGTH4> array1392_t;
// typedef std::array<ap_fixed_c, FIXED_LENGTH5> array1700_t;
// typedef std::array<ap_fixed_c, FIXED_LENGTH6> array860_t;
// typedef std::array<ap_fixed_c, FIXED_LENGTH7> array984_t;
// typedef std::array<ap_fixed_c, FIXED_LENGTH8> array2_t;

// // Custom types
// typedef std::array<ap_fixed_c, FIXED_LENGTH1> input180_data_type;

// typedef std::array<ap_fixed_c, FIXED_LENGTH2> input2136_data_type;
// typedef std::array<ap_fixed_c, FIXED_LENGTH3> input1068_data_type;
// typedef std::array<ap_fixed_c, FIXED_LENGTH4> input1392_data_type;
// typedef std::array<ap_fixed_c, FIXED_LENGTH5> input1700_data_type;
// typedef std::array<ap_fixed_c, FIXED_LENGTH6> input860_data_type;
// typedef std::array<ap_fixed_c, FIXED_LENGTH7> input984_data_type;
// typedef std::array<ap_fixed_c, FIXED_LENGTH8> input2_data_type;


// Define the AXI stream data type with an 8-bit wide data field.
typedef ap_axiu<64, 0, 0, 0> axi_fixed_t;

static inline ap_int8_c sat_clip_int8(ap_int<32> value) {
    if (value >  127) return  127;
    if (value < -128) return -128;
    return (ap_int8_c)value;
}

// y_q = clip((acc * scale_multiplier + 2^(right_shift-1)) >> right_shift)
static inline ap_int8_c requantize(acc32_t acc, ap_int<32> scale_multiplier, int right_shift) {
    ap_int<32> shifted_result;

    ap_int<64> scaled_product = (ap_int<64>)acc * (ap_int<64>)scale_multiplier;
    
    if (right_shift > 0) {
        ap_int<64> rounding_offset = (ap_int<64>)1 << (right_shift - 1);
        ap_int<64> rounded_value = scaled_product + rounding_offset;
        shifted_result = (ap_int<32>)(rounded_value >> right_shift);
    } else {
        shifted_result = (ap_int<32>)scaled_product;
    }
    
    return sat_clip_int8(shifted_result);
}

#endif
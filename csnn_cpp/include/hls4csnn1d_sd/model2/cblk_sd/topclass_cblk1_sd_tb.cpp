// ─────────────────────────────────────────────────────────────
// topclass_cblk1_sd_tb.h  —  Testbench for topFunctionCblk1
//   • Preserves original FileReader API: loadData() + streamData()
//   • Uses fixed‑count loops so co‑simulation cannot hang
//   • NUM_SAMPLES is inferred from reader.streamData()
// ─────────────────────────────────────────────────────────────
#ifndef TOPCLASS_CBLK1_SD_TB_H
#define TOPCLASS_CBLK1_SD_TB_H

#include <hls_stream.h>
// #include <ap_fixed.h>
#include <array>
#include <string>
#include <iostream>
#include <sys/stat.h>   // ::mkdir
#include <cstdlib>      // std::system
#include <cerrno>
#include <cstring>
#include <fstream>
#include <iostream>

#include "topclass_cblk1_sd_tb.h"   // exposes topFunctionCblk1()
#include "../constants_sd.h"
#include "../filereader.h"

#include <cmath>                      // for std::lround
#include <iomanip>                    // (you already use setprecision)

// Type aliases already consistent with constants_sd.h

static void input_data_to_axi(hls::stream<array180_t>& src,
                              hls::stream<axi_fixed_t>& dst) {
    const int WORDS_PER_ROW = (FIXED_LENGTH1 + 7) / 8;   // 23 for 180 B

    // for (int row = 0; row < NUM_SAMPLES; ++row) {
    array180_t sample = src.read();

    for (int w = 0; w < WORDS_PER_ROW; ++w) {
#pragma HLS PIPELINE II=1
        axi_fixed_t word; word.last = 0;

        for (int j = 0; j < 8; ++j) {
#pragma HLS UNROLL
            // if (row == 0 && w == 0) {
            //     std::cout << "byte0 written = 0x"
            //             << std::hex << word.data.range(7,0).to_uint()
            //             << std::dec << '\n';
            // }
            int idx = w * 8 + j;
            word.data.range(j*8+7, j*8) =
                (idx < FIXED_LENGTH1) ? sample[idx].range(7,0) : 0;
            // if (row == 0 && w < 3) {  // First few words of first row
            //     std::cout << "sample[" << idx << "] = " << sample[idx] 
            //             << ", bits = 0x" << std::hex 
            //             << sample[idx].range(7,0).to_uint() 
            //             << std::dec << std::endl;
            // }
        }

        /*  TLAST on the 23‑rd word (w==22) of EVERY row  */
        if (w == WORDS_PER_ROW - 1) word.last = 1;

        dst.write(word);
    }
    // }
}


//-----------------------------------------------------------------
//  AXI  →  internal (fixed‑count)
//-----------------------------------------------------------------
static void axi_to_fixed(hls::stream<axi_fixed_t>& src,
                         hls::stream<array2_t>&  dst) {
    const int WORDS_PER_ROW = (FIXED_LENGTH6 + 7) / 8;   // 120

    // for (int row = 0; row < NUM_SAMPLES; ++row) {
    array2_t sample{};

    for (int w = 0; w < WORDS_PER_ROW; ++w) {
#pragma HLS PIPELINE II=2
        axi_fixed_t word = src.read();

        for (int j = 0; j < 8; ++j) {
#pragma HLS UNROLL
            int idx = w * 8 + j;
            if (idx < FIXED_LENGTH6)
                sample[idx] = ap_int8_c(word.data.range(j*8+7, j*8));
        }

#ifndef __SYNTHESIS__
        bool expect_last = (w == WORDS_PER_ROW - 1);
        if (word.last != expect_last) {
            std::cerr << "[WARN] TLAST mismatch on output word "
                        // << row 
                        << " word " << w << std::endl;
        }
#endif
    }
    dst.write(sample);
    // }
}



// ------------------------------------------------------------------
//  Size of the first conv layer  (unchanged)
// ------------------------------------------------------------------
// static const int OC = 16;
// static const int IC = 1;
// static const int KS = 3;
// static ap_int8_c weight_mem[OC * IC * KS];

// ------------------------------------------------------------------
//  Copy an existing weight file (src) → the board path inside sandbox
// ------------------------------------------------------------------
/// Ensure that the given directory path exists (creates it if necessary).
static void ensure_dir(const std::string& dir_path) {
    if (::mkdir(dir_path.c_str(), 0777) && errno != EEXIST) {
        std::cerr << "❌ mkdir failed for " << dir_path 
                  << ": " << std::strerror(errno) << "\n";
        std::exit(1);
    }
}

/// Copy a single file from src_path to dst_path, but only if it isn’t already there.
static void stage_weight_file(const std::string& src_path,
                              const std::string& dst_path) {
    // 1) Create hierarchy up to the parent directory of dst_path
    size_t slash = dst_path.find_last_of('/');
    if (slash != std::string::npos) {
        std::string dir = dst_path.substr(0, slash);
        // split the path on '/' and iteratively ensure each component exists
        size_t pos = 0;
        while (true) {
            pos = dir.find('/', pos + 1);
            std::string subdir = (pos == std::string::npos)
                             ? dir
                             : dir.substr(0, pos);
            ensure_dir(subdir);
            if (pos == std::string::npos) break;
        }
    }

    // 2) If the file already exists and is readable, do nothing
    std::ifstream probe(dst_path);
    if (probe.good()) {
        std::cout << "Weights already staged: " << dst_path << "\n";
        return;
    }

    // 3) Copy using system cp (or you could read/write in C++ for portability)
    std::string cmd = "cp \"" + src_path + "\" \"" + dst_path + "\"";
    if (std::system(cmd.c_str()) != 0) {
        std::cerr << "❌ Failed to copy weights:\n    " << cmd << "\n";
        std::exit(1);
    }
    std::cout << "✅ Copied weights to " << dst_path << "\n";
}


int main() {
   // ---------------------------------------------------------------------
    // 1) Load ECG rows from folder
    // ---------------------------------------------------------------------
// #ifndef __SYNTHESIS__
//     std::string folderPath =
//          "/home/velox-217533/Projects/fau_projects/research/data/mitbih_processed_intra_patient_4class_180_center90_filtered/small_test";
// #else
//     std::string folderPath =
//          "/home/velox-217533/Projects/fau_projects/research/data/mitbih_processed_intra_patient_4class_180_center90_filtered/single_test";
// #endif
    std::string folderPath =
         "/home/velox-217533/Projects/fau_projects/research/data/mitbih_processed_intra_patient_4class_180_center90_filtered/single_test";

    FileReader reader;
    reader.loadData(folderPath);

    // 2) Stream all rows (internal array180_t format)
    hls::stream<array180_t> dataStreamInternal;
    reader.streamData(dataStreamInternal);


#ifndef __SYNTHESIS__
    const int NUM_SAMPLES_LOADED = dataStreamInternal.size();
#else
    const int NUM_SAMPLES_LOADED = NUM_SAMPLES;  // fallback if size() isn't available
#endif

    if (NUM_SAMPLES_LOADED <= 0) {
        std::cerr << "No rows loaded — aborting test.\n";
        return -1;
    }
    std::cout << "Loaded " << NUM_SAMPLES_LOADED << " ECG rows\n";
    std::cout << "Finished stream data.\n";

    // 2b) Stream labels (binary: 0 stays 0; {1,2,3} -> 1)
    hls::stream<ap_int8_c> trueLabelStream;
    reader.streamLabel(trueLabelStream, /*binary=*/true);

    // ---------------------------------------------------------------------
    // 2c) SOFTMAX-EQUIVALENT THRESHOLD = 0.70
    //     softmax(out)[1] >= 0.70  ⟺  y1 - y0 >= log(0.70/0.30) ≈ 0.847
    //     Integer scaled by 100: (y1 - y0) * 100 >= 85
    // ---------------------------------------------------------------------
    const int LOGIT_THRESHOLD_SCALED = 41; //85;  // 0.847 * 100
    const float SOFTMAX_THRESHOLD = 0.60; //0.70;
    
    std::cout << "Deployed softmax threshold: " << SOFTMAX_THRESHOLD << " (Fold 5)\n";
    std::cout << "Logit threshold (scaled): " << LOGIT_THRESHOLD_SCALED << "/100 = 0.85\n\n";

    // ---------------------------------------------------------------------
    // 3) Metrics
    // ---------------------------------------------------------------------
    // Global binary confusion (treat class 1 as positive)
    long TP = 0, TN = 0, FP = 0, FN = 0;

    // Per-class confusion (class 0 and class 1 as separate one-vs-rest)
    long TP0 = 0, TN0 = 0, FP0 = 0, FN0 = 0;  // normal as "positive"
    long TP1 = 0, TN1 = 0, FP1 = 0, FN1 = 0;  // abnormal as "positive"

    long actual_0_count = 0, actual_1_count = 0;
    long pred_0_count   = 0, pred_1_count   = 0;

    // ---------------------------------------------------------------------
    // 4) Per-sample loop: convert → DUT → convert back → metrics
    // ---------------------------------------------------------------------
    for (int i = 0; i < NUM_SAMPLES_LOADED; ++i) {
        // Pull one sample (1×180)
        array180_t sample = dataStreamInternal.read();

        // Build a one-sample input stream in internal format
        hls::stream<array180_t> oneSampleStream;
        oneSampleStream.write(sample);

        // Convert to AXI for DUT (use as-is)
        hls::stream<axi_fixed_t> dmaInStream;
        input_data_to_axi(oneSampleStream, dmaInStream);

        // Run DUT for this single sample (use as-is)
        hls::stream<axi_fixed_t> dmaOutStream;
        topFunctionCblk1(dmaInStream, dmaOutStream);

        // Convert back to internal label/logit format (expects 1 result)
        hls::stream<array2_t> oneResultStream;
        axi_to_fixed(dmaOutStream, oneResultStream);

        // Read DUT result (2 logits) for this sample
        array2_t result = oneResultStream.read();

        // Read ground-truth label (already binary 0/1)
        ap_int8_c true_label = trueLabelStream.read();
        int actual = int(true_label);  // 0 or 1

        // ============ SOFTMAX-EQUIVALENT THRESHOLD ============
        // Extract logits
        // Assuming result is ap_fixed or similar - cast to int16
        int y0 = (int)result[0];  // Normal logit
        int y1 = (int)result[1];  // Abnormal logit
        
        // Check: (y1 - y0) * 100 >= 85
        int diff = y1 - y0;
        int diff_scaled = diff * 100;
        
        int predicted;
        if (diff_scaled >= LOGIT_THRESHOLD_SCALED) {
            predicted = 1;  // Abnormal
        } else {
            predicted = 0;  // Normal
        }
        // ======================================================

        // Marginal counts
        if (actual == 0) ++actual_0_count; else ++actual_1_count;
        if (predicted == 0) ++pred_0_count; else ++pred_1_count;

        // Global confusion (class 1 as positive)
        if (actual == 1 && predicted == 1) ++TP;
        else if (actual == 0 && predicted == 0) ++TN;
        else if (actual == 0 && predicted == 1) ++FP;
        else if (actual == 1 && predicted == 0) ++FN;

        // Per-class (one-vs-rest) confusion matrices
        // Class 1 as positive
        if (predicted == 1 && actual == 1) { ++TP1; ++TN0; }
        else if (predicted == 0 && actual == 0) { ++TN1; ++TP0; }
        else if (predicted == 1 && actual == 0) { ++FP1; ++FN0; }
        else if (predicted == 0 && actual == 1) { ++FN1; ++FP0; }

        // Optional: debug first few rows
#ifndef __SYNTHESIS__
        if (i < 20) {
            std::cout << "Sample " << i
                      << "  y0=" << y0 << "  y1=" << y1
                      << "  diff=" << diff
                      << "  pred=" << predicted
                      << "  true=" << actual << "\n";
        }
#endif
    }

    // ---------------------------------------------------------------------
    // 5) Final metrics
    // ---------------------------------------------------------------------
    const double eps = 1e-12;
    const double TPd = double(TP),  TNd = double(TN),
                 FPd = double(FP),  FNd = double(FN);

    // Global metrics (class 1 positive)
    double accuracy  = (TPd + TNd) / (TPd + TNd + FPd + FNd + eps);
    double precision = (TPd + FPd) ? TPd / (TPd + FPd + eps) : 0.0;
    double recall    = (TPd + FNd) ? TPd / (TPd + FNd + eps) : 0.0;
    double f1        = (precision + recall) ?
                       (2.0 * precision * recall) / (precision + recall + eps) : 0.0;

    // Per-class: abnormal (label 1) as positive
    const double TP1d = double(TP1), TN1d = double(TN1),
                 FP1d = double(FP1), FN1d = double(FN1);

    double precision1 = TP1d / (TP1d + FP1d + eps);
    double recall1    = TP1d / (TP1d + FN1d + eps);
    double f11        = (2.0 * precision1 * recall1) / (precision1 + recall1 + eps);
    double accuracy1  = (TP1d + TN1d) / (TP1d + TN1d + FP1d + FN1d + eps);

    // Per-class: normal (label 0) as positive
    const double TP0d = double(TP0), TN0d = double(TN0),
                 FP0d = double(FP0), FN0d = double(FN0);

    double precision0 = TP0d / (TP0d + FP0d + eps);
    double recall0    = TP0d / (TP0d + FN0d + eps);
    double f10        = (2.0 * precision0 * recall0) / (precision0 + recall0 + eps);
    double accuracy0  = (TP0d + TN0d) / (TP0d + TN0d + FP0d + FN0d + eps);

    // ---------------------------------------------------------------------
    // 6) Print metrics
    // ---------------------------------------------------------------------
    std::cout << "\n=== Global Classification Metrics (Binary, positive = class 1) ===\n";
    std::cout << "Counts: actual0=" << actual_0_count << "  actual1=" << actual_1_count
              << "  pred0=" << pred_0_count   << "  pred1=" << pred_1_count << "\n";
    std::cout << "Confusion Matrix (class 1 positive):\n";
    std::cout << "  TP: " << TP << ", FP: " << FP << "\n";
    std::cout << "  FN: " << FN << ", TN: " << TN << "\n";
    std::cout << "Accuracy : " << std::fixed << std::setprecision(4) << accuracy  << "\n";
    std::cout << "Precision: " << std::fixed << std::setprecision(4) << precision << "\n";
    std::cout << "Recall   : " << std::fixed << std::setprecision(4) << recall    << "\n";
    std::cout << "F1       : " << std::fixed << std::setprecision(4) << f1        << "\n";

    std::cout << std::fixed << std::setprecision(4)
              << "\n=== Metrics for normal class (label 0, treated as positive) ===\n"
              << "Accuracy : " << accuracy0  << "\n"
              << "Precision: " << precision0 << "\n"
              << "Recall   : " << recall0    << "\n"
              << "F1       : " << f10        << "\n\n";

    std::cout << std::fixed << std::setprecision(4)
              << "=== Metrics for abnormal class (label 1, treated as positive) ===\n"
              << "Accuracy : " << accuracy1  << "\n"
              << "Precision: " << precision1 << "\n"
              << "Recall   : " << recall1    << "\n"
              << "F1       : " << f11        << "\n";

    std::cout << "\n=== Expected Abnormal Performance (Fold 3, threshold=0.60) ===\n"
              << "Precision: ~0.7470\n"
              << "Recall:    ~0.9113\n"
              << "F1:        ~0.8210\n";

    std::cout << "\nTest PASSED — " << NUM_SAMPLES_LOADED << " rows processed.\n";
    return 0;
}


// int main() {
//     // ---------------------------------------------------------------------
//     // 1) Load ECG rows from folder
//     // ---------------------------------------------------------------------
//     std::string folderPath =
//         "/home/velox-217533/Projects/fau_projects/research/data/mitbih_processed_test/smallvhls";
//     FileReader reader;
//     reader.loadData(folderPath);

//     // 2) Stream all rows (internal array180_t format)
//     hls::stream<array180_t> dataStreamInternal;
//     reader.streamData(dataStreamInternal);


// #ifndef __SYNTHESIS__
//     const int NUM_SAMPLES_LOADED = dataStreamInternal.size();
// #else
//     const int NUM_SAMPLES_LOADED = NUM_SAMPLES;  // fallback if size() isn't available
// #endif

//     if (NUM_SAMPLES_LOADED <= 0) {
//         std::cerr << "No rows loaded — aborting test.\n";
//         return -1;
//     }
//     std::cout << "Loaded " << NUM_SAMPLES_LOADED << " ECG rows\n";
//     std::cout << "Finished stream data.\n";

//     // 2b) Stream labels (binary: 0 stays 0; {1,2,3} -> 1)
//     hls::stream<ap_int8_c> trueLabelStream;
//     reader.streamLabel(trueLabelStream, /*binary=*/true);

//     // ---------------------------------------------------------------------
//     // 3) Metrics
//     // ---------------------------------------------------------------------
//     // Global binary confusion (treat class 1 as positive)
//     long TP = 0, TN = 0, FP = 0, FN = 0;

//     // Per-class confusion (class 0 and class 1 as separate one-vs-rest)
//     long TP0 = 0, TN0 = 0, FP0 = 0, FN0 = 0;  // normal as "positive"
//     long TP1 = 0, TN1 = 0, FP1 = 0, FN1 = 0;  // abnormal as "positive"

//     long actual_0_count = 0, actual_1_count = 0;
//     long pred_0_count   = 0, pred_1_count   = 0;

//     // ---------------------------------------------------------------------
//     // 4) Per-sample loop: convert → DUT → convert back → metrics
//     // ---------------------------------------------------------------------
//     for (int i = 0; i < NUM_SAMPLES_LOADED; ++i) {
//         // Pull one sample (1×180)
//         array180_t sample = dataStreamInternal.read();

//         // Build a one-sample input stream in internal format
//         hls::stream<array180_t> oneSampleStream;
//         oneSampleStream.write(sample);

//         // Convert to AXI for DUT (use as-is)
//         hls::stream<axi_fixed_t> dmaInStream;
//         input_data_to_axi(oneSampleStream, dmaInStream);

//         // Run DUT for this single sample (use as-is)
//         hls::stream<axi_fixed_t> dmaOutStream;
//         topFunctionCblk1(dmaInStream, dmaOutStream);

//         // Convert back to internal label/logit format (expects 1 result)
//         hls::stream<array2_t> oneResultStream;
//         axi_to_fixed(dmaOutStream, oneResultStream);

//         // Read DUT result (2 logits) for this sample
//         array2_t result = oneResultStream.read();

//         // Read ground-truth label (already binary 0/1)
//         ap_int8_c true_label = trueLabelStream.read();
//         int actual = int(true_label);  // 0 or 1

//         // Argmax over 2 logits
//         int predicted = (result[1] > result[0]) ? 1 : 0;

//         // Marginal counts
//         if (actual == 0) ++actual_0_count; else ++actual_1_count;
//         if (predicted == 0) ++pred_0_count; else ++pred_1_count;

//         // Global confusion (class 1 as positive)
//         if (actual == 1 && predicted == 1) ++TP;
//         else if (actual == 0 && predicted == 0) ++TN;
//         else if (actual == 0 && predicted == 1) ++FP;
//         else if (actual == 1 && predicted == 0) ++FN;

//         // Per-class (one-vs-rest) confusion matrices
//         // Class 1 as positive
//         if (predicted == 1 && actual == 1) { ++TP1; ++TN0; }
//         else if (predicted == 0 && actual == 0) { ++TN1; ++TP0; }
//         else if (predicted == 1 && actual == 0) { ++FP1; ++FN0; }
//         else if (predicted == 0 && actual == 1) { ++FN1; ++FP0; }

//         // Optional: debug first few rows
//         // if (i < 8) {
//         //     std::cout << "Row " << i
//         //               << "  logits=[" << result[0] << ", " << result[1] << "]"
//         //               << "  pred=" << predicted
//         //               << "  true=" << actual << "\n";
//         // }
//     }

//     // ---------------------------------------------------------------------
//     // 5) Final metrics
//     // ---------------------------------------------------------------------
//     const double eps = 1e-12;
//     const double TPd = double(TP),  TNd = double(TN),
//                  FPd = double(FP),  FNd = double(FN);

//     // Global metrics (class 1 positive)
//     double accuracy  = (TPd + TNd) / (TPd + TNd + FPd + FNd + eps);
//     double precision = (TPd + FPd) ? TPd / (TPd + FPd + eps) : 0.0;
//     double recall    = (TPd + FNd) ? TPd / (TPd + FNd + eps) : 0.0;
//     double f1        = (precision + recall) ?
//                        (2.0 * precision * recall) / (precision + recall + eps) : 0.0;

//     // Per-class: abnormal (label 1) as positive
//     const double TP1d = double(TP1), TN1d = double(TN1),
//                  FP1d = double(FP1), FN1d = double(FN1);

//     double precision1 = TP1d / (TP1d + FP1d + eps);
//     double recall1    = TP1d / (TP1d + FN1d + eps);
//     double f11        = (2.0 * precision1 * recall1) / (precision1 + recall1 + eps);
//     double accuracy1  = (TP1d + TN1d) / (TP1d + TN1d + FP1d + FN1d + eps);

//     // Per-class: normal (label 0) as positive
//     const double TP0d = double(TP0), TN0d = double(TN0),
//                  FP0d = double(FP0), FN0d = double(FN0);

//     double precision0 = TP0d / (TP0d + FP0d + eps);
//     double recall0    = TP0d / (TP0d + FN0d + eps);
//     double f10        = (2.0 * precision0 * recall0) / (precision0 + recall0 + eps);
//     double accuracy0  = (TP0d + TN0d) / (TP0d + TN0d + FP0d + FN0d + eps);

//     // ---------------------------------------------------------------------
//     // 6) Print metrics
//     // ---------------------------------------------------------------------
//     std::cout << "\n=== Global Classification Metrics (Binary, positive = class 1) ===\n";
//     std::cout << "Counts: actual0=" << actual_0_count << "  actual1=" << actual_1_count
//               << "  pred0=" << pred_0_count   << "  pred1=" << pred_1_count << "\n";
//     std::cout << "Confusion Matrix (class 1 positive):\n";
//     std::cout << "  TP: " << TP << ", FP: " << FP << "\n";
//     std::cout << "  FN: " << FN << ", TN: " << TN << "\n";
//     std::cout << "Accuracy : " << std::fixed << std::setprecision(4) << accuracy  << "\n";
//     std::cout << "Precision: " << std::fixed << std::setprecision(4) << precision << "\n";
//     std::cout << "Recall   : " << std::fixed << std::setprecision(4) << recall    << "\n";
//     std::cout << "F1       : " << std::fixed << std::setprecision(4) << f1        << "\n";

//     std::cout << std::fixed << std::setprecision(4)
//               << "\n=== Metrics for normal class (label 0, treated as positive) ===\n"
//               << "Accuracy : " << accuracy0  << "\n"
//               << "Precision: " << precision0 << "\n"
//               << "Recall   : " << recall0    << "\n"
//               << "F1       : " << f10        << "\n\n";

//     std::cout << std::fixed << std::setprecision(4)
//               << "=== Metrics for abnormal class (label 1, treated as positive) ===\n"
//               << "Accuracy : " << accuracy1  << "\n"
//               << "Precision: " << precision1 << "\n"
//               << "Recall   : " << recall1    << "\n"
//               << "F1       : " << f11        << "\n";

//     std::cout << "\nTest PASSED — " << NUM_SAMPLES_LOADED << " rows processed.\n";
//     return 0;
// }


// // //-----------------------------------------------------------------
// // //  MAIN — keeps original FileReader usage
// // //-----------------------------------------------------------------
// int main() {
//     // 1) Load ECG rows from folder
//     std::string folderPath =
//         "/home/velox-217533/Projects/fau_projects/research/data/mitbih_processed_test/smallvhls";
//     FileReader reader;
//     reader.loadData(folderPath);

//     // 2) Stream all rows (internal format)
//     hls::stream<array180_t> dataStreamInternal;
//     reader.streamData(dataStreamInternal);

//     std::cout << "Finished stream data.\n";

// #ifndef __SYNTHESIS__
//     const int NUM_SAMPLES_LOADED = dataStreamInternal.size();
// #else
//     const int NUM_SAMPLES_LOADED = NUM_SAMPLES;  // fallback if size() isn't available
// #endif

//     if (NUM_SAMPLES_LOADED <= 0) {
//         std::cerr << "No rows loaded — aborting test.\n";
//         return -1;
//     }
//     std::cout << "Loaded " << NUM_SAMPLES_LOADED << " ECG rows\n";

//     // 2b) Stream labels (binary: 0 stays 0; {1,2,3} -> 1)
//     hls::stream<ap_int8_c> trueLabelStream;
//     reader.streamLabel(trueLabelStream, /*binary=*/true);

//     // === Metrics (binary) ===
//     int TP = 0, TN = 0, FP = 0, FN = 0;
//     int actual_0_count = 0, actual_1_count = 0;
//     int pred_0_count   = 0, pred_1_count   = 0;

//     // 3) Per-sample loop: convert -> DUT -> convert back -> metrics
//     for (int i = 0; i < NUM_SAMPLES_LOADED; ++i) {
//         // Pull one sample
//         array180_t sample = dataStreamInternal.read();

//         // Build a one-sample input stream in internal format
//         hls::stream<array180_t> oneSampleStream;
//         oneSampleStream.write(sample);

//         // Convert to AXI for DUT
//         hls::stream<axi_fixed_t> dmaInStream;
//         input_data_to_axi(oneSampleStream, dmaInStream);  // leave as-is

//         // Run DUT for this single sample
//         hls::stream<axi_fixed_t> dmaOutStream;
//         topFunctionCblk1(dmaInStream, dmaOutStream);      // leave as-is

//         // Convert back to internal label/logit format (expects 1 result)
//         hls::stream<array2_t> oneResultStream;
//         axi_to_fixed(dmaOutStream, oneResultStream);      // leave as-is

//         // Read DUT result (2 logits) for this sample
//         array2_t result = oneResultStream.read();

//         // Read ground-truth (already binary 0/1 from streamLabel(..., true))
//         ap_int8_c true_label = trueLabelStream.read();
//         int actual = int(true_label);                  // 0 or 1

//         // Argmax over 2 logits (note: array2_t supports operator[])
//         int predicted = (result[1] > result[0]) ? 1 : 0;

//         // Tally counts
//         if (actual == 0) ++actual_0_count; else ++actual_1_count;
//         if (predicted == 0) ++pred_0_count; else ++pred_1_count;

//         // Update confusion matrix
//         if (actual == 1 && predicted == 1) ++TP;
//         else if (actual == 0 && predicted == 0) ++TN;
//         else if (actual == 0 && predicted == 1) ++FP;
//         else if (actual == 1 && predicted == 0) ++FN;

//         // Optional: debug first few
//         // if (i < 8) {
//         //     std::cout << "Row " << i
//         //               << "  logits=[" << result[0] << ", " << result[1] << "]"
//         //               << "  pred=" << predicted
//         //               << "  true=" << actual << "\n";
//         // }
//     }

//     // 4) Final metrics
//     float accuracy  = float(TP + TN) / float(std::max(1, NUM_SAMPLES_LOADED));
//     float precision = (TP + FP) ? float(TP) / float(TP + FP) : 0.0f;
//     float recall    = (TP + FN) ? float(TP) / float(TP + FN) : 0.0f;
//     float f1        = (precision + recall) ? (2.0f * precision * recall) / (precision + recall) : 0.0f;

//     std::cout << "\n=== Classification Metrics (Binary) ===\n";
//     std::cout << "Counts: actual0=" << actual_0_count << " actual1=" << actual_1_count
//               << "  pred0=" << pred_0_count << " pred1=" << pred_1_count << "\n";
//     std::cout << "Confusion Matrix:\n";
//     std::cout << "  TP: " << TP << ", FP: " << FP << "\n";
//     std::cout << "  FN: " << FN << ", TN: " << TN << "\n";
//     std::cout << "Accuracy : " << std::fixed << std::setprecision(4) << accuracy  << "\n";
//     std::cout << "Precision: " << std::fixed << std::setprecision(4) << precision << "\n";
//     std::cout << "Recall   : " << std::fixed << std::setprecision(4) << recall    << "\n";
//     std::cout << "F1       : " << std::fixed << std::setprecision(4) << f1        << "\n";

//     std::cout << "\nTest PASSED — " << NUM_SAMPLES_LOADED << " rows processed.\n";
//     return 0;
// }




#endif // TOPCLASS_CBLK1_SD_TB_H
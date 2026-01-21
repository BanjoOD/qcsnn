// ─────────────────────────────────────────────────────────────
// topclass_cblk1_sd_tb.h  —  Testbench for topFunctionCblk1
//   • Preserves original FileReader API: loadData() + streamData()
//   • Uses fixed‑count loops so co‑simulation cannot hang
//   • NUM_SAMPLES is inferred from reader.streamData()
// ─────────────────────────────────────────────────────────────
#ifndef TOPCLASS4_CBLK1_SD_TB_H
#define TOPCLASS4_CBLK1_SD_TB_H

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

#include "topclass4_cblk1_sd_tb.h"   // exposes topFunctionCblk1()
#include "../constants4_sd.h"
#include "../filereader4.h"

#include <cmath>                      // for std::lround
#include <iomanip>                    // (you already use setprecision)

// Type aliases already consistent with constants_sd.h

// ---------------------------------------------------------------------
// 1) Testbench: array180_t → axi_fixed_t (input_data_to_axi)
//    Packs 180 bytes into 23 AXI words (64-bit → 8 bytes/word).
// ---------------------------------------------------------------------
static void input_data_to_axi(hls::stream<array180_t>& src,
                              hls::stream<axi_fixed_t>& dst) {
    const int WORDS_PER_ROW = (FIXED_LENGTH1 + 7) / 8;   // 23 for 180 B

    // Single-sample version: caller wraps per-sample loop.
    array180_t sample = src.read();

    for (int w = 0; w < WORDS_PER_ROW; ++w) {
    #pragma HLS PIPELINE II=1
        axi_fixed_t word;
        word.data = 0;
        word.keep = 0;
        word.last = 0;

        // Pack up to 8 bytes into this 64-bit AXI word
        for (int j = 0; j < 8; ++j) {
        #pragma HLS UNROLL
            int idx = w * 8 + j;
            ap_uint<8> byte_val = 0;

            if (idx < FIXED_LENGTH1) {
                byte_val = sample[idx].range(7, 0);
                // Mark this byte as valid in KEEP
                word.keep |= (ap_uint<8>)(1u << j);
            }

            word.data.range(j * 8 + 7, j * 8) = byte_val;
        }

        // TLAST asserted on last word of the row
        if (w == WORDS_PER_ROW - 1) {
            word.last = 1;
        }

        dst.write(word);
    }
}


// ---------------------------------------------------------------------
// 4) Testbench: axi_fixed_t → array4_t (axi_to_fixed)
//    Unpacks one 64-bit AXI word into 4 int8 logits.
// ---------------------------------------------------------------------
static void axi_to_fixed(hls::stream<axi_fixed_t>& src,
                         hls::stream<array4_t>&   dst) {
    const int WORDS_PER_ROW = (FIXED_LENGTH9 + 7) / 8;   // 1 for 4 bytes

    array4_t sample{};

    for (int w = 0; w < WORDS_PER_ROW; ++w) {
    #pragma HLS PIPELINE II=1
        axi_fixed_t word = src.read();

        // We still loop 8 bytes but only idx < 4 are stored
        for (int j = 0; j < 8; ++j) {
        #pragma HLS UNROLL
            int idx = w * 8 + j;
            if (idx < FIXED_LENGTH9) {
                sample[idx] = ap_int8_c(word.data.range(j * 8 + 7, j * 8));
            }
        }

    #ifndef __SYNTHESIS__
        bool expect_last = (w == WORDS_PER_ROW - 1);
        if (word.last != expect_last) {
            std::cerr << "[WARN] TLAST mismatch on output word " << w << '\n';
        }
    #endif
    }

    dst.write(sample);
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

    // 2b) Stream labels as 4-class (0: N, 1: S, 2: V, 3: F)
    hls::stream<ap_int8_c> trueLabelStream;
    reader.streamLabel(trueLabelStream, /*binary=*/false);

    // ---------------------------------------------------------------------
    // 3) Metrics (multi-class, 4 classes)
    // ---------------------------------------------------------------------
    static const int NUM_CLASSES = 4;

    long TP[NUM_CLASSES];
    long TN[NUM_CLASSES];
    long FP[NUM_CLASSES];
    long FN[NUM_CLASSES];

    long actual_count[NUM_CLASSES];
    long pred_count[NUM_CLASSES];

    for (int c = 0; c < NUM_CLASSES; ++c) {
        TP[c] = TN[c] = FP[c] = FN[c] = 0;
        actual_count[c] = pred_count[c] = 0;
    }

    long total_correct = 0;

    // Optional: class names for printing
    const char* class_names[NUM_CLASSES] = {
        "Class 0 - Normal (N)",
        "Class 1 - SVEB (S)",
        "Class 2 - VEB (V)",
        "Class 3 - F (F)"
    };

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

        // Convert back to internal logit format (expects 4 results)
        hls::stream<array4_t> oneResultStream;
        axi_to_fixed(dmaOutStream, oneResultStream);

        // Read DUT result (4 logits) for this sample
        array4_t result = oneResultStream.read();

        // Read ground-truth label (0,1,2,3)
        ap_int8_c true_label = trueLabelStream.read();
        int actual = static_cast<int>(true_label);  // 0..3

        // Cast logits to int (assuming they're ap_fixed or similar)
        int y0 = static_cast<int>(result[0]);
        int y1 = static_cast<int>(result[1]);
        int y2 = static_cast<int>(result[2]);
        int y3 = static_cast<int>(result[3]);

        // -------------------- PLAIN ARGMAX (NO WEIGHTS) --------------------
        int predicted = 0;
        int best = y0;

        if (y1 > best) {
            best = y1;
            predicted = 1;
        }
        if (y2 > best) {
            best = y2;
            predicted = 2;
        }
        if (y3 > best) {
            best = y3;
            predicted = 3;
        }
        // -------------------------------------------------------------------

        // Bounds check (defensive, should not trigger)
        if (actual < 0 || actual >= NUM_CLASSES) {
            std::cerr << "Warning: actual label out of range: " << actual << "\n";
            continue;
        }
        if (predicted < 0 || predicted >= NUM_CLASSES) {
            std::cerr << "Warning: predicted label out of range: " << predicted << "\n";
            continue;
        }

        // Marginal counts
        ++actual_count[actual];
        ++pred_count[predicted];

        if (predicted == actual) {
            ++total_correct;
        }

        // Per-class one-vs-rest confusion updates
        for (int c = 0; c < NUM_CLASSES; ++c) {
            if (actual == c && predicted == c) {
                ++TP[c];
            } else if (actual == c && predicted != c) {
                ++FN[c];
            } else if (actual != c && predicted == c) {
                ++FP[c];
            } else {
                ++TN[c];
            }
        }

        // Optional: debug first few rows
#ifndef __SYNTHESIS__
        if (i < 20) {
            std::cout << "Sample " << i
                      << "  raw=[" << y0 << "," << y1 << "," << y2 << "," << y3 << "]"
                      << "  pred=" << predicted
                      << "  true=" << actual << "\n";
        }
#endif
    }

    // ---------------------------------------------------------------------
    // 5) Final metrics
    // ---------------------------------------------------------------------
    const double eps = 1e-12;

    // Overall accuracy (multi-class)
    double overall_accuracy =
        (NUM_SAMPLES_LOADED > 0)
        ? static_cast<double>(total_correct) /
          static_cast<double>(NUM_SAMPLES_LOADED)
        : 0.0;

    std::cout << "\n=== Overall Multi-class Accuracy ===\n";
    std::cout << "Total samples : " << NUM_SAMPLES_LOADED << "\n";
    std::cout << "Total correct : " << total_correct << "\n";
    std::cout << "Accuracy      : " << std::fixed << std::setprecision(4)
              << overall_accuracy << "\n";

    std::cout << "\n=== Per-class Metrics (one-vs-rest) ===\n";

    for (int c = 0; c < NUM_CLASSES; ++c) {
        double TPd = static_cast<double>(TP[c]);
        double TNd = static_cast<double>(TN[c]);
        double FPd = static_cast<double>(FP[c]);
        double FNd = static_cast<double>(FN[c]);

        double prec = (TPd + FPd > 0.0)
                        ? TPd / (TPd + FPd + eps)
                        : 0.0;
        double rec  = (TPd + FNd > 0.0)
                        ? TPd / (TPd + FNd + eps)
                        : 0.0;
        double f1   = (prec + rec > 0.0)
                        ? (2.0 * prec * rec) / (prec + rec + eps)
                        : 0.0;
        double acc_c = (TPd + TNd) /
                       (TPd + TNd + FPd + FNd + eps);

        std::cout << "\n" << class_names[c] << "\n";
        std::cout << "  Counts: actual=" << actual_count[c]
                  << "  pred="   << pred_count[c]   << "\n";
        std::cout << "  Confusion (one-vs-rest):\n";
        std::cout << "    TP: " << TP[c]
                  << "  FP: " << FP[c] << "\n";
        std::cout << "    FN: " << FN[c]
                  << "  TN: " << TN[c] << "\n";

        std::cout << std::fixed << std::setprecision(4)
                  << "  Accuracy : " << acc_c << "\n"
                  << "  Precision: " << prec  << "\n"
                  << "  Recall   : " << rec   << "\n"
                  << "  F1       : " << f1    << "\n";
    }

    std::cout << "\n=== Expected Python Performance (argmax) ===\n"
              << "Accuracy:         0.9838 (98.38%)\n"
              << "Normal Prec/Rec:  99.49% / 98.93%\n"      
              << "SVEB Prec/Rec:    81.01% / 88.87%\n"       
              << "VEB Prec/Rec:     96.34% / 96.34%\n"
              << "F Prec/Rec:       67.15% / 87.42%\n";
    

    std::cout << "\nTest COMPLETE — " << NUM_SAMPLES_LOADED << " rows processed.\n";
    return 0;
}


#endif // TOPCLASS_CBLK1_SD_TB_H




// int main() {
//     // ---------------------------------------------------------------------
//     // 1) Load ECG rows from folder
//     // ---------------------------------------------------------------------
// #ifndef __SYNTHESIS__
//     std::string folderPath =
//          "/home/velox-217533/Projects/fau_projects/research/data/mitbih_processed_intra_patient_4class_180_center90_filtered/small_test";
// #else
//     std::string folderPath =
//          "/home/velox-217533/Projects/fau_projects/research/data/mitbih_processed_intra_patient_4class_180_center90_filtered/single_test";
// #endif
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

//     // 2b) Stream labels as 4-class (0: N, 1: S, 2: V, 3: F)
//     hls::stream<ap_int8_c> trueLabelStream;
//     reader.streamLabel(trueLabelStream, /*binary=*/false);

//     // ---------------------------------------------------------------------
//     // 3) Metrics (multi-class, 4 classes)
//     // ---------------------------------------------------------------------
//     static const int NUM_CLASSES = 4;

//     long TP[NUM_CLASSES];
//     long TN[NUM_CLASSES];
//     long FP[NUM_CLASSES];
//     long FN[NUM_CLASSES];

//     long actual_count[NUM_CLASSES];
//     long pred_count[NUM_CLASSES];

//     for (int c = 0; c < NUM_CLASSES; ++c) {
//         TP[c] = TN[c] = FP[c] = FN[c] = 0;
//         actual_count[c] = pred_count[c] = 0;
//     }

//     long total_correct = 0;

//     // Optional: class names for printing
//     const char* class_names[NUM_CLASSES] = {
//         "Class 0 - Normal (N)",
//         "Class 1 - SVEB (S)",
//         "Class 2 - VEB (V)",
//         "Class 3 - F (F)"
//     };

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

//         // Convert back to internal logit format (expects 4 results)
//         hls::stream<array4_t> oneResultStream;
//         axi_to_fixed(dmaOutStream, oneResultStream);

//         // Read DUT result (4 logits) for this sample
//         array4_t result = oneResultStream.read();

//         // Read ground-truth label (0,1,2,3)
//         ap_int8_c true_label = trueLabelStream.read();
//         int actual = static_cast<int>(true_label);  // 0..3

//         // Argmax over 4 logits
//         int predicted = 0;
//         ap_int8_c best = result[0];
//         for (int c = 1; c < NUM_CLASSES; ++c) {
//             if (result[c] > best) {
//                 best = result[c];
//                 predicted = c;
//             }
//         }

//         // Bounds check (defensive, should not trigger)
//         if (actual < 0 || actual >= NUM_CLASSES) {
//             std::cerr << "Warning: actual label out of range: " << actual << "\n";
//             continue;
//         }
//         if (predicted < 0 || predicted >= NUM_CLASSES) {
//             std::cerr << "Warning: predicted label out of range: " << predicted << "\n";
//             continue;
//         }

//         // Marginal counts
//         ++actual_count[actual];
//         ++pred_count[predicted];

//         if (predicted == actual) {
//             ++total_correct;
//         }

//         // Per-class one-vs-rest confusion updates
//         for (int c = 0; c < NUM_CLASSES; ++c) {
//             if (actual == c && predicted == c) {
//                 ++TP[c];
//             } else if (actual == c && predicted != c) {
//                 ++FN[c];
//             } else if (actual != c && predicted == c) {
//                 ++FP[c];
//             } else {
//                 ++TN[c];
//             }
//         }

//         // Optional: debug first few rows
//         // if (i < 8) {
//         //     std::cout << "Row " << i
//         //               << "  logits=[" << result[0] << ", " << result[1]
//         //               << ", " << result[2] << ", " << result[3] << "]"
//         //               << "  pred=" << predicted
//         //               << "  true=" << actual << "\n";
//         // }
//     }

//     // ---------------------------------------------------------------------
//     // 5) Final metrics
//     // ---------------------------------------------------------------------
//     const double eps = 1e-12;

//     // Overall accuracy (multi-class)
//     double overall_accuracy =
//         (NUM_SAMPLES_LOADED > 0)
//         ? static_cast<double>(total_correct) /
//           static_cast<double>(NUM_SAMPLES_LOADED)
//         : 0.0;

//     std::cout << "\n=== Overall Multi-class Accuracy ===\n";
//     std::cout << "Total samples : " << NUM_SAMPLES_LOADED << "\n";
//     std::cout << "Total correct : " << total_correct << "\n";
//     std::cout << "Accuracy      : " << std::fixed << std::setprecision(4)
//               << overall_accuracy << "\n";

//     std::cout << "\n=== Per-class Metrics (one-vs-rest) ===\n";

//     for (int c = 0; c < NUM_CLASSES; ++c) {
//         double TPd = static_cast<double>(TP[c]);
//         double TNd = static_cast<double>(TN[c]);
//         double FPd = static_cast<double>(FP[c]);
//         double FNd = static_cast<double>(FN[c]);

//         double prec = (TPd + FPd > 0.0)
//                         ? TPd / (TPd + FPd + eps)
//                         : 0.0;
//         double rec  = (TPd + FNd > 0.0)
//                         ? TPd / (TPd + FNd + eps)
//                         : 0.0;
//         double f1   = (prec + rec > 0.0)
//                         ? (2.0 * prec * rec) / (prec + rec + eps)
//                         : 0.0;
//         double acc_c = (TPd + TNd) /
//                        (TPd + TNd + FPd + FNd + eps);

//         std::cout << "\n" << class_names[c] << "\n";
//         std::cout << "  Counts: actual=" << actual_count[c]
//                   << "  pred="   << pred_count[c]   << "\n";
//         std::cout << "  Confusion (one-vs-rest):\n";
//         std::cout << "    TP: " << TP[c]
//                   << "  FP: " << FP[c] << "\n";
//         std::cout << "    FN: " << FN[c]
//                   << "  TN: " << TN[c] << "\n";

//         std::cout << std::fixed << std::setprecision(4)
//                   << "  Accuracy : " << acc_c << "\n"
//                   << "  Precision: " << prec  << "\n"
//                   << "  Recall   : " << rec   << "\n"
//                   << "  F1       : " << f1    << "\n";
//     }

//     std::cout << "\nTest COMPLETE — " << NUM_SAMPLES_LOADED << " rows processed.\n";
//     return 0;
// }

// #endif // TOPCLASS_CBLK1_SD_TB_H
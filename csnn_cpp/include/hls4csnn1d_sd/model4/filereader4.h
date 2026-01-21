#ifndef FILE_READER_H
#define FILE_READER_H

#include <algorithm>
// #include <ap_fixed.h>
#include <ap_int.h>
#include <any>
#include <array>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <hls_stream.h>
#include <iostream>
#include <map>
#include <nlohmann/json.hpp>
#include <random>
#include <sstream>
#include <string>

#include <sys/stat.h>   // for ::mkdir
#include <cerrno>
#include <sys/types.h>



#include <vector>

#include <dirent.h>
#include <sys/types.h>
#include <string.h>

#include "constants4_sd.h"
 // brings qcsnet4_input_scale, qcsnet4_input_zero_point
#include "./weights_sd/qparams_qcsnet4_cblk1_input.h"

using namespace hls4csnn1d_cblk_sd;


class FileReader {
    public:
    // const float      STEP_F    = 0.015625f;           // 2^-4
    // const ap_fixed_c STEP_Q    = ap_fixed_c(STEP_F);
    // const float      INV_STEP  = 1.0f / STEP_F;     // 16.0
    // toggle behavior for short rows
    static constexpr bool PAD_SHORT_ROWS = false;   // true = pad, false = skip

        std::vector<array180_t> X;  // To store all the data
        std::vector<int> y;         // To store labels

        // Load data from multiple folders, concatenate, and assign labels
        void loadData(const std::string& basePath = "./mitbih_processed_test/smallvhls") {
            // Folders and label mapping
#ifndef __SYNTHESIS__
        // std::vector<std::string> folders = {"normalsmaller", "svebsmaller", "vebsmaller", "fsmaller"};
        // std::map<std::string, int> label_map = {{"normalsmaller", 0}, {"svebsmaller", 1}, {"vebsmaller", 2}, {"fsmaller", 3}};
        std::vector<std::string> folders = {"normal", "sveb", "veb", "f"};
        std::map<std::string, int> label_map = {{"normal", 0}, {"sveb", 1}, {"veb", 2}, {"f", 3}};
#else
        std::vector<std::string> folders = {"normalvhls", "svebvhls", "vebvhls", "fvhls"};
        std::map<std::string, int> label_map = {{"normalvhls", 0}, {"svebvhls", 1}, {"vebvhls", 2}, {"fvhls", 3}};

        // std::vector<std::string> folders = {"normal", "sveb", "veb", "f"};
        // std::map<std::string, int> label_map = {{"normal", 0}, {"sveb", 1}, {"veb", 2}, {"f", 3}};
        // std::vector<std::string> folders = {"normalsmaller"};
        // std::map<std::string, int> label_map = {{"normalsmaller", 0}};
            
#endif         
            for (const std::string& folder : folders) {
                std::string folderPath = basePath + "/" + folder;
                loadFolder(folderPath, label_map[folder]);
            }

            // Shuffle the data and labels together
            //shuffleData();
        }

        // Stream the loaded data into an hls_stream
        void streamData(hls::stream<array180_t>& outputStream) {
//             std::cout << "Data Size: " << X.size() << std::endl;
            for (size_t i = 0; i < X.size(); ++i) {
                outputStream.write(X[i]);  // Stream the row
            }
//             std::cout << "output stream Size: " << outputStream.size() << std::endl;
        }
        
        void streamLabel(hls::stream<ap_int8_c>& labelStream, bool binary) {
            if (binary) {
                for (size_t i = 0; i < y.size(); ++i) {
                    // Convert multiclass to binary: 0 stays 0, 1-3 become 1
                    int binary_label = (y[i] == 0) ? 0 : 1;
                    labelStream.write(ap_int8_c(binary_label));
                } 
            }else{ 
                for (size_t i = 0; i < y.size(); ++i) {
                    labelStream.write(y[i]);   // Stream the corresponding label
                } 
            }
        }

        // Function to print one row from the output stream
        void printOneRow(hls::stream<array180_t>& outputStream) {
            if (!outputStream.empty()) {
                // Read one row (array180_t) from the stream
                array180_t row = outputStream.read();

                // Print the values of the row
                std::cout << "Row: ";
                for (int i = 0; i < FIXED_LENGTH1; ++i) {
                    std::cout << row[i] << " ";
                }
                std::cout << std::endl;
            } else {
                std::cout << "The stream is empty!" << std::endl;
            }
        }

        // Method to read weights from JSON file and return map<string, any>
        JsonMap readJsonWeightsOrDims(const std::string& filePath) {
            // Open the JSON file
            std::ifstream inputFile(filePath);
            if (!inputFile.is_open()) {
                std::cerr << "Error opening file: " << filePath << std::endl;
                return {};
            }

            // Parse the JSON content
            json jsonData;
            inputFile >> jsonData;

            // Check for parsing errors
            if (jsonData.is_discarded()) {
                std::cerr << "Parse error: Invalid JSON content." << std::endl;
                return {};
            }


            // Convert jsonData to map<string, any> with flattened structure
            return extractLayerWeightsOrDims(jsonData);
        }

        inline void checksum_row(const array180_t& row, double& sum, double& sumsq, double& maxabs) {
              sum = 0.0; sumsq = 0.0; maxabs = 0.0;
              for (int k = 0; k < FIXED_LENGTH1; ++k) {
                  float v = (float)row[k];
                  sum   += v;
                  sumsq += v * v;
                  float a = std::fabs(v);
                  if (a > maxabs) maxabs = a;
              }
        }

        // Write an hls::stream to a text (CSV) file.
        // T is assumed to be a fixed-length container (e.g., std::array) with .size() and operator[].
        // Optionally, every 'rowsPerBatch' rows, an extra blank line is inserted.
        template <typename T>
        void write_hls_stream_to_text_file(hls::stream<T>& stream, const std::string& filename, int rowsPerBatch = 0) {
            std::ofstream ofs(filename);
            if (!ofs) {
                std::cerr << "Error opening file for writing: " << filename << "\n";
                return;
            }
            int rowCount = 0;
            while (!stream.empty()) {
                T data = stream.read();
                // Write each element of the container separated by commas.
                for (size_t i = 0; i < data.size(); i++) {
                    ofs << data[i];
                    if (i != data.size() - 1)
                        ofs << ",";
                }
                ofs << "\n";
                rowCount++;
                // Insert an extra blank line every rowsPerBatch rows, if requested.
                // if (rowsPerBatch > 0 && rowCount % rowsPerBatch == 0) {
                //     ofs << "\n";
                // }
            }
            ofs.close();
        }

        // // Read an hls::stream from a text (CSV) file.
        // // This function expects each nonempty line to contain comma-separated values that
        // // exactly match the number of elements in T.
        // // T must be a container type (such as std::array) with a fixed size.
        // template <typename T>
        // void read_hls_stream_from_text_file(hls::stream<T>& stream, const std::string& filename) {
        //     std::ifstream ifs(filename);
        //     if (!ifs) {
        //         std::cerr << "Error opening file for reading: " << filename << "\n";
        //         return;
        //     }
        //     std::string line;
        //     while (std::getline(ifs, line)) {
        //         // Skip empty lines (which may separate batches)
        //         if (line.empty()) continue;
        //         T data;
        //         std::istringstream iss(line);
        //         std::string token;
        //         size_t idx = 0;
        //         while (std::getline(iss, token, ',') && idx < data.size()) {
        //             std::istringstream tokenStream(token);
        //             // Deduce the underlying element type of T.
        //             typedef typename std::remove_reference<decltype(data[0])>::type ElementType;
        //             ElementType value;
        //             tokenStream >> value;
        //             data[idx++] = value;
        //         }
        //         if (idx != data.size()) {
        //             std::cerr << "Warning: Expected " << data.size() << " elements, but got " << idx << "\n";
        //         }
        //         stream.write(data);
        //     }
        //     ifs.close();

        //     for (size_t i = 0; i < X.size(); ++i) {
        //         outputStream.write(X[i]);  // Stream the row
        //         labelStream.write(y[i]);   // Stream the corresponding label
        //     }
        // }

        /* ---- number of samples loaded ---- */
		size_t size() const { return X.size(); }

		/* ---- push one mini-batch into a stream ---- */
		void streamBatch(hls::stream<array180_t>& out,
						size_t start, size_t batch)
		{
			for (size_t i = 0; i < batch && start + i < X.size(); ++i)
				out.write(X[start + i]);
		}

		/* ---- matching labels (binary or 4-class) ---- */
		void labelBatch(hls::stream<ap_int8_c>& out,
						size_t start, size_t batch, bool binary)
		{
			for (size_t i = 0; i < batch && start + i < y.size(); ++i) {
				int lbl = y[start + i];
				if (binary) lbl = (lbl == 0 ? 0 : 1);
				out.write(ap_int8_c(lbl));
			}
		}


// Save one quantized row (1×180) to its own CSV file.
// Returns the full path written (or empty string on failure).

// create directory if missing (POSIX; safe to call repeatedly)
inline bool make_dir_if_needed(const std::string& dir) {
    struct stat st;
    if (stat(dir.c_str(), &st) == 0) return (st.st_mode & S_IFDIR) != 0;
    if (mkdir(dir.c_str(), 0755) == 0) return true;
    return (errno == EEXIST);
}

// get filename stem (basename without extension) from a path
inline std::string basename_stem(const std::string& path) {
    size_t slash = path.find_last_of("/\\");
    std::string name = (slash == std::string::npos) ? path : path.substr(slash + 1);
    size_t dot = name.find_last_of('.');
    if (dot != std::string::npos) name = name.substr(0, dot);
    return name;
}


// save one row (1x180) into its own CSV file
inline std::string saveRowToCSV(const array180_t& row,
                                const std::string& out_dir,
                                const std::string& prefix,
                                size_t idx)
{
    make_dir_if_needed(out_dir);  // ignore failure; we'll try open anyway

    std::ostringstream name;
    name << out_dir << '/' << prefix << '_'
         << std::setw(6) << std::setfill('0') << idx << ".csv";
    const std::string path = name.str();

    std::ofstream ofs(path.c_str());
    if (!ofs.is_open()) {
        std::cerr << "Failed to open for write: " << path << "\n";
        return std::string();
    }

    ofs << std::fixed << std::setprecision(6);
    for (int k = 0; k < FIXED_LENGTH1; ++k) {
        if (k) ofs << ',';
        ofs << static_cast<float>(row[k]);   // 1×180 line
    }
    ofs << '\n';
    return path;
}


private:

// Helper function to check if string ends with .csv
bool endsWith(const std::string& str, const std::string& suffix) {
    return str.size() >= suffix.size() && 
           str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// Helper function to load all CSV files from a folder and append data
void loadFolder(const std::string& folderPath, int label) {
    DIR* dir = opendir(folderPath.c_str());
    if (dir != nullptr) {
        struct dirent* entry;
        while ((entry = readdir(dir)) != nullptr) {
            std::string filename(entry->d_name);
            // Skip . and .. directory entries
            if (filename != "." && filename != "..") {
                // Check if file ends with .csv
                if (endsWith(filename, ".csv")) {
                    std::string fullPath = folderPath;
                    // Add trailing slash if needed
                    if (folderPath[folderPath.length()-1] != '/') {
                        fullPath += "/";
                    }
                    fullPath += filename;
                    readCSV(fullPath, label);
                }
            }
        }
        closedir(dir);
    }
}


// // Save one quantized row (1×180) to its own CSV file.
// // Returns the full path written (or empty string on failure).

// // create directory if missing (POSIX; safe to call repeatedly)
// inline bool make_dir_if_needed(const std::string& dir) {
//     struct stat st;
//     if (stat(dir.c_str(), &st) == 0) return (st.st_mode & S_IFDIR) != 0;
//     if (mkdir(dir.c_str(), 0755) == 0) return true;
//     return (errno == EEXIST);
// }

// // get filename stem (basename without extension) from a path
// inline std::string basename_stem(const std::string& path) {
//     size_t slash = path.find_last_of("/\\");
//     std::string name = (slash == std::string::npos) ? path : path.substr(slash + 1);
//     size_t dot = name.find_last_of('.');
//     if (dot != std::string::npos) name = name.substr(0, dot);
//     return name;
// }


// // save one row (1x180) into its own CSV file
// inline std::string saveRowToCSV(const array180_t& row,
//                                 const std::string& out_dir,
//                                 const std::string& prefix,
//                                 size_t idx)
// {
//     make_dir_if_needed(out_dir);  // ignore failure; we'll try open anyway

//     std::ostringstream name;
//     name << out_dir << '/' << prefix << '_'
//          << std::setw(6) << std::setfill('0') << idx << ".csv";
//     const std::string path = name.str();

//     std::ofstream ofs(path.c_str());
//     if (!ofs.is_open()) {
//         std::cerr << "Failed to open for write: " << path << "\n";
//         return std::string();
//     }

//     ofs << std::fixed << std::setprecision(6);
//     for (int k = 0; k < FIXED_LENGTH1; ++k) {
//         if (k) ofs << ',';
//         ofs << static_cast<float>(row[k]);   // 1×180 line
//     }
//     ofs << '\n';
//     return path;
// }



static inline ap_int8_c quantize_input_sample(float x_real) {
    // x_q = round(x_real / scale) + zero_point, then clamp to int8
    const float inv_s = 1.0f / qcsnet4_cblk1_input_scale;
    float qf = std::nearbyintf(x_real * inv_s) + static_cast<float>(qcsnet4_cblk1_input_zero_point);
    int qi = static_cast<int>(qf);
    if (qi > 127)  qi = 127;
    if (qi < -128) qi = -128;
    return ap_int8_c(qi);
}


void readCSV(const std::string& filePath, int label){
    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cout << "[readCSV] ERROR: Failed to open " << filePath << '\n';
        return;
    }

    constexpr bool HAS_HEADER = false;  // Your CSV has no header
    constexpr bool HAS_INDEX_COLUMN = false;  // Your CSV has no index column
    
    std::string line;
    if (HAS_HEADER) {
        std::getline(file, line);  // discard header only if present
        std::cout << "[readCSV] Skipped header line\n";
    }

    constexpr bool APPLY_MINMAX_NORM = false;  // Data is already z-score normalized!

    int line_no = HAS_HEADER ? 1 : 0;

    while (std::getline(file, line)) {
        ++line_no;

        // Skip blank lines
        if (line.find_first_not_of(" \t\r\n,") == std::string::npos) {
            std::cout << "[readCSV] line " << line_no << " blank; skipping.\n";
            continue;
        }

        std::istringstream ls(line);

        float      buf[FIXED_LENGTH1] = {};
        array180_t row = {};
        int        col = 0;

        // ============ MODIFIED: Read ALL columns (no skipping) ============
        std::string val;
        while (std::getline(ls, val, ',')) {
            val = trim(val);
            if (val.empty()) continue;  // treat empty as missing
            
            if (col == 0 && HAS_INDEX_COLUMN) {
                // Skip first column only if it's an index
                col = 0;  // don't increment
                continue;
            }
            
            if (col < FIXED_LENGTH1) {
                buf[col++] = std::stof(val);
            }
        }
        // ===================================================================

        // ---- Handle short rows ----
        if (col < FIXED_LENGTH1) {
            if (!PAD_SHORT_ROWS) {
                std::cout << "[readCSV] line " << line_no << ": only "
                          << col << " values (need " << FIXED_LENGTH1
                          << "); skipping row.\n";
                continue;
            } else {
                float fill = (col > 0) ? buf[col - 1] : 0.0f;
                for (int k = col; k < FIXED_LENGTH1; ++k) buf[k] = fill;
                std::cout << "[readCSV] line " << line_no << ": padded from "
                          << col << " to " << FIXED_LENGTH1 << ".\n";
            }
        }

        // ---- Skip normalization (already done) ----
        float rmin = buf[0], rmax = buf[0];
        if (APPLY_MINMAX_NORM) {
            for (int k = 1; k < FIXED_LENGTH1; ++k) {
                if (buf[k] < rmin) rmin = buf[k];
                if (buf[k] > rmax) rmax = buf[k];
            }
        }

        // ---- Quantize using model qparams ----
        for (int k = 0; k < FIXED_LENGTH1; ++k) {
            float x_pre;
            if (APPLY_MINMAX_NORM) {
                if (rmax != rmin)
                    x_pre = -2.0f + (buf[k] - rmin) * 4.0f / (rmax - rmin);
                else
                    x_pre = 0.0f;
            } else {
                x_pre = buf[k];  // Use raw value
            }
            row[k] = quantize_input_sample(x_pre);
        }

        // ---- Accept the row ----
        X.push_back(row);
        y.push_back(label);
    }

    // std::cout << "[readCSV] Loaded " << (X.size() - (X.size() - y.size())) 
    //           << " samples from " << filePath << "\n";
    file.close();
}



// void readCSV(const std::string& filePath, int label){
//     std::ifstream file(filePath);
//     if (!file.is_open()) {
//         std::cerr << "Failed to open " << filePath << '\n';
//         return;
//     }

//     std::string line;
//     std::getline(file, line);  // discard header

//     // Set true only if your training pipeline used per-row min–max to [-2, 2]
//     constexpr bool APPLY_MINMAX_NORM = true;

//     while (std::getline(file, line)) {
//         std::istringstream ls(line);

//         float      buf[FIXED_LENGTH1] = {};
//         array180_t row                 = {};
//         int        col = 0;

//         // ---- read 180 floats (skip CSV index column) ----
//         bool first_col = true;
//         std::string val;
//         while (std::getline(ls, val, ',')) {
//             if (first_col) { first_col = false; continue; }
//             val = trim(val);
//             if (val.empty()) continue;
//             if (col < FIXED_LENGTH1) buf[col++] = std::stof(val);
//         }

//         // Handle short/long rows (same as before)
//         if (col < FIXED_LENGTH1) {
//             if (!PAD_SHORT_ROWS) {
//                 std::cout << "Row with " << col << " samples (expected "
//                           << FIXED_LENGTH1 << ") in " << filePath
//                           << " — skipping.\n";
//                 continue;
//             } else {
//                 float fill = (col > 0) ? buf[col-1] : 0.0f;
//                 for (int k = col; k < FIXED_LENGTH1; ++k) buf[k] = fill;
//             }
//         }

//         // ---- Normalize to your desired float range first ----
//         float rmin = buf[0], rmax = buf[0];
//         if (APPLY_MINMAX_NORM) {
//             for (int k = 1; k < FIXED_LENGTH1; ++k) {
//                 if (buf[k] < rmin) rmin = buf[k];
//                 if (buf[k] > rmax) rmax = buf[k];
//             }
//         }

        
//         // ---- Quantize using model qparams (CORRECTION) ----
//         for (int k = 0; k < FIXED_LENGTH1; ++k) {
//             float x_pre;
//             if (APPLY_MINMAX_NORM) {
//                 if (rmax != rmin) {
//                     // Map to [-2, 2] exactly as in your previous code
//                     x_pre = -2.0f + (buf[k] - rmin) * 4.0f / (rmax - rmin);
//                 } else {
//                     x_pre = 0.0f;
//                 }
//             } else {
//                 // If training did not use per-row min–max, pass the raw (or your fixed) normalization
//                 x_pre = buf[k];
//                 // e.g., x_pre = (buf[k] - MEAN) / STD;
//             }

//             // Replace the old cast `(ap_int<8>)x_norm` with proper quantization:
//             row[k] = quantize_input_sample(x_pre);
//         }
//                // ---- Print the entire quantized row BEFORE pushing to X ----
// #ifndef __SYNTHESIS__
//         std::cout << "[readCSV] row (int8): ";
//         for (int j = 0; j < FIXED_LENGTH1; ++j) {
//             std::cout << static_cast<int>(row[j])
//                       << (j + 1 < FIXED_LENGTH1 ? ' ' : '\n');
//         }
// #endif

//         X.push_back(row);
//         y.push_back(label);
//     }

//     file.close();
// }

// /*───────────────────────────────────────────────────────────
//   readCSV() – load one CSV file, normalize each 180-sample row
//   exactly like PyTorch:

//       x_norm = (x − min(row)) / (max(row) − min(row))  ∈ [0,1]

//   Then quantize ONCE to the network’s ingress type:
//       ap_fixed_c  (must be ap_fixed<8,4,AP_RND_CONV,AP_SAT>)

//   Notes:
//   • We clamp tiny numeric drift to [0,1] before casting.
//   • The ap_fixed cast performs convergent (nearest-even) rounding
//     and saturation to Q4.4 automatically.
// ───────────────────────────────────────────────────────────*/

// void readCSV(const std::string& filePath, int label)
// {
//     std::ifstream file(filePath);
//     if (!file.is_open()) {
//         std::cerr << "Failed to open " << filePath << '\n';
//         return;
//     }

//     std::string line;
//     std::getline(file, line);  // discard header

//     while (std::getline(file, line)) {
//         std::istringstream ls(line);

//         float      buf[FIXED_LENGTH1] = {};
//         array180_t row                = {};
//         int        col = 0;

//         // ---- read 180 floats (skip CSV index column) ----
//         bool first_col = true;
//         std::string val;
//         while (std::getline(ls, val, ',')) {
//             if (first_col) { first_col = false; continue; }   // skip index
//             val = trim(val);
//             if (val.empty()) continue;                        // skip empties
//             if (col < FIXED_LENGTH1) buf[col++] = std::stof(val);
//         }

//         // Handle short/long rows
//         if (col < FIXED_LENGTH1) {
//             if (!PAD_SHORT_ROWS) {
//                 std::cerr << "Row with " << col << " samples (expected "
//                           << FIXED_LENGTH1 << ") in " << filePath
//                           << " — skipping.\n";
//                 continue;   // SKIP this row
//             } else {
//                 // PAD: repeat last value (or choose 0.0f / mean)
//                 float fill = (col > 0) ? buf[col-1] : 0.0f;
//                 for (int k = col; k < FIXED_LENGTH1; ++k) buf[k] = fill;
//             }
//         } else if (col > FIXED_LENGTH1) {
//             // Truncate extras defensively
//             col = FIXED_LENGTH1;
//         }

//        //here >>>>> // ---- per-row min–max normalization ----
//         // float rlow = -4.0f, rhigh = 4.0f;
//         // float rlow = -3.0f, rhigh = 3.0f;
//         float rlow = -2f, rhigh = 2f;
//         float rmin = buf[0], rmax = buf[0];
//         for (int k = 1; k < FIXED_LENGTH1; ++k) {
//             float v = buf[k];
//             if (v < rmin) rmin = v;
//             if (v > rmax) rmax = v;
//         }

        
//         //here too >>>>>> ---- cast ONCE to Q4.4 (must be ap_fixed<8,4,AP_RND_CONV,AP_SAT>) ----
//         for (int k = 0; k < FIXED_LENGTH1; ++k) {
//             float x_norm;
//             if (rmax != rmin) x_norm = rlow + (buf[k] - rmin) * (rhigh-rlow) / (rmax - rmin);
//             else              x_norm = 0.0f;       // flat row

//             if (x_norm < rlow) x_norm = rlow;      // clamp
//             if (x_norm > rhigh) x_norm = rhigh;

//             row[k] = (ap_fixed_c)x_norm;           // single QFX cast
//         }

//         X.push_back(row);
//         y.push_back(label);

//         // derive a readable prefix from the source file path
//         std::string stem = basename_stem(filePath);

//         // unique, monotonic index based on how many rows we already have
//         saveRowToCSV(row, "quantized_outputs", stem, X.size());
//     }
//     file.close();
// }




// /*───────────────────────────────────────────────────────────
//   readCSV() – load one CSV file, normalise each 180‑sample row
//   exactly the way your PyTorch preprocessing did:

//         x_norm = (x − min(row)) / (max(row) − min(row))

//   Then the normalised value (∈ [0,1]) is scaled to Q1.6
//   ([0, 1.984375]) and cast to ap_fixed<8,2>.
// ───────────────────────────────────────────────────────────*/
// /* ───────────────────────────────────────────────────────────── */
// void readCSV(const std::string& filePath, int label)
// /* ───────────────────────────────────────────────────────────── */
// {
//     std::ifstream file(filePath);
//     if (!file.is_open()) {
//         std::cerr << "Failed to open " << filePath << '\n';
//         return;
//     }

//     std::string line;
//     std::getline(file, line);          // discard header

//     bool first_column = true;
//     while (std::getline(file, line)) {
//         std::istringstream ls(line);

//         float      buf[FIXED_LENGTH1] = {};
//         array180_t row                = {};
//         int        col = 0;
//         std::string val;

//         /* read 180 floats (skip CSV index column) */
//         while (std::getline(ls, val, ',')) {
//             if (first_column) { first_column = false; continue; }
//             val = trim(val);
//             if (col < FIXED_LENGTH1) buf[col++] = std::stof(val);
//             if (col >= FIXED_LENGTH1) break;
//         }
//         first_column = true;           // reset for next line

//         /* per-row min-max normalisation */
//         float rmin = buf[0], rmax = buf[0];
//         for (int k = 1; k < FIXED_LENGTH1; ++k) {
//             if (buf[k] < rmin) rmin = buf[k];
//             if (buf[k] > rmax) rmax = buf[k];
//         }
// // #ifndef __SYNTHESIS__
// //         static bool printed_float_row = false;
// //         static float first_row_float[FIXED_LENGTH1];
// // #endif

//         for (int k = 0; k < FIXED_LENGTH1; ++k) {
//             float x_norm = (rmax != rmin)
//                          ? (buf[k] - rmin) / (rmax - rmin)
//                          : buf[k];

//             /* float → signed-8-bit integer, round + saturate */
//             int qi = static_cast<int>(std::lround(x_norm * INV_STEP));
//             if (qi >  127) qi =  127;
//             if (qi < -128) qi = -128;
// // #ifndef __SYNTHESIS__
// //         if (!printed_float_row) first_row_float[k] = x_norm;
// // #endif

//             /* signed int → Q4.4 ap_fixed value */
//             row[k] = ap_fixed_c(qi) * STEP_Q;
//         }


// // #ifndef __SYNTHESIS__
// //         if (!printed_float_row) {
// //             printed_float_row = true;
// //             std::cout << "\nROW‑0  (float after normalise) ↓\n";
// //             for (int p = 0; p < FIXED_LENGTH1; ++p) {
// //                 std::cout << std::fixed << std::setprecision(6)
// //                         << first_row_float[p]
// //                         << ((p % 10 == 9) ? "\n" : " ");
// //             }
// //             std::cout << "\n";
// //         }
// // #endif

// // #ifndef __SYNTHESIS__                               // C‑sim only probe
// //         if (row_counter == 0) {
// //             std::cout << "first 8 normalised Q1.6 values of "
// //                       << filePath << " : ";
// //             for (int k = 0; k < 8; ++k) std::cout << row[k] << ' ';
// //             std::cout << '\n';
// //         }
// // #endif

//         X.push_back(row);      // store features
//         y.push_back(label);    // store label
//     }
//     file.close();
// }




// Helper function to trim whitespace from both ends of a string
std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(' ');
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(' ');
    return str.substr(first, last - first + 1);
}

// Helper function to shuffle data and labels together
void shuffleData() {
    // Create a vector of indices and shuffle it
    std::vector<size_t> indices(X.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    // Apply the shuffle to both X and y
    std::vector<array180_t> X_shuffled;
    std::vector<int> y_shuffled;
    for (size_t idx : indices) {
        X_shuffled.push_back(X[idx]);
        y_shuffled.push_back(y[idx]);
    }

    X = std::move(X_shuffled);
    y = std::move(y_shuffled);
}

// Function to extract the "weights" for each layer in the JSON
JsonMap extractLayerWeightsOrDims(const json& jsonData) {
    JsonMap resultMap;

    for (auto it = jsonData.begin(); it != jsonData.end(); ++it) {
        const std::string& layerName = it.key();
        const json& layerContent = it.value();

        // Check if the layer has a "weights" field and store it in the map
        if (layerContent.contains("weights")) {
            resultMap[layerName] = layerContent["weights"];
        } else if (layerContent.contains("dimensions")) {
            resultMap[layerName] = layerContent["dimensions"];
        } else{
            std::cerr << "Warning: Layer " << layerName << " does not contain 'weights'" << std::endl;
        }
    }

    return resultMap;
}


    };

        

#endif // FILE_READER_H

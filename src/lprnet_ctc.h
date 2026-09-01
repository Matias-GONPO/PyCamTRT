#pragma once

// LPRNet output decoding: greedy CTC over the [classes, timesteps] logits
// one engine output slot holds (68 x 18 for the sirius-ai weights).
// Shared by the cascade pipeline and its checkpoint (lprnet_test) so the
// verified code is the shipped code.

#include <string>
#include <vector>

// sirius-ai charset: indices 0-30 are Chinese province glyphs (rendered
// here as their pinyin abbreviations for ASCII logs), then digits,
// letters, and '-' = the CTC blank at index 67.
constexpr int kLprClasses = 68;
constexpr int kLprTimesteps = 18;
constexpr int kLprBlank = kLprClasses - 1;

inline const char* LprCharAscii(int idx) {
    static const char* kTable[kLprClasses] = {
        "<jing>", "<hu>", "<jin>", "<yu>", "<ji>", "<jin2>", "<meng>",
        "<liao>", "<ji2>", "<hei>", "<su>", "<zhe>", "<wan>", "<min>",
        "<gan>", "<lu>", "<yu2>", "<e>", "<xiang>", "<yue>", "<gui>",
        "<qiong>", "<chuan>", "<gui2>", "<yun>", "<zang>", "<shan>",
        "<gan2>", "<qing>", "<ning>", "<xin>",
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "A", "B", "C", "D", "E", "F", "G", "H", "J", "K",
        "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V",
        "W", "X", "Y", "Z", "I", "O", "-"};
    return (idx >= 0 && idx < kLprClasses) ? kTable[idx] : "?";
}

// Greedy CTC: per-timestep argmax, collapse adjacent repeats, drop blanks.
// `logits` layout: [classes, timesteps] (class-major, the engine's output).
inline std::vector<int> CtcGreedyDecode(const float* logits,
                                        int classes = kLprClasses,
                                        int timesteps = kLprTimesteps) {
    std::vector<int> out;
    int prev = -1;
    for (int t = 0; t < timesteps; t++) {
        int best = 0;
        float best_s = logits[t];
        for (int c = 1; c < classes; c++) {
            const float s = logits[c * timesteps + t];
            if (s > best_s) {
                best_s = s;
                best = c;
            }
        }
        if (best != prev && best != classes - 1) out.push_back(best);
        prev = best;
    }
    return out;
}

inline std::string LprLabelString(const std::vector<int>& label) {
    std::string s;
    for (int idx : label) s += LprCharAscii(idx);
    return s;
}

// Reads the evaluator's mark CSV (eval/marks.py): the labels, misses, false
// positives and id switches the scene draws, bucketed by capture time.

#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>
#include "types.hpp"

// What the evaluator says about a box: a labelled object, a missed object, a
// detection that matches nothing, or a track that switched identity.
enum class MarkKind : std::uint8_t { kLabel, kMiss, kFalsePositive, kIdSwitch };

// One marked box, in the vehicle frame of its capture time.
struct Mark {
  MarkKind kind = MarkKind::kLabel;
  Box box;
};

// Marks keyed by the capture time of the frame they belong to.
using MarksByTime = std::unordered_map<std::int64_t, std::vector<Mark>>;

// Reads the CSV at path; throws std::runtime_error on a bad header, row or
// kind, naming the path or the line.
MarksByTime load_marks(const std::string& path);

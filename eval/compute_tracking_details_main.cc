/* Copyright 2019 The Waymo Open Dataset Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Derived from compute_tracking_metrics_main.cc: the same parsing, with
// per-frame match details printed.

// Prints the totals and the per-frame matches of one context. marks.py reads
// this output to draw the per-frame marks.
// Usage:
// /path/to/compute_tracking_details_main pd_filename gt_filename
//
// pd_filename and gt_filename hold waymo::open_dataset::Objects protos. The
// output is tab-separated, one T line per breakdown followed by one F line
// per frame of that breakdown:
//   T  name  num_misses  num_fps  num_mismatches  num_matches  num_objects_gt
//   F  name  timestamp_micros  tp  fn  fp
// tp is a space-separated list of gt_id=pred_id pairs, fn and fp are
// space-separated ids, and an empty list is an empty field.

#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <set>
#include <streambuf>
#include <string>
#include <utility>
#include <vector>

#include "waymo_open_dataset/label.pb.h"
#include "waymo_open_dataset/metrics/config_util.h"
#include "waymo_open_dataset/metrics/tracking_metrics.h"
#include "waymo_open_dataset/protos/breakdown.pb.h"
#include "waymo_open_dataset/protos/metrics.pb.h"

namespace waymo {
namespace open_dataset {
namespace {
using Measure = TrackingMeasurement;
using Details = TrackingMeasurement::Details;

// Generates the stock main's config reduced to one breakdown (OBJECT_TYPE),
// one difficulty level (LEVEL_2) and one score cutoff. Every prediction has
// score 1.0, so a single cutoff gives the same matches as the stock cutoffs.
// The per-frame details are enabled.
Config GetConfig() {
  Config config;
  config.add_breakdown_generator_ids(Breakdown::OBJECT_TYPE);
  auto* d = config.add_difficulties();
  d->add_levels(Label::LEVEL_2);

  config.set_matcher_type(MatcherProto::TYPE_HUNGARIAN);
  config.add_iou_thresholds(0.0);
  config.add_iou_thresholds(0.7);
  config.add_iou_thresholds(0.5);
  config.add_iou_thresholds(0.5);
  config.add_iou_thresholds(0.5);
  config.set_box_type(Label::Box::TYPE_3D);

  config.add_score_cutoffs(0.0);
  config.set_include_details_in_measurements(true);

  return config;
}

// Prints the totals line of one breakdown.
void PrintTotals(const std::string& name, const Measure& m) {
  std::cout << "T\t" << name << '\t' << m.num_misses() << '\t' << m.num_fps()
            << '\t' << m.num_mismatches() << '\t' << m.num_matches() << '\t'
            << m.num_objects_gt() << '\n';
}

// Prints the ids separated by spaces; an empty list prints nothing.
template <typename Ids>
void PrintSpaced(const Ids& ids) {
  bool first = true;
  for (const auto& id : ids) {
    if (!first) std::cout << ' ';
    std::cout << id;
    first = false;
  }
}

// Prints the matches of one frame of one breakdown.
void PrintFrame(const std::string& name, int64_t timestamp, const Details& d) {
  std::cout << "F\t" << name << '\t' << timestamp << '\t';
  for (int j = 0; j < d.tp_gt_ids_size(); ++j) {
    if (j > 0) std::cout << ' ';
    std::cout << d.tp_gt_ids(j) << '=' << d.tp_pred_ids(j);
  }
  std::cout << '\t';
  PrintSpaced(d.fn_gt_ids());
  std::cout << '\t';
  PrintSpaced(d.fp_pred_ids());
  std::cout << '\n';
}

// Computes the per-frame matches by taking serialized prediction objects and
// serialized groundtruth objects. Returns the process exit code.
int Compute(const std::string& pd_str, const std::string& gt_str) {
  Objects pd_objects;
  if (!pd_objects.ParseFromString(pd_str)) {
    std::cerr << "Failed to parse predictions.\n";
    return 1;
  }
  Objects gt_objects;
  if (!gt_objects.ParseFromString(gt_str)) {
    std::cerr << "Failed to parse ground truths.\n";
    return 1;
  }

  std::map<std::string, std::map<int64_t, std::vector<Object>>> pd_map;
  std::map<std::string, std::map<int64_t, std::vector<Object>>> gt_map;
  std::map<std::string, std::set<int64_t>> all_example_keys;
  for (auto& o : *pd_objects.mutable_objects()) {
    all_example_keys[o.context_name()].insert(o.frame_timestamp_micros());
    pd_map[o.context_name()][o.frame_timestamp_micros()].push_back(
        std::move(o));
  }
  for (auto& o : *gt_objects.mutable_objects()) {
    all_example_keys[o.context_name()].insert(o.frame_timestamp_micros());
    gt_map[o.context_name()][o.frame_timestamp_micros()].push_back(
        std::move(o));
  }
  if (all_example_keys.size() != 1) {
    std::cerr << "Expected exactly one context_name, found "
              << all_example_keys.size() << ".\n";
    return 1;
  }

  const auto& [context_name, timestamp_set] = *all_example_keys.begin();
  const std::vector<int64_t> timestamps(timestamp_set.begin(),
                                        timestamp_set.end());
  std::vector<std::vector<Object>> pds;
  std::vector<std::vector<Object>> gts;
  for (int64_t ts : timestamps) {
    pds.push_back(pd_map[context_name][ts]);
    gts.push_back(gt_map[context_name][ts]);
  }

  const Config config = GetConfig();
  const std::vector<TrackingMeasurements> measurements =
      ComputeTrackingMeasurements(config, pds, gts);
  const std::vector<std::string> breakdown_names =
      GetBreakdownNamesFromConfig(config);
  if (breakdown_names.size() != measurements.size()) {
    std::cerr << "Metrics (size of " << measurements.size()
              << ") and breadown (size of " << breakdown_names.size()
              << ") does not match.\n";
    return 1;
  }

  for (int i = 0; i < measurements.size(); ++i) {
    const std::string& name = breakdown_names[i];
    const Measure& m = measurements[i].measurements(0);
    if (m.details_size() != static_cast<int>(timestamps.size())) {
      std::cerr << "Details (size of " << m.details_size()
                << ") and frames (size of " << timestamps.size()
                << ") do not match for " << name << ".\n";
      return 1;
    }
    PrintTotals(name, m);
    for (int f = 0; f < m.details_size(); ++f) {
      PrintFrame(name, timestamps[f], m.details(f));
    }
  }
  return 0;
}

}  // namespace
}  // namespace open_dataset
}  // namespace waymo

namespace {
// Read all content in a file to a string.
std::string ReadFileToString(const char* filename) {
  std::ifstream s(filename);
  const std::string content((std::istreambuf_iterator<char>(s)),
                            std::istreambuf_iterator<char>());
  return content;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0]
              << " prediction_filename ground_truth_filename.\n";
    return 1;
  }

  const char* pd_filename = argv[1];
  const char* gt_filename = argv[2];
  const std::string pd_str = ReadFileToString(pd_filename);
  const std::string gt_str = ReadFileToString(gt_filename);
  if (gt_str.empty()) {
    std::cerr << "Must specify ground truth.\n";
    return 1;
  }

  return waymo::open_dataset::Compute(pd_str, gt_str);
}

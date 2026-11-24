// Runs the pipeline live, one frame at a time: read a staged frame at its
// capture pace, track it, publish its confirmed tracks, take the next.

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <exception>
#include <thread>
#include <utility>
#include <vector>
#include <CLI/CLI.hpp>
#include <CLI/Validators.hpp>
#include <Eigen/Dense>
#include "filter.hpp"
#include "log.hpp"
#include "scene.hpp"
#include "tracker.hpp"
#include "types.hpp"

namespace {

using Clock = std::chrono::steady_clock;
using Micros = std::chrono::microseconds;

// Writes one row per confirmed track per frame, boxes in the frame's own
// vehicle frame, which is what Waymo's evaluator compares against, plus the
// world-frame speed and yaw rate for diagnostics.
class CsvWriter {
 public:
  explicit CsvWriter(const std::string& path) : stream_(path) {
    if (!stream_) throw std::runtime_error("cannot open " + path);
    stream_ << "frame_timestamp_micros,track_id,object_class,center_x,"
               "center_y,center_z,length,width,height,yaw,speed,"
               "yaw_rate\n";
  }

  void write(const Frame& frame, const std::vector<Track>& tracks) {
    const Eigen::Matrix4d vehicle_from_world = frame.vehicle_to_world.inverse();
    const double ego_yaw = vehicle_yaw(frame.vehicle_to_world);
    for (const Track& track : tracks) {
      const Eigen::Vector2d position = track.state.position();
      const Eigen::Vector4d world(position.x(), position.y(),
                                  track.state.center_z, 1.0);
      const Eigen::Vector4d local = vehicle_from_world * world;
      const double yaw = wrap_angle(track.state.yaw() - ego_yaw);
      stream_ << frame.capture_time_micros << ',' << track.track_id << ','
              << static_cast<int>(track.object_class) << ',' << local.x() << ','
              << local.y() << ',' << local.z() << ',' << track.state.length
              << ',' << track.state.width << ',' << track.state.height << ','
              << yaw << ',' << track.state.mean(kStateSpeed) << ','
              << track.state.mean(kStateYawRate) << '\n';
    }
  }

 private:
  std::ofstream stream_;
};

// Holds the loop to the frames' own pace: frame k is released when as much
// wall time has passed since the first frame as capture time has.
class Pacer {
 public:
  void wait_for(std::int64_t capture_time_micros) {
    if (!start_) start_.emplace(Clock::now(), capture_time_micros);
    const Micros elapsed(capture_time_micros - start_->second);
    std::this_thread::sleep_until(start_->first + elapsed);
  }

 private:
  std::optional<std::pair<Clock::time_point, std::int64_t>> start_;
};

// Blocks before the paced loop when the run waits for viewer subscriptions.
void wait_for_viewer(Scene* scene, const SceneOptions& options) {
  if (options.wait_subscriptions == 0) return;
  if (scene == nullptr || !scene->wait_for_subscriptions(
                              options.wait_subscriptions, options.wait_timeout))
    throw std::runtime_error("no viewer subscribed within " +
                             std::to_string(options.wait_timeout.count()) +
                             " ms");
}

// What the read-track-publish loop reads from and writes to. Reader and
// tracker are never null; a null sink is skipped.
struct Pipeline {
  SegmentReader* reader;
  Tracker* tracker;
  bool paced;
  Scene* scene;
  CsvWriter* csv;
};

// Runs the read-track-publish loop over every frame of the segment.
void track_segment(const Pipeline& run) {
  Pacer pacer;
  for (;;) {
    const std::optional<Frame> frame = run.reader->next_frame();
    if (!frame) break;
    if (run.paced) pacer.wait_for(frame->capture_time_micros);

    run.tracker->step(*frame);
    const std::vector<Track> tracks =
        run.tracker->confirmed_at(frame->capture_time_micros);

    if (run.csv != nullptr) run.csv->write(*frame, tracks);
    if (run.scene != nullptr) run.scene->publish(*frame, tracks);
  }
}

template <ObjectClass ClassValue>
void bind_priors(CLI::App& app, TrackerSettings& cfg, const char* name) {
  constexpr std::size_t index = class_index(ClassValue);
  app.add_option(std::string("--acceleration-sigma-") + name,
                 cfg.noise.priors[index].acceleration_sigma,
                 std::string(name) + " linear acceleration sigma, m/s^2")
      ->check(CLI::PositiveNumber);
}

// An empty prefix is the unprefixed arm; otherwise it is a topic path such as
// "/arm" with no trailing slash.
bool is_valid_topic_prefix(std::string_view prefix) {
  if (prefix.empty()) return true;
  return prefix.front() == '/' && prefix.back() != '/';
}

int run(int argument_count, char* const* arguments) {
  CLI::App app{"Live multi-object tracker on a staged Waymo segment"};
  std::string segment_path;
  std::string csv_path;
  SceneOptions scene_options;
  bool paced = false;
  TrackerSettings settings;
  app.add_option("--segment", segment_path, "staged .mcap segment to track")
      ->required();
  app.add_option("--live", scene_options.live_port,
                 "serve the scene to Lichtblick on this port, at the "
                 "frames' own pace");
  app.add_option("--record", scene_options.recording_path,
                 "also write the scene to this .mcap file");
  app.add_option("--topic-prefix", scene_options.topic_prefix,
                 "put the scene on PREFIX/scene; no points or tf");
  app.add_option("--export", csv_path, "write confirmed tracks as CSV");
  app.add_option("--wait-subscriptions", scene_options.wait_subscriptions,
                 "block until this many channel subscriptions arrive")
      ->needs("--live");
  app.add_option("--sigma-position", settings.noise.sigma_measurement_position,
                 "measurement position sigma, metres (default: the "
                 "detector's measured error on the segment)");
  app.add_option("--sigma-yaw", settings.noise.sigma_measurement_yaw,
                 "measurement yaw sigma, radians (default: the detector's "
                 "measured error on the segment)");
  app.add_option("--sigma-yaw-acceleration",
                 settings.noise.sigma_yaw_acceleration,
                 "yaw acceleration process noise, rad/s^2")
      ->check(CLI::PositiveNumber);
  bind_priors<ObjectClass::Vehicle>(app, settings, "vehicle");
  bind_priors<ObjectClass::Pedestrian>(app, settings, "pedestrian");
  bind_priors<ObjectClass::Cyclist>(app, settings, "cyclist");
  CLI11_PARSE(app, argument_count, arguments);
  paced = scene_options.live_port != 0;
  if (!is_valid_topic_prefix(scene_options.topic_prefix))
    throw std::runtime_error(
        "--topic-prefix must start with '/' and not end with '/'");

  SegmentReader reader(segment_path);
  if (app.count("--sigma-position") == 0)
    settings.noise.sigma_measurement_position = reader.info().position_sigma;
  if (app.count("--sigma-yaw") == 0)
    settings.noise.sigma_measurement_yaw = reader.info().yaw_sigma;

  Tracker tracker(settings);
  std::optional<Scene> scene;
  if (paced || !scene_options.recording_path.empty())
    scene.emplace(scene_options, settings.noise);
  std::optional<CsvWriter> csv;
  if (!csv_path.empty()) csv.emplace(csv_path);
  if (paced)
    std::cout << "serving ws://127.0.0.1:" << scene_options.live_port
              << " for Lichtblick\n";

  wait_for_viewer(scene ? &*scene : nullptr, scene_options);

  track_segment({.reader = &reader,
                 .tracker = &tracker,
                 .paced = paced,
                 .scene = scene ? &*scene : nullptr,
                 .csv = csv ? &*csv : nullptr});
  if (scene) scene->close();
  return 0;
}
}  // namespace

int main(int argument_count, char** arguments) {
  try {
    return run(argument_count, arguments);
  } catch (const std::exception& failure) {
    std::cerr << "tracker: " << failure.what() << "\n";
    return 1;
  }
}

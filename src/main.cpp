// Runs the pipeline live, one frame at a time: read a staged frame at its
// capture pace, track it, publish its confirmed tracks, take the next.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <exception>
#include <ratio>
#include <thread>
#include <utility>
#include <vector>
#include <CLI/CLI.hpp>
#include <CLI/Validators.hpp>
#include <Eigen/Dense>
#include "filter.hpp"
#include "log.hpp"
#include "marks.hpp"
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

// Writes one row per confirmed track per frame with the 3 s position forecast
// and its 2x2 covariance in the world frame.
class ForecastCsvWriter {
 public:
  explicit ForecastCsvWriter(const std::string& path) : stream_(path) {
    if (!stream_) throw std::runtime_error("cannot open " + path);
    stream_ << "frame_timestamp_micros,track_id,object_class,x,y,speed,"
               "forecast_x,forecast_y,cov_xx,cov_xy,cov_yy\n";
  }

  void write(std::int64_t frame_timestamp_micros, const Track& track,
             const MotionFilter& filter) {
    const Eigen::Vector2d position = track.state.position();
    const TrackState horizon = forecast(filter, track.state);
    const Eigen::Matrix2d covariance = horizon.covariance.topLeftCorner<2, 2>();
    stream_ << frame_timestamp_micros << ',' << track.track_id << ','
            << static_cast<int>(track.object_class) << ',' << position.x()
            << ',' << position.y() << ',' << track.state.mean(kStateSpeed)
            << ',' << horizon.mean(kStateX) << ',' << horizon.mean(kStateY)
            << ',' << covariance(0, 0) << ',' << covariance(0, 1) << ','
            << covariance(1, 1) << '\n';
  }

 private:
  std::ofstream stream_;
};

// Wall-clock milliseconds per frame for one stage of the pipeline.
struct StageTimes {
  std::vector<double> milliseconds;

  void record(Clock::time_point start, Clock::time_point end) {
    const std::chrono::duration<double, std::milli> elapsed = end - start;
    milliseconds.push_back(elapsed.count());
  }

  // The sample at index floor(fraction * (n - 1)) of the sorted times.
  double percentile(double fraction) const {
    if (milliseconds.empty()) return 0.0;
    std::vector<double> sorted = milliseconds;
    std::ranges::sort(sorted);
    const auto last = static_cast<double>(sorted.size() - 1);
    return sorted[static_cast<std::size_t>(std::floor(fraction * last))];
  }

  std::size_t overruns(double frame_period_milliseconds) const {
    const auto slow = [=](double step) {
      return step > frame_period_milliseconds;
    };
    return static_cast<std::size_t>(std::ranges::count_if(milliseconds, slow));
  }
};

// Read, track and publish times per frame, and their sum, which is the
// latency from a frame being available to its scene being out.
struct FrameTimes {
  StageTimes read;
  StageTimes track;
  StageTimes publish;
  StageTimes frame;

  void close_frame() {
    frame.milliseconds.push_back(read.milliseconds.back() +
                                 track.milliseconds.back() +
                                 publish.milliseconds.back());
  }

  void print(std::ostream& out, double frame_period_milliseconds) const {
    out << "read_p50_ms " << read.percentile(0.5) << " track_p50_ms "
        << track.percentile(0.5) << " publish_p50_ms "
        << publish.percentile(0.5) << " frame_p50_ms " << frame.percentile(0.5)
        << " frame_p99_ms " << frame.percentile(0.99) << " overruns "
        << frame.overruns(frame_period_milliseconds) << " frames "
        << frame.milliseconds.size() << "\n";
  }
};

// One run's per-frame stage times and the capture time of each frame.
struct TrackingResult {
  FrameTimes times;
  std::vector<std::int64_t> capture_times;
};

// Writes the per-frame stage times; throws if the file cannot be opened.
void write_frame_times(const std::string& path, const TrackingResult& run) {
  const FrameTimes& times = run.times;
  std::ofstream out(path, std::ios::app);
  if (!out) throw std::runtime_error("cannot open " + path);
  for (std::size_t frame = 0; frame < times.frame.milliseconds.size(); ++frame)
    out << frame << ',' << run.capture_times.at(frame) << ','
        << times.read.milliseconds.at(frame) << ','
        << times.track.milliseconds.at(frame) << ','
        << times.publish.milliseconds.at(frame) << ','
        << times.frame.milliseconds.at(frame) << '\n';
}

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

// Opens the frame-times file and writes the header so a bad path fails first.
void open_frame_times_or_throw(const std::string& path) {
  std::ofstream probe(path);
  if (!probe) throw std::runtime_error("cannot open " + path);
  probe << "frame,capture_time_micros,read_ms,track_ms,publish_ms,frame_ms\n";
}

// Blocks before the paced loop when the run waits for viewer subscriptions.
void wait_for_viewer(Scene* scene, const SceneOptions& options) {
  if (options.wait_subscriptions == 0) return;
  if (scene == nullptr || !scene->wait_for_subscriptions(
                              options.wait_subscriptions, options.wait_timeout))
    throw std::runtime_error("no viewer subscribed within " +
                             std::to_string(options.wait_timeout.count()) +
                             " ms");
}

// What the read-track-publish loop reads from and writes to. Reader, tracker
// and filter are never null; a null sink is skipped.
struct Pipeline {
  SegmentReader* reader;
  Tracker* tracker;
  bool paced;
  Scene* scene;
  CsvWriter* csv;
  ForecastCsvWriter* forecast_csv;
  const MotionFilter* forecast_filter;
};

// Runs the read-track-publish loop and returns the captured stage times.
TrackingResult track_segment(const Pipeline& run) {
  Pacer pacer;
  TrackingResult result;
  for (;;) {
    const Clock::time_point read_start = Clock::now();
    const std::optional<Frame> frame = run.reader->next_frame();
    if (!frame) break;
    result.times.read.record(read_start, Clock::now());
    if (run.paced) pacer.wait_for(frame->capture_time_micros);

    const Clock::time_point track_start = Clock::now();
    run.tracker->step(*frame);
    const std::vector<Track> tracks =
        run.tracker->confirmed_at(frame->capture_time_micros);
    const Clock::time_point track_end = Clock::now();
    result.times.track.record(track_start, track_end);

    if (run.csv != nullptr) run.csv->write(*frame, tracks);
    if (run.forecast_csv != nullptr) {
      for (const Track& track : tracks)
        run.forecast_csv->write(frame->capture_time_micros, track,
                                *run.forecast_filter);
    }
    if (run.scene != nullptr) run.scene->publish(*frame, tracks);
    result.times.publish.record(track_end, Clock::now());
    result.times.close_frame();
    result.capture_times.push_back(frame->capture_time_micros);
  }
  return result;
}

template <ObjectClass ClassValue>
void bind_priors(CLI::App& app, TrackerSettings& cfg, const char* name) {
  constexpr std::size_t index = class_index(ClassValue);
  app.add_option(std::string("--acceleration-sigma-") + name,
                 cfg.noise.priors[index].acceleration_sigma,
                 std::string(name) + " linear acceleration sigma, m/s^2")
      ->check(CLI::PositiveNumber);
  app.add_option(std::string("--position-floor-") + name,
                 cfg.noise.priors[index].position_floor_sigma,
                 std::string(name) + " forecast position floor sigma, m")
      ->check(CLI::NonNegativeNumber);
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
  std::string forecast_path;
  std::string frame_times_path;
  std::string marks_path;
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
                 "put the scene and marks on PREFIX/scene and PREFIX/marks; "
                 "no points or tf");
  app.add_option("--marks", marks_path,
                 "draw the evaluator's marks from this CSV (eval/marks.py)");
  app.add_option("--export", csv_path, "write confirmed tracks as CSV");
  app.add_option("--export-forecast", forecast_path,
                 "write 3 s forecast covariances as CSV");
  app.add_option("--frame-times", frame_times_path,
                 "write per-frame stage times as CSV");
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
  if (!marks_path.empty()) scene_options.marks = load_marks(marks_path);

  if (!frame_times_path.empty()) open_frame_times_or_throw(frame_times_path);

  SegmentReader reader(segment_path);
  if (app.count("--sigma-position") == 0)
    settings.noise.sigma_measurement_position = reader.info().position_sigma;
  if (app.count("--sigma-yaw") == 0)
    settings.noise.sigma_measurement_yaw = reader.info().yaw_sigma;

  Tracker tracker(settings);
  const MotionFilter forecast_filter(settings.noise);
  std::optional<Scene> scene;
  if (paced || !scene_options.recording_path.empty())
    scene.emplace(scene_options, settings.noise);
  std::optional<CsvWriter> csv;
  if (!csv_path.empty()) csv.emplace(csv_path);
  std::optional<ForecastCsvWriter> forecast_csv;
  if (!forecast_path.empty()) forecast_csv.emplace(forecast_path);
  if (paced)
    std::cout << "serving ws://127.0.0.1:" << scene_options.live_port
              << " for Lichtblick\n";

  wait_for_viewer(scene ? &*scene : nullptr, scene_options);

  const TrackingResult result =
      track_segment({.reader = &reader,
                     .tracker = &tracker,
                     .paced = paced,
                     .scene = scene ? &*scene : nullptr,
                     .csv = csv ? &*csv : nullptr,
                     .forecast_csv = forecast_csv ? &*forecast_csv : nullptr,
                     .forecast_filter = &forecast_filter});
  if (scene) scene->close();

  if (!frame_times_path.empty()) write_frame_times(frame_times_path, result);
  const std::vector<std::int64_t>& capture_times = result.capture_times;
  const double frame_period_milliseconds =
      capture_times.size() >= 2
          ? static_cast<double>(capture_times[1] - capture_times[0]) / 1000.0
          : 100.0;
  result.times.print(std::cout, frame_period_milliseconds);
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

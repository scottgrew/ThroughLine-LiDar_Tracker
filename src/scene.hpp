// Declares the publisher that streams each frame's lidar points and
// confirmed tracks to a live Lichtblick viewer and, when asked, to an .mcap.

#pragma once
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include "filter.hpp"
#include "marks.hpp"
#include "types.hpp"

using Milliseconds = std::chrono::milliseconds;

// Counts channel subscriptions so a paced run can wait for a viewer.
struct Subscriptions {
  // Increments the count and wakes any thread blocked on a subscription wait.
  void record();
  // Blocks until count subscriptions arrive or the timeout passes.
  bool wait_until(std::size_t count, Milliseconds timeout);

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  std::size_t count_ = 0;
};

// Where the scene goes. A port of zero means no live server, and an empty
// path means no recording.
struct SceneOptions {
  std::uint16_t live_port = 0;
  std::string recording_path;
  std::size_t wait_subscriptions = 0;
  Milliseconds wait_timeout{30000};
  // Goes in front of /scene and /marks. An arm with a prefix publishes no
  // /points or /tf, which the unprefixed arm carries.
  std::string topic_prefix;
  // The evaluator's marks keyed by capture time, drawn on /marks.
  MarksByTime marks;
};

// Owns the live server and the recording. publish() sends one frame to both;
// close() finishes the recording and stops the server.
class Scene {
 public:
  Scene(const SceneOptions& options, FilterNoise noise);
  ~Scene();
  Scene(const Scene&) = delete;
  Scene& operator=(const Scene&) = delete;
  Scene(Scene&&) = default;
  Scene& operator=(Scene&&) = default;
  void publish(const Frame& frame, const std::vector<Track>& tracks);
  void close();
  // Blocks until count channel subscriptions have arrived or the timeout
  // passes; true if they arrived. Returns true at once when count is 0.
  bool wait_for_subscriptions(std::size_t count, Milliseconds timeout);

 private:
  struct Sinks;
  MotionFilter filter_;
  std::unique_ptr<Sinks> sinks_;
  // Frame 0's position, subtracted from every published coordinate so the
  // scene stays within float precision.
  Eigen::Vector3d origin_ = Eigen::Vector3d::Zero();
  bool has_origin_ = false;
  std::vector<std::uint64_t> drawn_track_ids_;
  // Packed x, y, z and colour per point, kept so no frame allocates it.
  std::vector<float> packed_points_;
  // The marks from SceneOptions, copied so each frame looks its own time up.
  MarksByTime marks_;
};

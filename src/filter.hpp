// Declares the extended Kalman filter each track runs over x, y, yaw,
// speed and yaw rate. tracker.cpp and recording.cpp hold one each.

#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <Eigen/Dense>
#include "types.hpp"

// What the filter assumes about an object class before it has seen it
// move: how fast it may already be going, how fast it may be turning, and
// how hard it may accelerate. Speed in m/s, yaw rate in rad/s,
// acceleration in m/s^2. Speed and yaw-rate sigmas are chosen. The
// acceleration sigmas are measured: on 23 validation segments they bring
// the mean NEES of the 3 s forecast against the labels (2 degrees of
// freedom, target 2) to 1.92 for moving vehicles, 1.91 for pedestrians
// and 1.71 for cyclists, and leave-one-segment-out picks the same values
// in 22 of 23 folds (6 of 6 for cyclists). See
// stage/forecast_consistency.py.
struct ClassPrior {
  double speed_sigma;
  double yaw_rate_sigma;
  double acceleration_sigma;
  // The detector's persistent box-centre offset from the label, in metres.
  // The filter cannot average a fixed offset away, so the forecast adds
  // this sigma squared to each position variance. 0 until measured.
  double position_floor_sigma = 0.0;
};

constexpr ClassPrior kVehiclePrior{
    .speed_sigma = 10.0, .yaw_rate_sigma = 0.3, .acceleration_sigma = 3.0};
constexpr ClassPrior kPedestrianPrior{
    .speed_sigma = 2.0, .yaw_rate_sigma = 1.0, .acceleration_sigma = 2.0};
constexpr ClassPrior kCyclistPrior{
    .speed_sigma = 5.0, .yaw_rate_sigma = 0.3, .acceleration_sigma = 2.0};

constexpr std::size_t class_index(ObjectClass object_class) {
  switch (object_class) {
    case ObjectClass::Vehicle:
      return 0;
    case ObjectClass::Pedestrian:
      return 1;
    case ObjectClass::Cyclist:
      return 2;
  }
  return 0;
}

// Process and measurement noise the filter uses to weigh a
// prediction against a measurement.
struct FilterNoise {
  // Process noise is the sigma of unmodelled yaw acceleration in
  // rad/s^2; linear acceleration comes from the class prior. Measured with
  // the class priors' acceleration sigmas (0.5 of 0.5, 1 and 2).
  double sigma_yaw_acceleration = 0.5;
  // Measurement sigmas are the box jitter stage_segment.py measured for
  // the segment, in metres and radians. MotionFilter refuses <= 0.
  double sigma_measurement_position = 0.0;
  double sigma_measurement_yaw = 0.0;
  // What the filter assumes per class before it has seen an object move;
  // indexed by class_index. Defaults are the priors above.
  std::array<ClassPrior, 3> priors{kVehiclePrior, kPedestrianPrior,
                                   kCyclistPrior};
};

// How far ahead the scene draws each track's forecast, and the spacing of
// its points. Chosen.
constexpr std::int64_t kForecastHorizonMicros = 3'000'000;
constexpr std::int64_t kForecastStepMicros = 100'000;

// Below this yaw rate the motion model treats the path as straight,
// avoiding a divide-by-zero in the curved-path equations in filter.cpp.
constexpr double kStraightLineYawRate = 1e-4;

double wrap_angle(double radians);

// Constant-turn-rate extended Kalman filter. It holds the noise, so each
// call takes only the state it works on.
class MotionFilter {
 public:
  explicit MotionFilter(FilterNoise noise);

  // A state at the measured box with speed and yaw rate unknown to the
  // width of the class prior; the caller stamps its time.
  TrackState start_state(const Detection& detection) const;

  // Moves the state forward to time_micros; a state already at or past
  // that time is left alone.
  void advance(TrackState& state, std::int64_t time_micros) const;
  void predict(TrackState& state, double dt_seconds) const;

  // Squared Mahalanobis distance between the state and a measured box.
  [[nodiscard]] double distance_squared(const TrackState& state, Box box) const;
  void update(TrackState& state, Box box) const;

 private:
  FilterNoise noise_;
  Eigen::Matrix3d measurement_noise_;
};

// The state kForecastHorizonMicros ahead, advanced in kForecastStepMicros
// steps (the same path the scene draws). The position-floor variance is
// added once at the horizon, to the position block of the covariance only.
TrackState forecast(const MotionFilter& filter, TrackState state);

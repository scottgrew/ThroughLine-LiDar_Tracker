// Tests that subscription waits return on schedule and when record() runs.

#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <thread>
#include "scene.hpp"

namespace {

constexpr std::uint16_t kTestLivePort = 19'876;
constexpr FilterNoise kSceneTestNoise{.sigma_yaw_acceleration = 0.5,
                                      .sigma_measurement_position = 0.1,
                                      .sigma_measurement_yaw = 0.02};

}  // namespace

TEST_CASE("wait_for_subscriptions with count zero", "[subscriptions]") {
  SceneOptions options;
  Scene scene(options, kSceneTestNoise);
  REQUIRE(scene.wait_for_subscriptions(0, std::chrono::milliseconds(1)));
}

TEST_CASE("subscription wait times out with no client", "[subscriptions]") {
  SceneOptions options{.live_port = kTestLivePort};
  Scene scene(options, kSceneTestNoise);
  const auto start = std::chrono::steady_clock::now();
  REQUIRE_FALSE(scene.wait_for_subscriptions(1, std::chrono::milliseconds(50)));
  const auto elapsed = std::chrono::steady_clock::now() - start;
  REQUIRE(elapsed >= std::chrono::milliseconds(50));
  REQUIRE(elapsed < std::chrono::seconds(1));
}

TEST_CASE("wait_for_subscriptions succeeds after record", "[subscriptions]") {
  Subscriptions subscriptions;
  std::thread helper([&subscriptions] {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    subscriptions.record();
    subscriptions.record();
  });
  REQUIRE(subscriptions.wait_until(2, std::chrono::seconds(2)));
  helper.join();
}
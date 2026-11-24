// Tests that the evaluator's mark CSV loads into per-time buckets and that a
// bad file is rejected with the path or line that is wrong.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include "marks.hpp"

namespace {

constexpr char kHeader[] =
    "frame_timestamp_micros,kind,object_type,object_id,center_x,center_y,"
    "center_z,length,width,height,yaw";

// A CSV in the temporary directory, removed when the test ends.
class TempCsv {
 public:
  explicit TempCsv(const std::string& contents) : path_(make_path()) {
    std::ofstream out(path_);
    out << contents;
  }
  ~TempCsv() { std::filesystem::remove(path_); }
  TempCsv(const TempCsv&) = delete;
  TempCsv& operator=(const TempCsv&) = delete;

  const std::string& path() const { return path_; }

 private:
  static std::string make_path() {
    const auto ticks =
        std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path name =
        "tracker_marks_" + std::to_string(ticks) + ".csv";
    return (std::filesystem::temp_directory_path() / name).string();
  }

  std::string path_;
};

}  // namespace

TEST_CASE("load_marks buckets each row by its capture time", "[marks]") {
  const TempCsv csv(
      std::string(kHeader) +
      "\n"
      "1000000,label,1,abc_-9,1.25,-2.0,0.5,4.5,2.0,1.5,0.25\n"
      "1000000,miss,1,7,3.5,4.0,0.5,4.0,2.0,1.5,-1.0\n"
      "2000000,false_positive,2,x-Y_0,0.0,0.0,0.0,1.0,1.0,1.0,0.0\n"
      "3000000,id_switch,1,12,-1.0,2.5,0.75,2.0,1.0,0.5,3.0\n");

  const MarksByTime marks = load_marks(csv.path());

  REQUIRE(marks.size() == 3);
  REQUIRE(marks.at(1'000'000).size() == 2);
  const Mark& label = marks.at(1'000'000)[0];
  REQUIRE(label.kind == MarkKind::kLabel);
  REQUIRE(label.box.center_x == 1.25);
  REQUIRE(label.box.center_y == -2.0);
  REQUIRE(label.box.center_z == 0.5);
  REQUIRE(label.box.length == 4.5);
  REQUIRE(label.box.width == 2.0);
  REQUIRE(label.box.height == 1.5);
  REQUIRE(label.box.yaw == 0.25);

  const Mark& miss = marks.at(1'000'000)[1];
  REQUIRE(miss.kind == MarkKind::kMiss);
  REQUIRE(miss.box.center_x == 3.5);
  REQUIRE(miss.box.center_y == 4.0);
  REQUIRE(miss.box.center_z == 0.5);
  REQUIRE(miss.box.length == 4.0);
  REQUIRE(miss.box.width == 2.0);
  REQUIRE(miss.box.height == 1.5);
  REQUIRE(miss.box.yaw == -1.0);

  const Mark& false_positive = marks.at(2'000'000)[0];
  REQUIRE(false_positive.kind == MarkKind::kFalsePositive);
  REQUIRE(false_positive.box.length == 1.0);

  const Mark& id_switch = marks.at(3'000'000)[0];
  REQUIRE(id_switch.kind == MarkKind::kIdSwitch);
  REQUIRE(id_switch.box.center_x == -1.0);
  REQUIRE(id_switch.box.center_y == 2.5);
  REQUIRE(id_switch.box.center_z == 0.75);
  REQUIRE(id_switch.box.length == 2.0);
  REQUIRE(id_switch.box.width == 1.0);
  REQUIRE(id_switch.box.height == 0.5);
  REQUIRE(id_switch.box.yaw == 3.0);
}

TEST_CASE("load_marks rejects a wrong header and names the path", "[marks]") {
  const TempCsv csv("frame_timestamp_micros,kind\n");
  REQUIRE_THROWS_WITH(load_marks(csv.path()),
                      Catch::Matchers::ContainsSubstring(csv.path()));
}

TEST_CASE("load_marks rejects an unknown kind and names the line", "[marks]") {
  const TempCsv csv(std::string(kHeader) +
                    "\n"
                    "1000000,ghost,1,7,3.5,4.0,0.5,4.0,2.0,1.5,-1.0\n");
  REQUIRE_THROWS_WITH(load_marks(csv.path()),
                      Catch::Matchers::ContainsSubstring("line 2") &&
                          Catch::Matchers::ContainsSubstring("ghost"));
}

TEST_CASE("load_marks rejects a row with the wrong field count", "[marks]") {
  const TempCsv csv(std::string(kHeader) +
                    "\n"
                    "1000000,miss,1,7,3.5,4.0,0.5,4.0,2.0,1.5\n");
  REQUIRE_THROWS_AS(load_marks(csv.path()), std::runtime_error);
}

// Parses the evaluator's mark CSV row by row, so each row is checked on its
// own line and a bad file names where it went wrong.

#include "marks.hpp"

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <locale>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>
#include "types.hpp"

namespace {

constexpr std::string_view kHeader =
    "frame_timestamp_micros,kind,object_type,object_id,center_x,center_y,"
    "center_z,length,width,height,yaw";
constexpr std::size_t kFieldCount = 11;
constexpr std::size_t kTimeField = 0;
constexpr std::size_t kKindField = 1;
constexpr std::size_t kCenterXField = 4;
constexpr std::size_t kCenterYField = 5;
constexpr std::size_t kCenterZField = 6;
constexpr std::size_t kLengthField = 7;
constexpr std::size_t kWidthField = 8;
constexpr std::size_t kHeightField = 9;
constexpr std::size_t kYawField = 10;

std::vector<std::string_view> split_fields(std::string_view line) {
  std::vector<std::string_view> fields;
  std::size_t start = 0;
  for (;;) {
    const std::size_t comma = line.find(',', start);
    if (comma == std::string_view::npos) {
      fields.push_back(line.substr(start));
      return fields;
    }
    fields.push_back(line.substr(start, comma - start));
    start = comma + 1;
  }
}

std::runtime_error not_a_number(std::string_view text, std::size_t line) {
  return std::runtime_error("line " + std::to_string(line) +
                            ": not a number: '" + std::string(text) + "'");
}

std::int64_t parse_integer(std::string_view text, std::size_t line_number) {
  std::int64_t value = 0;
  const char* first = std::to_address(text.begin());
  const char* last = std::to_address(text.end());
  const auto [stop, error] = std::from_chars(first, last, value);
  if (error != std::errc() || stop != last)
    throw not_a_number(text, line_number);
  return value;
}

// std::from_chars for floating point is unavailable on the deployment target,
// so doubles are read through a stream in the classic locale, which is the
// same no-locale parse.
double parse_double(std::string_view text, std::size_t line_number) {
  std::istringstream stream{std::string(text)};
  stream.imbue(std::locale::classic());
  double value = 0.0;
  stream >> value;
  // Reading the whole text sets eofbit; trailing characters leave it clear.
  if (stream.fail() || !stream.eof()) throw not_a_number(text, line_number);
  return value;
}

MarkKind parse_kind(std::string_view text, std::size_t line_number) {
  if (text == "label") return MarkKind::kLabel;
  if (text == "miss") return MarkKind::kMiss;
  if (text == "false_positive") return MarkKind::kFalsePositive;
  if (text == "id_switch") return MarkKind::kIdSwitch;
  throw std::runtime_error("line " + std::to_string(line_number) +
                           ": unknown mark kind '" + std::string(text) + "'");
}

// The capture time and the mark of one data row.
using TimedMark = std::pair<std::int64_t, Mark>;

TimedMark parse_row(std::string_view row, std::size_t line_number) {
  const std::vector<std::string_view> fields = split_fields(row);
  if (fields.size() != kFieldCount)
    throw std::runtime_error("line " + std::to_string(line_number) +
                             ": expected " + std::to_string(kFieldCount) +
                             " fields, found " + std::to_string(fields.size()));
  const std::int64_t time = parse_integer(fields[kTimeField], line_number);
  Mark mark;
  mark.kind = parse_kind(fields[kKindField], line_number);
  mark.box = Box{
      .center_x = parse_double(fields[kCenterXField], line_number),
      .center_y = parse_double(fields[kCenterYField], line_number),
      .center_z = parse_double(fields[kCenterZField], line_number),
      .length = parse_double(fields[kLengthField], line_number),
      .width = parse_double(fields[kWidthField], line_number),
      .height = parse_double(fields[kHeightField], line_number),
      .yaw = parse_double(fields[kYawField], line_number),
  };
  return {time, mark};
}

}  // namespace

MarksByTime load_marks(const std::string& path) {
  std::ifstream stream(path);
  if (!stream) throw std::runtime_error("cannot open " + path);
  std::string line;
  if (!std::getline(stream, line) || line != kHeader)
    throw std::runtime_error("marks file " + path + " has header '" + line +
                             "', expected '" + std::string(kHeader) + "'");
  MarksByTime marks;
  std::size_t line_number = 1;
  while (std::getline(stream, line)) {
    ++line_number;
    const auto [time, mark] = parse_row(line, line_number);
    marks[time].push_back(mark);
  }
  return marks;
}

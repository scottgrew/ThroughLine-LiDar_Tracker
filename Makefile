SEGMENT ?= 10203656353524179475_7625_000_7645_000
DATA    ?= $(HOME)/waymo-data
SEGMENTS ?= $(basename $(notdir $(wildcard $(DATA)/staged/*.mcap)))
SEED    ?= 0
PYTHON  ?= $(HOME)/venvs/tracker/bin/python
LOG     ?= $(DATA)/staged/$(SEGMENT).mcap
DETECTIONS ?= $(DATA)/casa/data/waymo/detections/val/result.pkl
TRACKS  ?= tracks.csv
FORECAST ?= forecast.csv
TRACKER_FLAGS ?=
MCAP    ?= demo.mcap
PORT    ?= 8765

RUFF       ?= $(dir $(shell command -v $(PYTHON)))ruff
CLANG_TIDY ?= $(firstword $(wildcard \
  $(HOME)/.vscode/extensions/ms-vscode.cpptools-*/LLVM/bin/clang-tidy) \
  clang-tidy)
CPP_FILES  := $(wildcard src/*.cpp src/*.hpp tests/*.cpp tests/*.hpp)
SIMD_MACROS := MMX SSE SSE2 SSE3 SSSE3 SSE4_1 SSE4_2 PCLMUL AVX AVX2
TIDY_FLAGS := --quiet -p build \
  --extra-arg=-isysroot$(shell xcrun --show-sdk-path) \
  --extra-arg=-resource-dir=$(shell clang -print-resource-dir) \
  --extra-arg=-DEIGEN_DONT_VECTORIZE \
  --extra-arg=-isystem/usr/local/include \
  $(foreach m,$(SIMD_MACROS),--extra-arg=-U__$(m)__)
RUFF_FLAGS := --line-length 80 --select E,F,W,I,N,UP,B \
  --extend-exclude eval/generated

RUNS        ?= 5
LATENCY_DIR ?= latency
SHARD       ?= $(HOME)/data/womd/validation.tfrecord-00000-of-00150
LIMIT       ?=

.PHONY: help build test check stage stage-emulated stage-womd detector-error \
        donor-library acceptance-pooled live run record consistency official \
        latency tview clean

help:
	@echo "make build     compile the tracker and the tests"
	@echo "make test      build, then run the tests"
	@echo "make check     clang-format, clang-tidy, ruff and one-line headers"
	@echo "make stage     parquet + CasA detections -> $(LOG)"
	@echo "make stage-emulated  labels + emulate -> $(DATA)/staged-emulated/$(SEGMENT).mcap"
	@echo "make stage-womd      WOMD shard + emulate -> $(DATA)/staged-womd/"
	@echo "make detector-error  CasA error table -> $(DATA)/detector_error.txt"
	@echo "make donor-library   CasA rows -> stage/casa_donor_library.npz"
	@echo "make acceptance-pooled  donor gate -> $(ACCEPTANCE)"
	@echo "make live      track the segment live on ws://127.0.0.1:$(PORT)"
	@echo "make run       track the segment flat out, write $(TRACKS)"
	@echo "make record    track the segment live, also write $(MCAP)"
	@echo "make tview     three-arm tracking view -> $(DATA)/tview/$(SEGMENT)/tview.mcap"
	@echo "make latency   baseline and viewer frame-times captures ($(RUNS) runs)"
	@echo "make official  Waymo's evaluator on $(TRACKS) (needs Docker)"
	@echo "make clean     remove compiled objects, keep fetched dependencies"

build/CMakeCache.txt:
	cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

build: build/CMakeCache.txt
	cmake --build build -j2

WOMD_PROTOS := stage/womd_protos/scenario_pb2.py stage/womd_protos/map_pb2.py

stage/womd_protos/scenario_pb2.py stage/womd_protos/map_pb2.py: \
    proto/womd_protos/scenario.proto proto/womd_protos/map.proto
	protoc --python_out=stage -I proto \
	  proto/womd_protos/scenario.proto proto/womd_protos/map.proto
	touch stage/womd_protos/__init__.py

test: build stage/tracker_pb2.py $(WOMD_PROTOS)
	./build/tracker_tests
	for module in tests/test_*.py; do \
	  PYTHONPATH=stage:tests:eval $(PYTHON) -c "import inspect, $$(basename $$module .py) as m; \
	    [f() for n, f in inspect.getmembers(m, inspect.isfunction) if n.startswith('test_')]" \
	    || exit 1; \
	done
	@echo "python tests passed"

check: build
	clang-format --dry-run --Werror $(CPP_FILES) eval/*.cc
	$(CLANG_TIDY) $(TIDY_FLAGS) src/*.cpp
	$(RUFF) check $(RUFF_FLAGS) stage eval
	@! grep -nE '^[A-Za-z].*\(.*[,(]$$' $(CPP_FILES) || \
	  (echo "function header spans lines" && false)
	@! grep -nE '^  [A-Za-z].*\(.*[,(]$$' src/*.hpp tests/*.hpp || \
	  (echo "member header spans lines" && false)
	@! grep -nE '^ *def .*[^:]$$' stage/*.py eval/*.py || \
	  (echo "python header spans lines" && false)

stage/tracker_pb2.py: proto/tracker.proto
	protoc --python_out=stage -I proto proto/tracker.proto

$(LOG): stage/tracker_pb2.py
	mkdir -p $(dir $(LOG))
	$(PYTHON) stage/stage_segment.py --parquet-root $(DATA)/parquet \
	  --segment $(SEGMENT) --detections $(DETECTIONS) --out $(LOG)

stage: $(LOG)

$(DATA)/staged-emulated/$(SEGMENT).mcap: stage/tracker_pb2.py
	mkdir -p $(dir $@)
	$(PYTHON) stage/stage_segment.py --parquet-root $(DATA)/parquet \
	  --segment $(SEGMENT) --emulate $(SEED) --out $@

stage-emulated: $(DATA)/staged-emulated/$(SEGMENT).mcap

stage-womd: stage/tracker_pb2.py $(WOMD_PROTOS)
	mkdir -p $(DATA)/staged-womd/
	$(PYTHON) stage/stage_womd.py --shard $(SHARD) --out $(DATA)/staged-womd/ \
	  --seed $(SEED) $(if $(LIMIT),--limit $(LIMIT),)

stage/casa_error_model.json: stage/tracker_pb2.py
	$(PYTHON) stage/detector_error.py --parquet-root $(DATA)/parquet \
	  --detections $(DETECTIONS) --model-out $@ $(SEGMENTS) \
	  > $(DATA)/detector_error.txt

detector-error: stage/casa_error_model.json

stage/casa_donor_library.npz: stage/tracker_pb2.py
	$(PYTHON) stage/donor_library.py --parquet-root $(DATA)/parquet \
	  --detections $(DETECTIONS) --out $@ $(SEGMENTS)

donor-library: stage/casa_donor_library.npz

SEEDS ?= 0 1 2
ACCEPTANCE ?= $(DATA)/acceptance_pooled.json

acceptance-pooled: build
ifndef EVALUATOR
	$(error EVALUATOR is not set)
endif
	$(PYTHON) stage/pooled_acceptance.py --parquet-root $(DATA)/parquet \
	  --detections $(DETECTIONS) --segments $(SEGMENTS) \
	  --seeds $(SEEDS) --acceptance $(ACCEPTANCE)

live: build $(LOG)
	./build/tracker --segment $(LOG) --live $(PORT)

run: build $(LOG)
	./build/tracker --segment $(LOG) --export $(TRACKS)

consistency: build $(LOG)
	./build/tracker --segment $(LOG) --export-forecast $(FORECAST) \
	  $(TRACKER_FLAGS)
	$(PYTHON) stage/forecast_consistency.py --segment $(LOG) \
	  --parquet-root $(DATA)/parquet --forecast $(FORECAST)

record: build $(LOG)
	./build/tracker --segment $(LOG) --live $(PORT) --record $(MCAP)

latency: build $(LOG)
	@if [ "$(RUNS)" -lt 5 ]; then \
	  echo "latency harness needs at least 5 runs (RUNS=$(RUNS))" && false; \
	fi
	@mkdir -p $(LATENCY_DIR)
	@i=1; while [ "$$i" -le "$(RUNS)" ]; do \
	  ./build/tracker --segment $(LOG) --live $(PORT) \
	    --frame-times $(LATENCY_DIR)/baseline-$$i.csv > /dev/null || exit $$?; \
	  i=$$((i + 1)); \
	done
	@i=1; while [ "$$i" -le "$(RUNS)" ]; do \
	  node eval/viewer_client.mjs --port $(PORT) \
	    --out $(LATENCY_DIR)/client-$$i.json & \
	  client_pid=$$!; \
	  ./build/tracker --segment $(LOG) --live $(PORT) \
	    --wait-subscriptions 3 \
	    --frame-times $(LATENCY_DIR)/viewer-$$i.csv > /dev/null; \
	  tracker_status=$$?; \
	  wait $$client_pid; client_status=$$?; \
	  if [ $$tracker_status -ne 0 ] || [ $$client_status -ne 0 ]; then exit 1; fi; \
	  i=$$((i + 1)); \
	done
	$(PYTHON) eval/latency_summary.py \
	  --baseline $$(for i in $$(seq 1 $(RUNS)); do printf '%s ' "$(LATENCY_DIR)/baseline-$$i.csv"; done) \
	  --viewer $$(for i in $$(seq 1 $(RUNS)); do printf '%s ' "$(LATENCY_DIR)/viewer-$$i.csv"; done)

official:
	PYTHON=$(PYTHON) PARQUET_ROOT=$(DATA)/parquet \
	  eval/run_official.sh $(SEGMENT) $(TRACKS)

tview: build stage/tracker_pb2.py
	SEGMENT=$(SEGMENT) DATA=$(DATA) PYTHON=$(PYTHON) DETECTIONS=$(DETECTIONS) \
	  SEED=$(SEED) eval/tview.sh

clean:
	cmake --build build --target clean

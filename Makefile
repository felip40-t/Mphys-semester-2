PYTHON         := .venv/bin/python
PARSE_SCRIPT   := src/diboson/io/parse_lhe.py
ANALYSE_SCRIPT := src/diboson/main.py
export PYTHONPATH := src

# ── Variable overrides (take a value, so kept as variables) ───────────────────
#   NEVENTS=50000
#   OUTPUT_DIR=outputs/data/raw/ZZ/tests
#   START_REGION="4 2"   (cos_idx mass_idx — resume events/analysis from that bin onwards)
#   RUN_START=5          (MadGraph run number to resume parsing from)
NEVENTS       :=
OUTPUT_DIR    :=
START_REGION  :=
RUN_START     :=

# ── Modifier targets (pass as extra words on the command line) ────────────────
#   make events-zz whole-phase-space
#   make events-zz whole-phase-space NEVENTS=50000
#   make events-zz START_REGION="4 2"
#   make parse-zz whole-phase-space
#   make parse-zz append
#   make parse-zz whole-phase-space OUTPUT_DIR=outputs/data/raw/ZZ/tests
#   make parse-zz RUN_START=5
#   make analyse-zz raw
#   make analyse-zz plot-only
#   make analyse-zz raw plot-only
#   make analyse-zz START_REGION="4 2"

_WPS       := $(if $(filter whole-phase-space,$(MAKECMDGOALS)),--whole-phase-space)
_APPEND    := $(if $(filter append,$(MAKECMDGOALS)),--append)
_RAW       := $(if $(filter raw,$(MAKECMDGOALS)),--raw)
_PLOT_ONLY := $(if $(filter plot-only,$(MAKECMDGOALS)),--plot-only)

_NEVENTS_FLAG       := $(if $(NEVENTS),--nevents $(NEVENTS))
_OUTPUT_DIR_FLAG    := $(if $(OUTPUT_DIR),--output-dir $(OUTPUT_DIR))
_START_REGION_FLAG  := $(if $(START_REGION),--start-region $(START_REGION))
_RUN_START_FLAG     := $(if $(RUN_START),--run-start $(RUN_START))

_EVENT_FLAGS   := $(_NEVENTS_FLAG) $(_WPS) $(_START_REGION_FLAG)
_PARSE_FLAGS   := $(_WPS) $(_APPEND) $(_OUTPUT_DIR_FLAG) $(_RUN_START_FLAG)
_ANALYSE_FLAGS := $(_RAW) $(_PLOT_ONLY) $(_START_REGION_FLAG)

.PHONY: events events-zz events-ww \
        parse parse-zz parse-ww \
        analyse analyse-zz analyse-ww \
        whole-phase-space append raw plot-only

# Modifier no-ops — exist only to be detected via MAKECMDGOALS
whole-phase-space append raw plot-only: ;

# ── Event generation ──────────────────────────────────────────────────────────
events: events-zz events-ww

events-zz:
	$(PYTHON) src/diboson/event_gen/automate.py --process ZZ $(_EVENT_FLAGS)

events-ww:
	$(PYTHON) src/diboson/event_gen/automate.py --process WW $(_EVENT_FLAGS)

# ── LHE parsing ───────────────────────────────────────────────────────────────
parse: parse-zz parse-ww

parse-zz:
	$(PYTHON) $(PARSE_SCRIPT) --process ZZ $(_PARSE_FLAGS)

parse-ww:
	$(PYTHON) $(PARSE_SCRIPT) --process WW $(_PARSE_FLAGS)

# ── Analysis ──────────────────────────────────────────────────────────────────
analyse: analyse-zz analyse-ww

analyse-zz:
	$(PYTHON) $(ANALYSE_SCRIPT) --process ZZ $(_ANALYSE_FLAGS)

analyse-ww:
	$(PYTHON) $(ANALYSE_SCRIPT) --process WW $(_ANALYSE_FLAGS)

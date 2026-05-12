PYTHON         := .venv/bin/python
PARSE_SCRIPT   := src/diboson/io/parse_lhe.py
ANALYSE_SCRIPT := src/diboson/main.py
export PYTHONPATH := src

# ── Variable overrides (take a value, so kept as variables) ───────────────────
#   NEVENTS=50000
#   OUTPUT_DIR=outputs/data/raw/ZZ/tests
NEVENTS    :=
OUTPUT_DIR :=

# ── Modifier targets (pass as extra words on the command line) ────────────────
#   make events-zz whole-phase-space
#   make events-zz whole-phase-space NEVENTS=50000
#   make parse-zz whole-phase-space
#   make parse-zz append
#   make parse-zz whole-phase-space OUTPUT_DIR=outputs/data/raw/ZZ/tests
#   make analyse-zz raw
#   make analyse-zz plot-only
#   make analyse-zz raw plot-only

_WPS       := $(if $(filter whole-phase-space,$(MAKECMDGOALS)),--whole-phase-space)
_APPEND    := $(if $(filter append,$(MAKECMDGOALS)),--append)
_RAW       := $(if $(filter raw,$(MAKECMDGOALS)),--raw)
_PLOT_ONLY := $(if $(filter plot-only,$(MAKECMDGOALS)),--plot-only)

_NEVENTS_FLAG    := $(if $(NEVENTS),--nevents $(NEVENTS))
_OUTPUT_DIR_FLAG := $(if $(OUTPUT_DIR),--output-dir $(OUTPUT_DIR))

_EVENT_FLAGS   := $(_NEVENTS_FLAG) $(_WPS)
_PARSE_FLAGS   := $(_WPS) $(_APPEND) $(_OUTPUT_DIR_FLAG)
_ANALYSE_FLAGS := $(_RAW) $(_PLOT_ONLY)

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

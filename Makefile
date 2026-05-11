PYTHON         := .venv/bin/python
PARSE_SCRIPT   := src/diboson/io/parse_lhe.py
ANALYSE_SCRIPT := src/diboson/main.py
export PYTHONPATH := src

# Event-generation flags (optional overrides)
#   NEVENTS=50000            override the number of events per run
#   WHOLE_PHASE_SPACE=1      generate one uncut run instead of looping over bins
NEVENTS           :=
WHOLE_PHASE_SPACE :=

_NEVENTS_FLAG = $(if $(NEVENTS),--nevents $(NEVENTS))
_WPS_FLAG     = $(if $(WHOLE_PHASE_SPACE),--whole-phase-space)
_EVENT_FLAGS  = $(_NEVENTS_FLAG) $(_WPS_FLAG)

.PHONY: events events-zz events-ww parse parse-zz parse-ww analyse analyse-zz analyse-ww

events: events-zz events-ww

events-zz:
	$(PYTHON) src/diboson/event_gen/automate.py --process ZZ $(_EVENT_FLAGS)

events-ww:
	$(PYTHON) src/diboson/event_gen/automate.py --process WW $(_EVENT_FLAGS)

PARSE_FLAGS   :=
ANALYSE_FLAGS :=

parse: parse-zz parse-ww

parse-zz:
	$(PYTHON) $(PARSE_SCRIPT) --process ZZ $(PARSE_FLAGS)

parse-ww:
	$(PYTHON) $(PARSE_SCRIPT) --process WW $(PARSE_FLAGS)

analyse: analyse-zz analyse-ww

analyse-zz:
	$(PYTHON) $(ANALYSE_SCRIPT) --process ZZ $(ANALYSE_FLAGS)

analyse-ww:
	$(PYTHON) $(ANALYSE_SCRIPT) --process WW $(ANALYSE_FLAGS)

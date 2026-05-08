PYTHON := .venv/bin/python

.PHONY: events events-zz events-ww

events: events-zz events-ww

events-zz:
	$(PYTHON) src/diboson/event_gen/automate.py --process ZZ

events-ww:
	$(PYTHON) src/diboson/event_gen/automate.py --process WW

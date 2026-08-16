# Build the report. Prefers latexmk; falls back to tectonic, then to a manual
# pdflatex/bibtex sequence.

MAIN := main

.PHONY: all latexmk tectonic manual clean distclean regenerate

all:
	@if command -v latexmk >/dev/null 2>&1; then $(MAKE) latexmk; \
	elif command -v tectonic >/dev/null 2>&1; then $(MAKE) tectonic; \
	else $(MAKE) manual; fi

latexmk:
	latexmk -pdf -interaction=nonstopmode -halt-on-error $(MAIN).tex

tectonic:
	tectonic -X compile $(MAIN).tex --keep-logs --outdir .

manual:
	pdflatex -interaction=nonstopmode $(MAIN).tex
	-bibtex $(MAIN)
	pdflatex -interaction=nonstopmode $(MAIN).tex
	pdflatex -interaction=nonstopmode $(MAIN).tex

# Rebuild sections, tables and figures from the stored pipeline results,
# then recompile. Run from the repository root.
regenerate:
	cd .. && python run_final_stage.py && python build_report.py
	$(MAKE) all

clean:
	rm -f $(MAIN).aux $(MAIN).log $(MAIN).out $(MAIN).bbl $(MAIN).blg \
	      $(MAIN).toc $(MAIN).fls $(MAIN).fdb_latexmk $(MAIN).synctex.gz

distclean: clean
	rm -f $(MAIN).pdf

# Build the arXiv version of docs/PAPER.md.
#   make         build main.pdf: pdflatex, bibtex, pdflatex, pdflatex if pdflatex is installed,
#                tectonic (https://tectonic-typesetting.github.io) otherwise
#   make sync    refresh the figures and references.bib from docs/
#   make check   build, then fail on undefined references or citations, overfull boxes wider
#                than 10pt, or a figure file that main.tex includes but is missing
#   make arxiv   build, check, then write arxiv.tar.gz (main.tex, main.bbl, figures) for upload

FIGURES := v2_arm_ece v3_n_curve v4_offpolicy v4_noise_sweep v4_scale v4_emotion_n_curve
# Every file main.tex includes, as written in \includegraphics.
INCLUDED = $(shell grep -o '\\includegraphics\(\[[^]]*\]\)\?{[^}]*}' main.tex | sed 's/.*{\(.*\)}/\1/')
PDFLATEX ?= pdflatex
BIBTEX ?= bibtex
TECTONIC ?= tectonic
LATEXFLAGS := -interaction=nonstopmode -halt-on-error
OVERFULL_MAX := 10

.PHONY: all sync check figures-present arxiv clean

all: main.pdf

main.pdf: main.tex references.bib $(FIGURES:%=figures/%.png)
	@if command -v $(PDFLATEX) >/dev/null 2>&1 && command -v $(BIBTEX) >/dev/null 2>&1; then \
	  echo "engine: pdflatex"; \
	  $(PDFLATEX) $(LATEXFLAGS) main.tex >/dev/null && \
	  $(BIBTEX) main && \
	  $(PDFLATEX) $(LATEXFLAGS) main.tex >/dev/null && \
	  $(PDFLATEX) $(LATEXFLAGS) main.tex >/dev/null || { tail -40 main.log; exit 1; }; \
	else \
	  echo "engine: tectonic (pdflatex not found)"; \
	  $(TECTONIC) -X compile --keep-logs --keep-intermediates main.tex; \
	fi

sync:
	cp ../docs/references.bib references.bib
	for f in $(FIGURES); do cp ../docs/figures/$$f.png figures/$$f.png; done

figures-present:
	@missing=0; for f in $(INCLUDED); do \
	  [ -f "$$f" ] || { echo "missing figure file: $$f"; missing=1; }; \
	done; exit $$missing

check: figures-present main.pdf
	@! grep -E "(Citation|Reference) .* undefined" main.log
	@! grep -q "There were undefined references" main.log
	@! grep -E "^LaTeX Warning: File .* not found" main.log
	@n=$$(grep -o 'Overfull \\[hv]box ([0-9.]*pt too' main.log | awk '{gsub(/[(pt]/,"",$$3); if ($$3+0 > $(OVERFULL_MAX)) c++} END {print c+0}'); \
	  all=$$(grep -c 'Overfull \\[hv]box' main.log); \
	  echo "overfull boxes: $$all in total, $$n wider than $(OVERFULL_MAX)pt"; \
	  [ $$n -eq 0 ] || { grep 'Overfull \\[hv]box' main.log; exit 1; }
	@grep -o "Output written on [^)]*" main.log

# arXiv rebuilds from main.tex and main.bbl with pdflatex, so references.bib stays out.
arxiv: check
	@! grep -nE '\\(input|include)\b|\{/' main.tex || { echo "main.tex has \\input, \\include or an absolute path"; exit 1; }
	rm -f arxiv.tar.gz
	COPYFILE_DISABLE=1 tar -czf arxiv.tar.gz main.tex main.bbl $(INCLUDED)
	@tar -tzf arxiv.tar.gz

clean:
	rm -f main.pdf main.log main.blg main.bbl main.aux main.out main.toc main.xdv arxiv.tar.gz

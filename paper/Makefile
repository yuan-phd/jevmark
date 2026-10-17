# Build the arXiv version of docs/PAPER.md with tectonic (https://tectonic-typesetting.github.io).
#   make         build main.pdf
#   make sync    refresh the figures and references.bib from docs/
#   make arxiv   build arxiv.tar.gz (main.tex, main.bbl, references.bib, figures) for upload
#   make check   build and fail on any undefined citation or reference

FIGURES := v2_arm_ece v3_n_curve v4_offpolicy v4_noise_sweep v4_scale v4_emotion_n_curve
TECTONIC ?= tectonic

.PHONY: all sync check arxiv clean

all: main.pdf

main.pdf: main.tex references.bib $(FIGURES:%=figures/%.png)
	$(TECTONIC) -X compile --keep-logs --keep-intermediates main.tex

sync:
	cp ../docs/references.bib references.bib
	for f in $(FIGURES); do cp ../docs/figures/$$f.png figures/$$f.png; done

check: main.pdf
	@! grep -E "(Citation|Reference) .* undefined" main.log
	@grep -o "Output written on [^)]*" main.log

arxiv: main.pdf
	tar -czf arxiv.tar.gz main.tex main.bbl references.bib $(FIGURES:%=figures/%.png)

clean:
	rm -f main.pdf main.log main.blg main.bbl main.aux main.out main.toc main.xdv arxiv.tar.gz

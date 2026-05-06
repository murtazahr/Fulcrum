# Build the Fulcrum manuscript.
#
# Requires:  texlive-full (or equivalent) with latexmk
# Produces:  main.pdf
#
# Targets:
#   make           - build main.pdf (default)
#   make clean     - remove auxiliary build files (keeps main.pdf)
#   make distclean - remove all build artifacts including main.pdf
#   make watch     - continuous rebuild on file change (requires latexmk -pvc)

LATEX = latexmk -pdf -bibtex -interaction=nonstopmode -halt-on-error
MAIN = main

.PHONY: all clean distclean watch

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex references.bib $(wildcard sections/*.tex)
	$(LATEX) $(MAIN).tex

clean:
	latexmk -c $(MAIN).tex
	rm -f sections/*.aux

distclean:
	latexmk -C $(MAIN).tex
	rm -f sections/*.aux $(MAIN).pdf

watch:
	$(LATEX) -pvc $(MAIN).tex

TITLE ?= 1
ABSTRACT ?= 1
INTRO ?= 1
JOB = pbpf-t$(TITLE)-a$(ABSTRACT)-i$(INTRO)

.PHONY: all variants static
all:
	mkdir -p build
	latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build -jobname=$(JOB) -pretex='\def\TitleVersion{$(TITLE)}\def\AbstractVersion{$(ABSTRACT)}\def\IntroVersion{$(INTRO)}' -usepretex pbpf_iclr2027.tex

variants:
	python3 validate_variants.py

static:
	python3 validate_variants.py --static-only

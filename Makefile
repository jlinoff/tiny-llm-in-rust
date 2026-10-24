SRC_FILES := $(shell git ls-files)
CORPUS    ?= "data/corpus.txt"
MERGES    ?= 1000

# Stages
.PHONY: tokenize
tokenize: ## Learns the tokenizer from the training text.
	$(call hdr,"$@")
	@rm -f tok.bpe
	time python3 tokenizer/mini_bpe.py $(CORPUS) -n $(MERGES) --save tok.bpe

.PHONY: clean
clean: ## Clean up all generated files.
	git clean -fdx
	rm -f *~

.PHONY: help
zip: project.zip ## Create the project zip file: project.zip.

project.zip: $(SRC_FILES)
	zip $@ $(SRC_FILES)


.PHONY: help
help:  ## This help message.
	$(call hdr,"$@")
	@printf "\n\033[35;1m%s\n" "Targets"
	@grep -E '^[^.[:space:]][^[:space:]]*:.*[[:space:]]##' $(MAKEFILE_LIST) 2>/dev/null | \
		grep -E -v '^ *#' | \
		grep -E -v "egrep|sort|sed|MAKEFILE" | \
		sed -e 's/:[[:space:]].*##/##/' -e 's/^[^:#]*://' | \
		awk -F'##' '{printf("%-18s %s\n",$$1,$$2)}' | \
		sort -f | \
		sed -e 's@^@   @'

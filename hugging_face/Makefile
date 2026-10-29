SHELL     := /bin/bash

.PHONY: download
download: models/SmolLM-135M/model.safetensors ## download SmolLM-135M

models/SmolLM-135M/model.safetensors:
	pipenv run hf download HuggingFaceTB/SmolLM-135M  --local-dir ./models/SmolLM-135

.PHONY: help
help:  ## This help message.
	$(call hdr,$@)
	@printf "\n\033[35;1m%s\033[0m\n" "Targets"
	@grep -E '^[^.[:space:]][^[:space:]]*:.*[[:space:]]##' $(MAKEFILE_LIST) 2>/dev/null | \
		grep -E -v '^ *#' | \
		grep -E -v "egrep|sort|sed|MAKEFILE" | \
		sed -e 's/:[[:space:]].*##/##/' -e 's/^[^:#]*://' | \
		awk -F'##' '{printf("%-18s %s\n",$$1,$$2)}' | \
		sort -f | \
		sed -e 's@^@   @'

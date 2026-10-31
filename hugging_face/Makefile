SHELL     := /bin/bash

.PHONY: download
download: models/SmolLM-135M/model.safetensors ## download SmolLM-135M

models/SmolLM-135M/model.safetensors:
	pipenv run hf download HuggingFaceTB/SmolLM-135M  --local-dir ./models/SmolLM-135

# Once this is done
# cd tiny-llm-in-rust
# make llama LLAMA_DIR=hugging_face/models/SmolLM-1.7B PROMPT='Once upon a time' LLM_ARGS="--seed 5"
models/SmolLM-1.7B/model.safetensors:
	pipenv run hf download HuggingFaceTB/SmolLM-1.7B --local-dir models/SmolLM-1.7B --exclude "onnx/*"
	cd models/SmolLM-1.7B && pipenv run python3 ../../join.py

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

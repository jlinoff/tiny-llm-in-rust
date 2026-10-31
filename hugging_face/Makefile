SHELL := /bin/bash
.DEFAULT_GOAL := help

# Downloading open-weight models into models/ (git-ignored), and Hugging
# Face's reference numbers for checking them (PLAN.md, The endpoint, and
# Trying a new model). Run `pipenv install` here first.

# Print a section header: $(call hdr,text)
hdr = @printf "\n\033[35;1m== %s ==\033[0m\n" "$(1)"

.PHONY: download-small
download-small: models/SmolLM-135M/model.safetensors ## Download SmolLM-135M (513 MB).

.PHONY: download-big
download-big: models/SmolLM-1.7B/model.safetensors ## Download SmolLM-1.7B (6.8 GB) and merge its shards.

models/SmolLM-135M/model.safetensors:
	$(call hdr,$@)
	pipenv run hf download HuggingFaceTB/SmolLM-135M --local-dir models/SmolLM-135M --exclude "onnx/*"

# Then, from the top of the repository:
#   make llama LLAMA_DIR=hugging_face/models/SmolLM-1.7B PROMPT='Once upon a time' LLAMA_ARGS='--seed 5'
models/SmolLM-1.7B/model.safetensors:
	$(call hdr,$@)
	pipenv run hf download HuggingFaceTB/SmolLM-1.7B --local-dir models/SmolLM-1.7B --exclude "onnx/*"
	pipenv run python3 ../tools/join_shards.py models/SmolLM-1.7B

# Hugging Face's reference numbers, for make llama-check; then, from the top:
#   make llama-check LLAMA_DIR=hugging_face/models/SmolLM-1.7B
models/%/reference.json: models/%/model.safetensors
	$(call hdr,$@)
	pipenv run python3 ../tools/llama_reference.py models/$* -o $@

.PHONY: reference-small reference-big
reference-small: models/SmolLM-135M/reference.json ## Hugging Face's reference numbers for SmolLM-135M.
reference-big: models/SmolLM-1.7B/reference.json ## Hugging Face's reference numbers for SmolLM-1.7B.

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

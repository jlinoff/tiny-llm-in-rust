SHELL     := /bin/bash
SRC_FILES := $(shell git ls-files)
CORPUS    ?= data/corpus.txt
MERGES    ?= 1000
OUT       ?= out

# Print a section header: $(call hdr,text)
hdr = @printf "\n\033[35;1m== %s ==\033[0m\n" "$(1)"

# Stages
.PHONY: tokenize
tokenize: $(OUT)/tok.bpe ## Step 1: learn the tokenizer from the training text.

# Rebuilt only when the corpus or the script changes.
$(OUT)/tok.bpe: $(CORPUS) tokenizer/mini_bpe.py | $(OUT)
	$(call hdr,tokenize)
	time python3 tokenizer/mini_bpe.py $(CORPUS) -n $(MERGES) --save $@

# run make -B encode MERGES=2000 to rebuild everything when a variable changes.
.PHONY: encode
encode: $(OUT)/corpus.ids ## Step 2: encode the corpus into token IDs.

# Rebuilt when the tokenizer, the corpus or the script changes.
$(OUT)/corpus.ids: $(OUT)/tok.bpe $(CORPUS) tokenizer/mini_bpe.py | $(OUT)
	$(call hdr,encode)
	time python3 tokenizer/mini_bpe.py --load $(OUT)/tok.bpe --encode $(CORPUS) -o $@

$(OUT):
	mkdir -p $@

# Step 3: build. Rebuilt when the tokenizer, the config or the Rust source changes.
TRANSFORMER     := transformer/target/release/transformer
TRANSFORMER_SRC := transformer/Cargo.toml $(wildcard transformer/src/*.rs)

.PHONY: build
build: $(OUT)/model-init.safetensors ## Step 3: create the model with random weights.

$(OUT)/model-init.safetensors: $(OUT)/tok.bpe config.toml $(TRANSFORMER) | $(OUT)
	$(call hdr,build)
	$(TRANSFORMER) build --tokenizer $(OUT)/tok.bpe --config config.toml -o $@

$(TRANSFORMER): $(TRANSFORMER_SRC)
	$(call hdr,build transformer)
	cargo build --release --manifest-path transformer/Cargo.toml

.PHONY: bench
bench: $(TRANSFORMER) ## Measure this machine's matrix-multiply speed.
	$(call hdr,$@)
	$(TRANSFORMER) bench

# attention (the learning step before transformer/)
ATTN       := attention/target/release/mini_attention
ATTN_SRC   := attention/Cargo.toml $(wildcard attention/src/*.rs)

.PHONY: attention
attention: $(ATTN) ## Self-attention: simple example -> random -> train -> heads -> block -> induction
	$(call hdr,attention example)
	$(ATTN) example
	$(call hdr,attention random)
	$(ATTN) random
	$(call hdr,attention train)
	$(ATTN) train
	$(call hdr,attention heads)
	$(ATTN) heads
	$(call hdr,attention block)
	$(ATTN) block
	$(call hdr,attention induction)
	$(ATTN) induction

# Rebuilt only when the Rust source changes.
$(ATTN): $(ATTN_SRC)
	$(call hdr,build mini_attention)
	cargo build --release --manifest-path attention/Cargo.toml

# word2vec (the historical side branch; not part of the LLM flow)
W2V        := word2vec/target/release/mini_w2v
W2V_SRC    := word2vec/Cargo.toml $(wildcard word2vec/src/*.rs)
W2V_CORPUS ?= data/text8
W2V_ARGS   ?= -p king,cat,paris,computer,three -e 10 -d 300 -m skipgram
W2V_OUT    ?= $(OUT)/vectors.txt

.PHONY: word2vec
word2vec: $(W2V_CORPUS) $(W2V) | $(OUT) ## Train word vectors with the standalone word2vec tool.
	$(call hdr,$@)
	@mkdir -p $(dir $(W2V_OUT))
	time $(W2V) $(W2V_CORPUS) $(W2V_ARGS) -o $(W2V_OUT)

# Rebuilt only when the Rust source changes.
$(W2V): $(W2V_SRC)
	$(call hdr,build mini_w2v)
	cargo build --release --manifest-path word2vec/Cargo.toml

# Downloaded once; data/ is git-ignored and survives make clean.
data/text8:
	$(call hdr,$@)
	mkdir -p data
	rm -f data/text8.gz
	curl -fL -o data/text8.gz https://github.com/RaRe-Technologies/gensim-data/releases/download/text8/text8.gz
	gunzip data/text8.gz


.PHONY: clean
clean: ## Remove generated files (keeps data/ and uncommitted work).
	cd word2vec && cargo clean
	cd attention && cargo clean
	rm -rf $(OUT) project.zip
	rm -f *~

.PHONY: zip
zip: project.zip ## Create the project zip file: project.zip.

project.zip: $(SRC_FILES)
	zip $@ $(SRC_FILES)

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
	@printf "\n\033[35;1m%s\033[0m\n" "Variables"
	@printf "   %-18s  %s\n" "CORPUS" "$(CORPUS)"
	@printf "   %-18s  %s\n" "MERGES" "$(MERGES)"
	@printf "   %-18s  %s\n" "OUT" "$(OUT)"
	@printf "   %-18s  %s\n" "W2V" "$(W2V)"
	@printf "   %-18s  %s\n" "W2V_CORPUS" "$(W2V_CORPUS)"
	@printf "   %-18s  %s\n" "W2V_ARGS" "$(W2V_ARGS)"
	@printf "   %-18s  %s\n" "W2V_OUT" "$(W2V_OUT)"

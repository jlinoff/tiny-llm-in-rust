SHELL     := /bin/bash
SRC_FILES := $(shell git ls-files)
CORPUS    ?= data/corpus.txt
MERGES    ?= 1000
OUT       ?= out

# Print a section header: $(call hdr,text)
hdr = @printf "\n\033[35;1m== %s ==\033[0m\n" "$(1)"

# Stages
.PHONY: tokenize
tokenize: $(OUT)/tok.bpe ## Learn the tokenizer from the training text.

# Rebuilt only when the corpus or the script changes. Input comes from
# /dev/null so the script's interactive prompt ends immediately.
$(OUT)/tok.bpe: $(CORPUS) tokenizer/mini_bpe.py | $(OUT)
	$(call hdr,tokenize)
	time python3 tokenizer/mini_bpe.py $(CORPUS) -n $(MERGES) --save $@ < /dev/null

$(OUT):
	mkdir -p $@

.PHONY: clean
clean: ## Remove generated files (keeps data/ and uncommitted work).
	rm -rf $(OUT) project.zip
	rm -f *~

.PHONY: zip
zip: project.zip ## Create the project zip file: project.zip.

project.zip: $(SRC_FILES)
	zip $@ $(SRC_FILES)

# word2vec (the historical side branch; not part of the LLM flow)
W2V        := word2vec/target/release/mini_w2v
W2V_SRC    := word2vec/Cargo.toml $(wildcard word2vec/src/*.rs)
W2V_CORPUS ?= data/text8
W2V_ARGS   ?= -p king,cat,paris,computer,three

.PHONY: word2vec
word2vec: $(W2V_CORPUS) $(W2V) | $(OUT) ## Train word vectors with the standalone word2vec tool.
	$(call hdr,$@)
	time $(W2V) $(W2V_CORPUS) $(W2V_ARGS) -o $(OUT)/vectors.txt

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

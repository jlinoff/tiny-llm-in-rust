SHELL     := /bin/bash
SRC_FILES := $(shell git ls-files)
OUT       ?= out

# The corpus: TinyStories' validation file (19 MB, about 22,000 stories), for
# developing; its stories are separated by '<|endoftext|>' lines (SPLIT_ON).
# For a plain text file with no separators, use SPLIT_ON= (empty).
CORPUS    ?= data/TinyStories-valid.txt
# The tokenizer is learned from TOK_CORPUS: the corpus itself by default, or
# a smaller sample of it (the full run learns it from V2's validation file).
TOK_CORPUS ?= $(CORPUS)
# The settings for steps 3-6.
CONFIG    ?= config.toml
SPLIT_ON  ?= <|endoftext|>
SPLIT      = $(if $(SPLIT_ON),--split-on '$(SPLIT_ON)')
# 3,836 merges: V = 256 bytes + 3,836 merges + 4 special tokens (EOS and the
# three turn markers) = 4,096.
MERGES    ?= 3836

# Print a section header: $(call hdr,text)
hdr = @printf "\n\033[35;1m== %s ==\033[0m\n" "$(1)"

# Stages
.PHONY: tokenize
tokenize: $(OUT)/tok.bpe ## Step 1: learn the tokenizer from the training text.

# Rebuilt only when the corpus or the script changes.
$(OUT)/tok.bpe: $(TOK_CORPUS) tokenizer/mini_bpe.py | $(OUT)
	$(call hdr,tokenize)
	time python3 tokenizer/mini_bpe.py $(TOK_CORPUS) -n $(MERGES) $(SPLIT) --save $@

# run make -B encode MERGES=2000 to rebuild everything when a variable changes.
.PHONY: encode
encode: $(OUT)/corpus.ids ## Step 2: encode the corpus into token IDs.

# Rebuilt when the tokenizer, the corpus or the script changes.
$(OUT)/corpus.ids: $(OUT)/tok.bpe $(CORPUS) tokenizer/mini_bpe.py | $(OUT)
	$(call hdr,encode)
	time python3 tokenizer/mini_bpe.py --load $(OUT)/tok.bpe --encode $(CORPUS) $(SPLIT) -o $@

$(OUT):
	mkdir -p $@

# Step 3: build. Rebuilt when the tokenizer, the config or the Rust source changes.
TRANSFORMER     := transformer/target/release/transformer
TRANSFORMER_SRC := transformer/Cargo.toml $(wildcard transformer/src/*.rs)

.PHONY: build
build: $(OUT)/model-init.safetensors ## Step 3: create the model with random weights.

$(OUT)/model-init.safetensors: $(OUT)/tok.bpe $(CONFIG) $(TRANSFORMER) | $(OUT)
	$(call hdr,build)
	$(TRANSFORMER) build --tokenizer $(OUT)/tok.bpe --config $(CONFIG) -o $@

$(TRANSFORMER): $(TRANSFORMER_SRC)
	$(call hdr,build transformer)
	cargo build --release --manifest-path transformer/Cargo.toml

.PHONY: bench
bench: $(TRANSFORMER) $(OUT)/model-init.safetensors ## Measure matrix-multiply speed and a real training step.
	$(call hdr,$@)
	$(TRANSFORMER) bench --checkpoint $(OUT)/model-init.safetensors

# Step 4: train. Starts from model-init.safetensors; to resume a stopped run
# instead, run the command with --model $(OUT)/model.safetensors.
.PHONY: train
train: $(OUT)/model.safetensors ## Step 4: train the model on corpus.ids.

$(OUT)/model.safetensors: $(OUT)/model-init.safetensors $(OUT)/corpus.ids $(CONFIG) $(TRANSFORMER) | $(OUT)
	$(call hdr,train)
	time $(TRANSFORMER) train --model $(OUT)/model-init.safetensors --data $(OUT)/corpus.ids --config $(CONFIG) -o $@

# Step 5: finetune (post-training). The conversations are built from the
# corpus's stories: a request naming three words from each story, answered
# by the story. chat-eval.tsv holds requests from the held-out stories.
# CHAT_ARGS: make_chat.py's options, such as '--skip-common 200'.
CHAT_ARGS ?=

$(OUT)/chat.txt: $(CORPUS) chat/make_chat.py | $(OUT)
	$(call hdr,chat data)
	python3 chat/make_chat.py $(CORPUS) $(SPLIT) $(CHAT_ARGS) -o $(OUT)/chat.txt --eval $(OUT)/chat-eval.tsv --eval-chat $(OUT)/chat-eval.txt

# Written by the same command as chat.txt.
$(OUT)/chat-eval.tsv $(OUT)/chat-eval.txt: $(OUT)/chat.txt ;

$(OUT)/chat.ids: $(OUT)/chat.txt $(OUT)/tok.bpe tokenizer/mini_bpe.py | $(OUT)
	$(call hdr,encode chat)
	python3 tokenizer/mini_bpe.py --load $(OUT)/tok.bpe --encode $(OUT)/chat.txt --split-on '<|endoftext|>' --allow-special -o $@

.PHONY: finetune
finetune: $(OUT)/chat.safetensors ## Step 5: fine-tune the trained model on conversations.

$(OUT)/chat.safetensors: $(OUT)/model.safetensors $(OUT)/chat.ids $(CONFIG) $(TRANSFORMER) | $(OUT)
	$(call hdr,finetune)
	time $(TRANSFORMER) finetune --model $(OUT)/model.safetensors --data $(OUT)/chat.ids --tokenizer $(OUT)/tok.bpe --config $(CONFIG) -o $@

# How much the models use the requests: the pretrained model (requests as
# plain text) against the fine-tuned one (as conversations). Each run does
# the paired-loss check (each held-out story's loss given its own request
# vs another's), then samples stories and counts the requested words. Like
# generate, the models are required but not prerequisites.
EVAL_ARGS ?=

.PHONY: evaluate
evaluate: $(TRANSFORMER) $(OUT)/chat-eval.tsv $(OUT)/chat-eval.txt ## Measure fine-tuning: requested words in stories, before and after.
	@test -f $(OUT)/chat.safetensors || { echo "no $(OUT)/chat.safetensors: run make finetune first"; exit 1; }
	$(call hdr,evaluate pretrained)
	$(TRANSFORMER) evaluate --model $(OUT)/model.safetensors --tokenizer $(OUT)/tok.bpe --config $(CONFIG) --requests $(OUT)/chat-eval.tsv --paired $(OUT)/chat-eval.txt $(EVAL_ARGS)
	$(call hdr,evaluate fine-tuned)
	$(TRANSFORMER) evaluate --model $(OUT)/chat.safetensors --tokenizer $(OUT)/tok.bpe --config $(CONFIG) --requests $(OUT)/chat-eval.tsv --paired $(OUT)/chat-eval.txt --chat $(EVAL_ARGS)

# Talk to the fine-tuned model: make chat PROMPT='Write a story with the words frog, bell and egg.'
.PHONY: chat
chat: $(TRANSFORMER) ## Step 6 with the fine-tuned model: PROMPT is the request.
	$(call hdr,$@)
	@test -f $(OUT)/chat.safetensors || { echo "no $(OUT)/chat.safetensors: run make finetune first"; exit 1; }
	$(TRANSFORMER) generate --model $(OUT)/chat.safetensors --tokenizer $(OUT)/tok.bpe --config $(CONFIG) --chat --prompt '$(PROMPT)' $(GEN_ARGS)

# Step 6. generate (inference)
# Always runs (nothing is written). Override the prompt or the sampling:
#   make generate PROMPT='Once upon a time' GEN_ARGS='--temperature 0.5 --seed 7'
# The model is required but deliberately not a prerequisite: it depends on
# config.toml and the binary, so editing [generate] or rebuilding would
# otherwise retrain it first.
PROMPT ?= Once upon a time
GEN_ARGS ?=

.PHONY: generate
generate: $(TRANSFORMER) ## Step 6: write text from PROMPT with the trained model (inference).
	$(call hdr,$@)
	@test -f $(OUT)/model.safetensors || { echo "no $(OUT)/model.safetensors: run make train first"; exit 1; }
	$(TRANSFORMER) generate --model $(OUT)/model.safetensors --tokenizer $(OUT)/tok.bpe --config $(CONFIG) --prompt '$(PROMPT)' $(GEN_ARGS)


# The full TinyStories run (PLAN.md, "The full TinyStories run"): V2's
# training file (GPT-4 stories only), the 7.4M-parameter model in
# config-full.toml, about a day of training. Any target with full- in front
# runs with these settings, its files in out-full/ (which make clean leaves
# alone): make full-bench, full-train, full-finetune, full-evaluate,
# full-chat. For a day-long run: caffeinate -i make full-train
FULL_VARS = OUT=out-full CORPUS=data/TinyStoriesV2-GPT4-train.txt \
	TOK_CORPUS=data/TinyStoriesV2-GPT4-valid.txt CONFIG=config-full.toml \
	CHAT_ARGS='--max-stories 200000 --eval-fraction 0.004'

full-%: data/TinyStoriesV2-GPT4-train.txt data/TinyStoriesV2-GPT4-valid.txt ## The full run: make full-train, full-finetune, full-evaluate, ...
	$(MAKE) $* $(FULL_VARS)

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

# Data
# Downloaded once; data/ is git-ignored and survives make clean.
data/text8:
	$(call hdr,$@)
	mkdir -p data
	rm -f data/text8.gz
	curl -fL -o data/text8.gz https://github.com/RaRe-Technologies/gensim-data/releases/download/text8/text8.gz
	gunzip data/text8.gz

# Larger corpus for better inference
data/TinyStories-valid.txt:
	$(call hdr,$@)
	mkdir -p data
	curl -fL -o $@.part https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt
	mv $@.part $@

# TinyStories V2 (GPT-4 stories only), for the full run: 2.2 GB and 22 MB.
data/TinyStoriesV2-GPT4-train.txt data/TinyStoriesV2-GPT4-valid.txt:
	$(call hdr,$@)
	mkdir -p data
	curl -fL -o $@.part https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/$(notdir $@)
	mv $@.part $@

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
	@printf "   %-18s  %s\n" "CHAT_ARGS" "$(CHAT_ARGS)"
	@printf "   %-18s  %s\n" "CONFIG" "$(CONFIG)"
	@printf "   %-18s  %s\n" "CORPUS" "$(CORPUS)"
	@printf "   %-18s  %s\n" "EVAL_ARGS" "$(EVAL_ARGS)"
	@printf "   %-18s  %s\n" "GEN_ARGS" "$(GEN_ARGS)"
	@printf "   %-18s  %s\n" "MERGES" "$(MERGES)"
	@printf "   %-18s  %s\n" "OUT" "$(OUT)"
	@printf "   %-18s  %s\n" "PROMPT" "$(PROMPT)"
	@printf "   %-18s  %s\n" "SPLIT_ON" "$(SPLIT_ON)"
	@printf "   %-18s  %s\n" "TOK_CORPUS" "$(TOK_CORPUS)"
	@printf "   %-18s  %s\n" "W2V" "$(W2V)"
	@printf "   %-18s  %s\n" "W2V_CORPUS" "$(W2V_CORPUS)"
	@printf "   %-18s  %s\n" "W2V_ARGS" "$(W2V_ARGS)"
	@printf "   %-18s  %s\n" "W2V_OUT" "$(W2V_OUT)"

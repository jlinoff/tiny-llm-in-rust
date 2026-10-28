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

# Each step depends only on its own section of the config file: build on
# [model], train on [training], finetune on [finetune] (copied to
# $(OUT)/config/ by tools/config_section.py, rewritten only when a value in
# that section changes). So editing [finetune] reruns only fine-tuning, and
# editing [generate] or a comment reruns nothing. The recipe runs whenever
# the config file is newer, which is cheap and leaves unchanged sections'
# files, and so the models, alone.
$(OUT)/config/%.toml: $(CONFIG) tools/config_section.py | $(OUT)
	@python3 tools/config_section.py $(CONFIG) $* $@

# Step 3: build. Rebuilt when the tokenizer or [model] changes.
# The transformer binary is an order-only prerequisite (after the |) of every
# model file: make builds it if it's out of date, but a rebuilt binary doesn't
# make existing models out of date, so new code never retrains a model by
# itself. (After changing how models are built or trained, rebuild them on
# purpose: delete the file, or make -B.)
TRANSFORMER     := transformer/target/release/transformer
TRANSFORMER_SRC := transformer/Cargo.toml $(wildcard transformer/src/*.rs)

.PHONY: build
build: $(OUT)/model-init.safetensors ## Step 3: create the model with random weights.

$(OUT)/model-init.safetensors: $(OUT)/tok.bpe $(OUT)/config/model.toml | $(OUT) $(TRANSFORMER)
	$(call hdr,build)
	$(TRANSFORMER) build --tokenizer $(OUT)/tok.bpe --config $(CONFIG) -o $@

# CARGO_FEATURES=blas builds with Apple's Accelerate library for the large
# matrix products (PLAN.md, decision 8): the default on macOS, where it made
# a training step 2.68 times as fast; CARGO_FEATURES= builds the plain Rust
# loops. Make doesn't notice a change of features: rebuild on purpose with
# make -B $(TRANSFORMER) CARGO_FEATURES=...
CARGO_FEATURES ?= $(if $(filter Darwin,$(shell uname -s)),blas,)

$(TRANSFORMER): $(TRANSFORMER_SRC)
	$(call hdr,build transformer)
	cargo build --release --manifest-path transformer/Cargo.toml $(if $(CARGO_FEATURES),--features $(CARGO_FEATURES))

.PHONY: bench
bench: $(TRANSFORMER) $(OUT)/model-init.safetensors ## Measure matrix-multiply speed and a real training step.
	$(call hdr,$@)
	$(TRANSFORMER) bench --checkpoint $(OUT)/model-init.safetensors

# Step 4: train. Starts from model-init.safetensors; to resume a stopped run
# instead, run the command with --model $(OUT)/model.safetensors.
.PHONY: train
train: $(OUT)/model.safetensors ## Step 4: train the model on corpus.ids.

$(OUT)/model.safetensors: $(OUT)/model-init.safetensors $(OUT)/corpus.ids $(OUT)/config/training.toml | $(OUT) $(TRANSFORMER)
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

$(OUT)/chat.safetensors: $(OUT)/model.safetensors $(OUT)/chat.ids $(OUT)/config/finetune.toml | $(OUT) $(TRANSFORMER)
	$(call hdr,finetune)
	time $(TRANSFORMER) finetune --model $(OUT)/model.safetensors --data $(OUT)/chat.ids --tokenizer $(OUT)/tok.bpe --config $(CONFIG) -o $@

# Step 5 with LoRA (PLAN.md, "LoRA: fine-tuning a low-rank adapter"): the
# same conversations and [finetune] settings, but the model is frozen and
# only a small adapter is trained (chat-lora.safetensors), then folded into
# the model (chat-lora-merged.safetensors) for generate and evaluate.
# LORA_ARGS: the rank, and a learning rate (LoRA usually wants a higher one);
# add --lora-embeddings markers (or all) to train embedding rows too.
LORA_ARGS ?= --lora-rank 8 --learning-rate 0.003

$(OUT)/chat-lora.safetensors: $(OUT)/model.safetensors $(OUT)/chat.ids $(OUT)/config/finetune.toml | $(OUT) $(TRANSFORMER)
	$(call hdr,finetune with LoRA)
	time $(TRANSFORMER) finetune --model $(OUT)/model.safetensors --data $(OUT)/chat.ids --tokenizer $(OUT)/tok.bpe --config $(CONFIG) $(LORA_ARGS) -o $@

$(OUT)/chat-lora-merged.safetensors: $(OUT)/model.safetensors $(OUT)/chat-lora.safetensors | $(OUT) $(TRANSFORMER)
	$(call hdr,merge)
	$(TRANSFORMER) merge --model $(OUT)/model.safetensors --adapter $(OUT)/chat-lora.safetensors -o $@

.PHONY: lora
lora: $(OUT)/chat-lora-merged.safetensors ## Step 5 with LoRA: train a low-rank adapter, then merge it.

.PHONY: lora-evaluate
lora-evaluate: $(TRANSFORMER) $(OUT)/chat-eval.tsv $(OUT)/chat-eval.txt ## Measure the LoRA-tuned model, as evaluate does.
	@test -f $(OUT)/chat-lora-merged.safetensors || { echo "no $(OUT)/chat-lora-merged.safetensors: run make lora first"; exit 1; }
	$(call hdr,evaluate LoRA)
	$(TRANSFORMER) evaluate --model $(OUT)/chat-lora-merged.safetensors --tokenizer $(OUT)/tok.bpe --config $(CONFIG) --requests $(OUT)/chat-eval.tsv --paired $(OUT)/chat-eval.txt --chat $(EVAL_ARGS)

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
# CHAT_MODEL picks another, such as the LoRA one: CHAT_MODEL=out/chat-lora-merged.safetensors
CHAT_MODEL ?= $(OUT)/chat.safetensors

.PHONY: chat
chat: $(TRANSFORMER) ## Step 6 with the fine-tuned model: PROMPT is the request.
	$(call hdr,$@)
	@test -f $(CHAT_MODEL) || { echo "no $(CHAT_MODEL): run make finetune first"; exit 1; }
	$(TRANSFORMER) generate --model $(CHAT_MODEL) --tokenizer $(OUT)/tok.bpe --config $(CONFIG) --chat --prompt '$(PROMPT)' $(GEN_ARGS)

# Look inside a model (PLAN.md, "Looking inside the model"): head scores,
# the logit lens, one head's attention. PROBE_MODEL picks the checkpoint,
# such as a copy of one taken during training; PROBE_ARGS passes options,
# such as '--head 1.3'. Like generate, the model isn't a prerequisite.
PROBE_MODEL ?= $(OUT)/model.safetensors
PROBE_ARGS ?=

.PHONY: probe
probe: $(TRANSFORMER) ## Look inside the trained model: head scores, logit lens, attention.
	$(call hdr,$@)
	@test -f $(PROBE_MODEL) || { echo "no $(PROBE_MODEL): run make train first"; exit 1; }
	$(TRANSFORMER) probe --model $(PROBE_MODEL) --tokenizer $(OUT)/tok.bpe $(PROBE_ARGS)

# Step 6. generate (inference)
# Always runs (nothing is written). Override the prompt or the sampling:
#   make generate PROMPT='Once upon a time' GEN_ARGS='--temperature 0.5 --seed 7'
# The model is required but deliberately not a prerequisite, so generating
# never starts a training run (say, after editing [training] to try
# something later).
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
# full-chat, full-lora, full-lora-evaluate. For a day-long run:
# caffeinate -i make full-train
FULL_DATA = CORPUS=data/TinyStoriesV2-GPT4-train.txt \
	TOK_CORPUS=data/TinyStoriesV2-GPT4-valid.txt \
	CHAT_ARGS='--max-stories 200000 --eval-fraction 0.004'
FULL_VARS = OUT=out-full CONFIG=config-full.toml $(FULL_DATA)

full-%: data/TinyStoriesV2-GPT4-train.txt data/TinyStoriesV2-GPT4-valid.txt ## The full run (PLAN.md): full-bench, full-train, full-finetune, full-evaluate, full-chat, full-lora
	$(MAKE) $* $(FULL_VARS)

# The full run's flow end to end, with very few steps (config-full-check.toml):
# the same data, tokenizer and model shape, in about half an hour, into
# out-check/. Checks that the flow works, not its results; the tokenizer
# fingerprint, the token counts and the initial model's weights fingerprint
# should match out-full/'s exactly. (An explicit rule, so it wins over full-%.)
.PHONY: full-check
full-check: data/TinyStoriesV2-GPT4-train.txt data/TinyStoriesV2-GPT4-valid.txt ## Check the full run's flow end to end in about half an hour (out-check/).
	$(MAKE) finetune evaluate OUT=out-check CONFIG=config-full-check.toml $(FULL_DATA)

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

.PHONY: toc
toc: ## Update PLAN.md's table of contents and check its internal links.
	python3 tools/toc.py PLAN.md

.PHONY: clean
clean: ## Remove generated files (keeps data/ and out-full/; see FORCE).
	@if [ "$(OUT)" = out-full ] && [ -z "$(FORCE)" ]; then \
		echo "refusing to delete out-full/ (the full run: a day of training);"; \
		echo "to really delete it: make full-clean FORCE=1"; exit 1; fi
	cd word2vec && cargo clean
	cd attention && cargo clean
	rm -rf $(OUT) out-check project.zip
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
	@printf "   %-18s  %s\n" "CARGO_FEATURES" "$(CARGO_FEATURES)"
	@printf "   %-18s  %s\n" "CHAT_ARGS" "$(CHAT_ARGS)"
	@printf "   %-18s  %s\n" "CHAT_MODEL" "$(CHAT_MODEL)"
	@printf "   %-18s  %s\n" "CONFIG" "$(CONFIG)"
	@printf "   %-18s  %s\n" "CORPUS" "$(CORPUS)"
	@printf "   %-18s  %s\n" "EVAL_ARGS" "$(EVAL_ARGS)"
	@printf "   %-18s  %s\n" "FORCE" "$(FORCE)   (1 lets make full-clean delete out-full/)"
	@printf "   %-18s  %s\n" "GEN_ARGS" "$(GEN_ARGS)"
	@printf "   %-18s  %s\n" "LORA_ARGS" "$(LORA_ARGS)"
	@printf "   %-18s  %s\n" "MERGES" "$(MERGES)"
	@printf "   %-18s  %s\n" "OUT" "$(OUT)"
	@printf "   %-18s  %s\n" "PROBE_ARGS" "$(PROBE_ARGS)"
	@printf "   %-18s  %s\n" "PROBE_MODEL" "$(PROBE_MODEL)"
	@printf "   %-18s  %s\n" "PROMPT" "$(PROMPT)"
	@printf "   %-18s  %s\n" "SPLIT_ON" "$(SPLIT_ON)"
	@printf "   %-18s  %s\n" "TOK_CORPUS" "$(TOK_CORPUS)"
	@printf "   %-18s  %s\n" "W2V" "$(W2V)"
	@printf "   %-18s  %s\n" "W2V_CORPUS" "$(W2V_CORPUS)"
	@printf "   %-18s  %s\n" "W2V_ARGS" "$(W2V_ARGS)"
	@printf "   %-18s  %s\n" "W2V_OUT" "$(W2V_OUT)"

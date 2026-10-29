#!/bin/bash
# The whole flow on a tiny corpus, in seconds: a check that every step still
# runs and hands its files to the next (used by .github/workflows/main.yaml;
# runs locally too, from the top of the repository: tools/ci/smoke.sh).
#
# tools/ci/stories.txt holds a dozen short original stories in TinyStories'
# style, separated by <|endoftext|> lines; tools/ci/tiny.toml a tiny model.
# The steps: learn a tokenizer, encode, make chat data, build, train,
# generate, fine-tune, evaluate, probe. Nothing is learned that's worth
# keeping; the point is that nothing fails. Everything goes in a temporary
# folder.
set -euo pipefail
T=transformer/target/release/transformer
D=$(mktemp -d)
trap 'rm -rf "$D"' EXIT
CI=tools/ci
SPLIT=(--split-on '<|endoftext|>')

step() { printf '\n== %s ==\n' "$1"; }

step tokenize
python3 tokenizer/mini_bpe.py $CI/stories.txt -n 400 "${SPLIT[@]}" --save "$D/tok.bpe" -q
step encode
python3 tokenizer/mini_bpe.py --load "$D/tok.bpe" --encode $CI/stories.txt "${SPLIT[@]}" -o "$D/corpus.ids" -q
step "chat data"
python3 chat/make_chat.py $CI/stories.txt "${SPLIT[@]}" --min-stories 2 --skip-common 20 \
    --eval-fraction 0.25 -o "$D/chat.txt" --eval "$D/chat-eval.tsv" --eval-chat "$D/chat-eval.txt"
python3 tokenizer/mini_bpe.py --load "$D/tok.bpe" --encode "$D/chat.txt" "${SPLIT[@]}" \
    --allow-special -o "$D/chat.ids" -q
step build
$T build --tokenizer "$D/tok.bpe" --config $CI/tiny.toml -o "$D/init.safetensors"
step train
$T train --model "$D/init.safetensors" --data "$D/corpus.ids" --config $CI/tiny.toml -o "$D/model.safetensors"
step generate
$T generate --model "$D/model.safetensors" --tokenizer "$D/tok.bpe" --config $CI/tiny.toml \
    --prompt "Once upon a time" --max-tokens 20
step finetune
$T finetune --model "$D/model.safetensors" --data "$D/chat.ids" --tokenizer "$D/tok.bpe" \
    --config $CI/tiny.toml -o "$D/chat.safetensors"
step evaluate
$T evaluate --model "$D/chat.safetensors" --tokenizer "$D/tok.bpe" --config $CI/tiny.toml \
    --requests "$D/chat-eval.tsv" --paired "$D/chat-eval.txt" --chat --count 4
step probe
$T probe --model "$D/model.safetensors" --tokenizer "$D/tok.bpe" --sequences 2 \
    --prompt "Pip saw a red ball. Pip ran to the red ball. Pip saw a red ball."
step inspect
$T inspect "$D/chat.safetensors"
printf '\nThe whole flow ran.\n'

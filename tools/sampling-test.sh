#!/bin/bash
# Compare sampling = "random" and "epochs" on the small model: the same
# initial weights (out/model-init.safetensors) and data (out/corpus.ids),
# trained with [training] from config.toml, varying only the sampling and
# its seed (PLAN.md, "Where the windows come from"). Run from the top of the
# repository after `make train` (so out/ exists):
#   tools/sampling-test.sh                   # seeds 1-5, about 30 minutes on an M4
#   SEEDS="1 2 3" tools/sampling-test.sh
# Everything goes in out/sampling/ (make clean removes it). Ends with a
# table of the final validation losses.
set -euo pipefail
T=transformer/target/release/transformer
D=out/sampling
SEEDS=${SEEDS:-"1 2 3 4 5"}
STEPS=${STEPS:-}   # for a quick trial only: fewer [training] steps
for f in "$T" out/model-init.safetensors out/corpus.ids config.toml; do
    [ -e "$f" ] || { echo "missing $f: run make train first"; exit 1; }
done
mkdir -p "$D"

# config.toml's [training] with this sampling mode and seed (and STEPS).
config() {
    awk -v mode="$1" -v seed="$2" -v steps="$STEPS" '
        /^\[/ { sec = $0 }
        sec == "[training]" && /^seed[ =]/ { print "seed = " seed; next }
        sec == "[training]" && steps != "" && /^steps[ =]/ { print "steps = " steps; next }
        { print }
        $0 == "[training]" { print "sampling = \"" mode "\"" }
    ' config.toml
}

for seed in $SEEDS; do
    for mode in random epochs; do
        name="$mode-$seed"
        if [ -f "$D/$name.safetensors" ]; then
            echo "== $name: already done"
            continue
        fi
        config "$mode" "$seed" > "$D/$name.toml"
        echo "== $name"
        "$T" train --model out/model-init.safetensors --data out/corpus.ids \
            --config "$D/$name.toml" -o "$D/$name.tmp.safetensors" > "$D/$name.log"
        mv "$D/$name.tmp.safetensors" "$D/$name.safetensors"
        grep -E "sampling =|Trained to" "$D/$name.log" | sed 's/^/   /'
    done
done

# The final validation loss of each run, from its checkpoint's metadata.
loss() {
    "$T" inspect "$D/$1.safetensors" | awk '$1 == "valid_loss" { printf "%.3f", $2 }'
}
echo
printf "%-6s %8s %8s %8s\n" seed random epochs diff
for seed in $SEEDS; do
    r=$(loss "random-$seed"); e=$(loss "epochs-$seed")
    printf "%-6s %8s %8s %+8.3f\n" "$seed" "$r" "$e" "$(echo "$e - $r" | bc -l)"
done

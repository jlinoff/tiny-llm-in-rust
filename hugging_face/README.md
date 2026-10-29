Hugging Face model download.

```
mkdir hugging_face
cd hugging_face
pipenv install huggingface-hub
pipenv run hf download HuggingFaceTB/SmolLM-135M  --local-dir ./models/SmolLM-135
```

```
cd cd models/SmolLM-135M/
shasum -a 256 model.safetensors
gzip -9 -k model.safetensors
gzip -9 -k model.safetensors model.safetensors.gz
ls -lh
shasum -a 256 model.safetensors
```

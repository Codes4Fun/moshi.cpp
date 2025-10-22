
# moshi.cpp

A partial port of Kyutai's Moshi to C++ and ggml.
* https://github.com/kyutai-labs/moshi

As of right now, it's development is primarily for educational purposes. I'm using this to learn about AI, torch, ggml, and other libraries and tools.

## Status

Currently does a test of text to speech (tts), does not have a cli but in code these can be changed:
* The phrase can be changed.
* Voice can be changed.
* The seed for randomization can be changed.
* The model can be set to a 1.6b model with different voices, or 0.75b model.

On an RTX 4090 using the ggml-cuda backend, the 1.6b model performs at a rate of 3 seconds of output audio for about 1 second of generation time. So 15 seconds of audio takes 5 seconds to generate. The 0.75b model performs at a rate of 4 seconds of output audio for 1 second of generation time.

Still needs lots of optimization and refactoring.

I left out a lot of unused code, to save time, but also because there was no easy way for me to test everything.

## Data / Weights

It uses model data directly from kyutai's hugging face repository, with the 4 main model files here:
* https://huggingface.co/kyutai/tts-1.6b-en_fr/tree/main

This is a configuration file for the 1.6b model:
* https://huggingface.co/kyutai/tts-1.6b-en_fr/resolve/main/config.json?download=true

This is the tokenizer model file for sentencepiece:
* https://huggingface.co/kyutai/tts-1.6b-en_fr/resolve/main/tokenizer_spm_8k_en_fr_audio.model?download=true

This is the lm model file:
* https://huggingface.co/kyutai/tts-1.6b-en_fr/resolve/main/dsm_tts_1e68beda%40240.safetensors?download=true

This is the mimi decoder model file:
* https://huggingface.co/kyutai/tts-1.6b-en_fr/resolve/main/tokenizer-e351c8d8-checkpoint125.safetensors?download=true

All the voices are here:
* https://huggingface.co/kyutai/tts-voices/tree/main

The default voice used in this project:
* https://huggingface.co/kyutai/tts-voices/resolve/main/expresso/ex03-ex01_happy_001_channel1_334s.wav.1e68beda%40240.safetensors?download=true

with these files downloaded you should have a subdirectory that roughly looks like this:
```
kyutai/tts-1.6b-en_fr/config.json
kyutai/tts-1.6b-en_fr/dsm_tts_1e68beda@240.safetensors
kyutai/tts-1.6b-en_fr/tokenizer-e351c8d8-checkpoint125.safetensors
kyutai/tts-1.6b-en_fr/tokenizer_spm_8k_en_fr_audio.model
kyutai/tts-voices/expresso/ex03-ex01_happy_001_channel1_334s.wav.1e68beda@240.safetensors
```
with any additional voices and other files you may have downloaded.

Optionally there is a 0.75b model, that currently doesn't support voices or that I don't know yet how to use them. You only need 2 additional files, with the other two you can copy from the 1.6b directory.
* https://huggingface.co/kyutai/tts-0.75b-en-public/resolve/main/config.json?download=true
* https://huggingface.co/kyutai/tts-0.75b-en-public/resolve/main/dsm_tts_d6ef30c7%401000.safetensors?download=true

It's directory should look like this:
```
kyutai/tts-0.75b-en-public/config.json
kyutai/tts-0.75b-en-public/dsm_tts_d6ef30c7%401000.safetensors
kyutai/tts-0.75b-en-public/tokenizer-e351c8d8-checkpoint125.safetensors
kyutai/tts-0.75b-en-public/tokenizer_spm_8k_en_fr_audio.model
```

Presently to use that model, you will need to modify `main.cpp` by commenting out the 1.6b code line and uncommenting the 0.75b code line.

## Building and Running

It currently builds on linux. I have not tried any other platform.

It requires SentencePiece and ggml. If you already have these, you need to update test.mk to point towards their locations.

I tested with sentencepiece-0.2.0-Linux.7z, which can be downloaded here:
* https://github.com/google/sentencepiece/releases/tag/v0.2.0
and copied libsentencepiece.so.0 to the root directory of this project.

On windows with msys2 I found I needed to build sentencepiece myself.

If you plan to test this just with the cpu, you can build with the current version of ggml:
* https://github.com/ggml-org/ggml

But if you want to use cuda, you will need my modified version of ggml that adds hardware support for larger ggml_top_k tensors:
https://github.com/Codes4Fun/ggml/tree/for_moshi

so in the root of this project directory you should end up with these files:
```
libggml-base.so
libggml-cpu.so
libggml-cuda.so
libggml.so
libsentencepiece.so.0
```
the libggml-cuda.so is optional if you used my ggml.

and header files in these subdirectories:
```
ggml/include/
sentencepiece-0.2.0-Linux/include/
```

you can also edit the test.mk file, but from here you can just use this command:
```
make -f test.mk run
```
that will build and then run the test that will generate a file `audio_test.wav`.



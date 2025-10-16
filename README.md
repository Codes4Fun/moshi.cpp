
# moshi.cpp

A partial port of Kyutai's Moshi to C++ and ggml.
https://github.com/kyutai-labs/moshi

As of right now, it is for educational purposes. I'm using this to learn about AI, torch, and ggml.

## Status

Currently does a test of text to speech (tts), does not have a cli but in code these can be changed:
* The phrase can be changed.
* Voice can be changed.
* The seed for randomization can be changed.

On an RTX 4090 using the ggml-cuda backend, it performs at a rate of 3 seconds of output audio for about 1 second of generation time. So 15 seconds of audio takes 5 seconds to generate.

This code is a refactor of another project that more invasively modified the original moshi python code, and has a bug where the audio is slightly off, that I need to investigate.

Still needs lots of optimization and refactoring.

I left out a lot of unused code, to save time, but also because there was no easy way for me to test everything.

## Data / Weights

It uses model data directly from kyutai's hugging face repository, with the 3 main model files here:
https://huggingface.co/kyutai/tts-1.6b-en_fr/tree/main

This is the tokenizer model file for sentencepiece:
https://huggingface.co/kyutai/tts-1.6b-en_fr/resolve/main/tokenizer_spm_8k_en_fr_audio.model?download=true

This is the lm model file:
https://huggingface.co/kyutai/tts-1.6b-en_fr/resolve/main/dsm_tts_1e68beda%40240.safetensors?download=true

This is the mimi decoder model file:
https://huggingface.co/kyutai/tts-1.6b-en_fr/resolve/main/tokenizer-e351c8d8-checkpoint125.safetensors?download=true

The voices are here:
https://huggingface.co/kyutai/tts-voices/tree/main

The default voice:
https://huggingface.co/kyutai/tts-voices/resolve/main/expresso/ex03-ex01_happy_001_channel1_334s.wav.1e68beda%40240.safetensors?download=true

with these files downloaded you should have a subdirectory that roughly looks like this:

kyutai/tts-1.6b-en_fr/dsm_tts_1e68beda@240.safetensors
kyutai/tts-1.6b-en_fr/tokenizer-e351c8d8-checkpoint125.safetensors
kyutai/tts-1.6b-en_fr/tokenizer_spm_8k_en_fr_audio.model
kyutai/tts-voices/expresso/ex03-ex01_happy_001_channel1_334s.wav.1e68beda@240.safetensors

with any additional voices and other files you may have downloaded.

## Building

It currently builds on linux. I have not tried any other platform.

It requires SentencePiece and ggml. If you already have these, you need to update test.mk to point towards their locations.

I tested with sentencepiece-0.2.0-Linux.7z, which can be downloaded here:
https://github.com/google/sentencepiece/releases/tag/v0.2.0
and copied libsentencepiece.so.0 to the root directory of this project.

I pulled the source code to ggml from it's repository here:
https://github.com/ggml-org/ggml
and used cmake to build it with cuda and vulkan enabled and copied the libraries files to the root directory of this project.

so in the root of this project directory you should end up with these files:
libggml-base.so
libggml-cpu.so
libggml-cuda.so
libggml.so
libggml-vulkan.so
libsentencepiece.so.0

and header files in these subdirectories:
ggml/include/
sentencepiece-0.2.0-Linux/include/

you can also edit the test.mk file, but from here you can just use this command:

make -f test.mk run

that will build and then run the test that will generate a file audio_test.wav.



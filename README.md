
# moshi.cpp

A partial port of Kyutai's Moshi to C++ and ggml.
* https://github.com/kyutai-labs/moshi

As of right now, it's development is primarily for learning and development. It's being done to learn about AI, torch, ggml, and other libraries and tools.

## Status

Currently does a test of tts (text to speech) (see: `main.cpp`), does not have a cli but in code these can be changed:
* The phrase can be changed.
* Voice can be changed.
* The seed for randomization can be changed.
* The model can be set to a 1.6b model with different voices, or 0.75b model.

On an RTX 4090 using the ggml-cuda backend, the 1.6b model performs at a rate of 3 seconds of output audio for about 1 second of generation time. So 15 seconds of audio takes 5 seconds to generate.

The 0.75b model, does not yet have an option to trim the prefix so it will not start with the prompt, but also I think there is an error in the transition between the prefix audio and the prompt generated audio. The prefix only supports 24khz or 48khz sample rate wav files at 10 seconds.

Still needs lots of optimization and refactoring. Right now, for example, it rebuilds the graph for each frame.

I left out a lot of unused code, to save time, but also because there was no easy way for me to test everything, as I add more tests, I will port more code and features.

## Data / Weights

There are two tts models to choose from, one is 1.6b and the other is 0.75b.

The 1.6b model uses cross attention and requires specially made weights for voices, while the 0.75b model uses wav files to start inference, just as you would a system prompt. This means you can only choose specially made voices for the 1.6b model, while you can in theory use any wav file sample of a voice you want to match.

I recommend downloading the contents of these 3 hugging face repositories:
* https://huggingface.co/kyutai/tts-1.6b-en_fr/tree/main
* https://huggingface.co/kyutai/tts-0.75b-en-public/tree/main
* https://huggingface.co/kyutai/tts-voices/tree/main

The files should look like this, with tts-voices having a lot of voice samples and weights.
```
kyutai/tts-1.6b-en_fr/config.json
kyutai/tts-1.6b-en_fr/dsm_tts_1e68beda@240.safetensors
kyutai/tts-1.6b-en_fr/tokenizer-e351c8d8-checkpoint125.safetensors
kyutai/tts-1.6b-en_fr/tokenizer_spm_8k_en_fr_audio.model
kyutai/tts-0.75b-en-public/config.json
kyutai/tts-0.75b-en-public/dsm_tts_d6ef30c7%401000.safetensors
kyutai/tts-0.75b-en-public/tokenizer-e351c8d8-checkpoint125.safetensors
kyutai/tts-0.75b-en-public/tokenizer_spm_8k_en_fr_audio.model
kyutai/tts-voices/*
```

Currently to switch models and voices, you will need to modify `main.cpp` by commenting out the 1.6b code line and uncommenting the 0.75b code line.

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



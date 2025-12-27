
# moshi.cpp

A port of Kyutai's Moshi to C++ and ggml.
* https://github.com/kyutai-labs/moshi

As of right now, it exists primarily for learning and development. It's being done to learn about AI, torch, ggml, and other libraries and tools.

## Status

There are multiple tools that demo different components:
* mimi-encode - demonstrates using mimi to encode different inputs to a mimi file
* mimi-decode - demonstrates using mimi to decode and output different files
* mimi-play - decodes mimi files and plays them through sdl
* mimi-echo - realtime demo that allows you to hear mimi compression
* moshi-tts - demonstrates text inputs to audio outputs
* moshi-stt - demonstrates audio inputs to text outputs
* moshi-sts - demonstrates audio inputs to audio (and text) outputs

TODO: finish a download tool, GGUF/quantization support, integrate llama.cpp to implement an unmute like program, add a gui.

### Performance / Optimizations

On an RTX 4090 using the ggml-cuda backend, the 1.6b model performs at a rate of 3 seconds of output audio for about 1 second of generation time. So 15 seconds of audio takes 5 seconds to generate.

Still needs lots of optimization and refactoring. Right now, for example, it rebuilds graphs each frame.

I left out a lot of unused code from the original, to save time, but also because there was no easy way for me to test all of it, as I add more demos, I will port more code and features over.

I did create an optimization that does not exist in moshi, and that is, instead of generating an attention bias mask each frame, it generates a reusable pattern once at initialization, and reuses it like you would a lookup table. Not only does this reduce the work to just changing an offset in the pattern tensor, but it makes easier an implementation that originally involved boolean logic operations and dealing with infinities.

## Build Dependencies

The moshi library depends on:
* SentencePiece (tested with 0.2.0)
* GGML

The tools additionally depend on:
* FFmpeg (7+)
* SDL2

### Sentence Piece

SentencePiece has only been tested using static linking built from source:
* https://github.com/google/sentencepiece/releases/tag/v0.2.0

### GGML

If you plan to build vulkan you should use my modified version of ggml:
* https://github.com/Codes4Fun/ggml

otherwise you can use the official version:
* https://github.com/ggml-org/ggml

Example build with cuda and vulkan:
```
git clone --branch for_moshi --single-branch https://github.com/codes4fun/ggml
cd ggml
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DGGML_CUDA=ON -DGGML_VULKAN=ON
```
You might need to set `CMAKE_CUDA_COMPILER` to where nvcc is located, and `Vulkan_GLSLC_EXECUTABLE` to where glslc is located. Using a newer version of CMake (4.1+) can usually resolve that.

### FFmpeg

For FFmpeg it requires a newer version than most linux package systems include, it can be built from source, or you can use binaries for linux or windows here:
* https://github.com/BtbN/FFmpeg-Builds/releases

I've tested the `ffmpeg-master-latest-*-lgpl-shared` versions.

Other download options at the official site: https://ffmpeg.org/download.html

### SDL2

For SDL2, it can be installed using standard package managers, for Ubuntu:
```
sudo apt install libsdl2-dev
```
And windows SDL2 devel libraries (SDL2-devel-2.30.11-VC.zip) can be downloaded here :
* https://github.com/libsdl-org/SDL/releases/tag/release-2.30.11

## Building

With dependencies in place you can use cmake by first cloning this repository and then creating a build directory:
```
git clone https://github.com/codes4fun/moshi.cpp
cd moshi.cpp
mkdir build
cd build
```
and then generate a build using cmake, which for example on windows would look like this (changing generation target and paths as needed):
```
cmake .. -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=RelWithDebInfo -DGGML_INCLUDE_DIR=C:/repos/ggml/include -DGGML_LIBRARY_DIR=C:/repos/ggml/build/src -DSentencePiece_INCLUDE_DIR=C:/repos/sentencepiece/src -DSentencePiece_LIBRARY_DIR=C:/repos/sentencepiece/build/src -DCMAKE_PREFIX_PATH=C:\lib\SDL2-2.30.11 -DFFmpeg_DIR=C:\lib\ffmpeg-master-latest-win64-lgpl-shared
```
or Ubuntu, change the paths if necessary:
```
cmake .. \
 -DGGML_INCLUDE_DIR=~/repos/ggml/include\
 -DGGML_LIBRARY_DIR=~/repos/ggml/build/src\
 -DSentencePiece_INCLUDE_DIR=~/repos/sentencepiece/include\
 -DSentencePiece_LIBRARY_DIR=~/repos/sentencepiece/lib\
 -DFFmpeg_DIR=~/lib/ffmpeg-master-latest-linux64-lgpl-shared
```

And finally build it.
```
cmake --build .
```
That will create a bin directory under build. You will need to copy over ggml libraries, and if needed the ffmpeg libraries. On windows you will need to also copy over sdl2.

## Data / Weights

There are two tts models to choose from, one is 1.6b and the other is 0.75b.

The 1.6b tts model uses cross attention and requires specially made weights for voices, while the 0.75b tts model uses wav files to start inference, just as you would a system prompt. This means you can only choose specially made voices for the 1.6b model, while you can in theory use any wav file sample of a voice you want to match. The downside to 0.75b model is it only seems to be able to generate 10 seconds of voice before falling silent.

The 1b-en_fr-candle stt model has support for vad (voice activity detection), while the larger 2.6b stt model does not.

I recommend downloading the contents of these 5 hugging face repositories:
* https://huggingface.co/kyutai/tts-1.6b-en_fr/tree/main
* https://huggingface.co/kyutai/tts-0.75b-en-public/tree/main
* https://huggingface.co/kyutai/tts-voices/tree/main
* https://huggingface.co/kyutai/stt-1b-en_fr-candle
* https://huggingface.co/kyutai/stt-2.6b-en

But if you want to download a minimum, then I recommend the `tts-1.6b-en_fr`, `stt-1b-en_fr-candle` and `tts-voices/expresso/ex03-ex01_happy_001_channel1_334s.wav.1e68beda@240.safetensors`, those are the defaults.

The result file paths should look like this, with tts-voices having a lot of voice samples and weights.
```
kyutai/tts-1.6b-en_fr/config.json
kyutai/tts-1.6b-en_fr/dsm_tts_1e68beda@240.safetensors
kyutai/tts-1.6b-en_fr/tokenizer-e351c8d8-checkpoint125.safetensors
kyutai/tts-1.6b-en_fr/tokenizer_spm_8k_en_fr_audio.model
kyutai/tts-voices/expresso/ex03-ex01_happy_001_channel1_334s.wav.1e68beda@240.safetensors
kyutai/stt-1b-en_fr-candle/config.json
kyutai/stt-1b-en_fr-candle/mimi-pytorch-e351c8d8@125.safetensors
kyutai/stt-1b-en_fr-candle/model.safetensors
kyutai/stt-1b-en_fr-candle/tokenizer_en_fr_audio_8000.model
```

If you have an RTX 4090 or newer card with 20GB or more vram, you can try out the speech-to-speech model, moshika.
* https://huggingface.co/kyutai/moshika-pytorch-bf16


# Benchmarks

A simple way to do benchmarking is to first generate a wav using moshi-tts and then use that wav with moshi-stt. If you store the models in a separate directory, set the environment variable MODEL_CACHE to the root directory containing kyutai folder to make it easier. You can use this command for benchmarking text-to-speech:

```
./moshi-tts --bench
```

This will default to "The quick brown fox jumped over the sleeping dog." and disables output, sets the seed to 0 and temperature to 0 for consistent results. If you have a specific device you want to benchmark, you can get a list via `./moshi-tts -l` and to target a specific device (like `CUDA0`, `Vulkan0`, or `CPU`) and/or want to set the number of threads, you can modify the command like this:

```
./moshi-tts --bench -d CPU --threads 8
```

For benchmarking speech-to-text, you need an audio input file first, which I would recommend generating by adding an output file to the tts bench option:

```
./moshi-tts --bench -o test.wav
```

Then you can use test.wav to run stt.

```
./moshi-stt -i test.wav
```

These commands output frames per second. Although tts also outputs tokens per second, that is for reference since token pronouncation can take variable frames to compute.

Moshi operates at 12.5 frames per second, so anything below that would not work for real time applications.

CUDA benchmarks:

| make   | name             | driver | tts fps | stt fps |
|--------|------------------|--------|---------|---------|
| NVIDIA | RTX 4090 Ti      | CUDA   |   40.07 |  101.99 |
| NVIDIA | GTX 2070         | CUDA   |   15.49 |   60.84 |
| NVIDIA | GTX 1070         | CUDA   |    7.04 |   23.40 |

Vulkan benchmarks:

| make   | name             | driver | tts fps | stt fps |
|--------|------------------|--------|---------|---------|
| NVIDIA | RTX 4090 Ti      | Vulkan |   23.26 |   43.41 |
| NVIDIA | GTX 2070         | Vulkan |   13.39 |   21.83 |
|    AMD | Radeon 8060S     | Vulkan |    8.38 |   19.23 |
|    AMD | Radeon 780M      | Vulkan |    5.98 |   18.43 |
| NVIDIA | GTX 1070         | Vulkan |    4.12 |   15.59 |
|    AMD | Radeon 890M      | Vulkan |    4.15 |    9.42 |
|  Intel | UHD Graphics 630 | Vulkan |    0.90 |    2.67 |

CPU benchmarks:

| make  | name              | driver | tts fps | stt fps | threads |
|-------|-------------------|--------|---------|---------|---------|
|   AMD | Ryzen AI MAX+ 395 | CPU    |    4.24 |    8.36 |       8 |
|   AMD | Ryzen AI 9 HX370  | CPU    |    4.18 |    7.48 |       8 |
|   AMD | Ryzen 7 8845HS    | CPU    |    3.71 |    6.77 |       8 |
|   AMD | Ryzen 7 8840U     | CPU    |    2.89 |    6.45 |       8 |
| Intel | Core i7-8750H     | CPU    |    2.73 |    5.03 |       6 |
| Intel | Core i7-9750H     | CPU    |    2.54 |    5.09 |       6 |
| Intel | Core i7-6700T     | CPU    |    1.62 |    3.04 |       4 |

## Design Notes

I was originally looking at designing the API after gstreamer and/or potentially integrating it with it, but I found gstreamer was rather hard to debug when things didn't work and they immediately didn't work. I still like the idea of pipes, but I decided to follow how FFmpeg connects decoders resamplers and encoders. I am not entirely set on this, as I have lots of other ideas, such as both streaming to SDL and being able to record to an mp3 file, but also in the future it may make sense for data to stay on the GPU as long as it can, so rather hiding how things are connected would make sense.

Internally I tried to replicate what the original moshi did by using single header files for code, following it's file hierarchy. To make it easier for anyone interested to compare python to c++.

My coding style is a combination of C++ and C, largely because C++ through deep abstraction can make it hard to debug, read, maintain, and refactor code. So I try to keep abstractions shallow, mostly used for reducing code bloat with automation. There are other misc things I do primarily for readability.

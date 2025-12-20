
# moshi.cpp

A partial port of Kyutai's Moshi to C++ and ggml.
* https://github.com/kyutai-labs/moshi

As of right now, it's development is primarily for learning and development. It's being done to learn about AI, torch, ggml, and other libraries and tools.

## Status

There are multiple tools that demo different components:
* mimi-encode - demonstrates using mimi to encode different inputs to a mimi file
* mimi-decode - demonstrates using mimi to decode and output different files
* mimi-play - decodes mimi files and playsback through sdl
* mimi-echo - realtime demo that allows you to hear mimi compression
* moshi-tts - demonstrates text inputs to audio outputs
* moshi-stt - demonstrates audio inputs to text outputs

TODO: finish a download tool, integrate llama.cpp to implement an unmute like program, add a gui.

### Performance / Optimizations

On an RTX 4090 using the ggml-cuda backend, the 1.6b model performs at a rate of 3 seconds of output audio for about 1 second of generation time. So 15 seconds of audio takes 5 seconds to generate.

Still needs lots of optimization and refactoring. Right now, for example, it rebuilds the graph for each frame.

I left out a lot of unused code, to save time, but also because there was no easy way for me to test everything, as I add more tests, I will port more code and features.

I did create an optimization that does not exist in moshi, and that is instead of generate an attention bias mask each frame, to generate a reusable pattern at initialization, the equivalent of a lookup table. Not only does this reduce the work to just changing an offset in the pattern tensor, but it also allows it to function with vulkan in ggml, which does not support negations that I originally used to replicate what moshi was doing.

## Building

The library depends on SentencePiece (tested with 0.2.0) and GGML. The tools add additional dependences FFmpeg (7+) and SDL2.

SentencePiece has only been tested using static linking built from source:
* https://github.com/google/sentencepiece/releases/tag/v0.2.0

GGML can be built from my version, modified to work with vulkan:
* https://github.com/Codes4Fun/ggml

or the offical repository:
* https://github.com/ggml-org/ggml

Example build with cuda and vulkan:
```
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE:STRING=Release -DGGML_BACKEND_DL:BOOL=ON -DGGML_CPU_ALL_VARIANTS:BOOL=ON -DGGML_CUDA:BOOL=ON -DGGML_VULKAN:BOOL=ON
```
You might need to set CMAKE_CUDA_COMPILER to where nvcc is located, and Vulkan_GLSLC_EXECUTABLE to where glslc is located.

For FFmpeg, older versions of Ubutnu will not have the latest, it can be built from source, or both linux and windows binaries from here:
* https://github.com/BtbN/FFmpeg-Builds/releases

For SDL2, it can be installed using standard package managers, for Ubuntu:
```
sudo apt install libsdl2-dev
```
And for MSYS2 MinGW x64:
```
pacman -S mingw-w64-x86_64-SDL2
```
And windows SDL2 devel libraries (SDL2-devel-2.30.11-VC.zip) can be downloaded here :
* https://github.com/libsdl-org/SDL/releases/tag/release-2.30.11


With most of these in place you can use cmake by first creating a build directory:
```
mkdir build
cd build
```
and then generate a build on windows for example if using nmake, change generation target and paths as needed:
```
cmake .. -G "NMake Makefiles" -DCMAKE_BUILD_TYPE:STRING=RelWithDebInfo -DGGML_INCLUDE_DIR=C:/repos/ggml/include -DGGML_LIBRARY_DIR=C:/repos/ggml/build/src -DSentencePiece_INCLUDE_DIR=C:/repos/sentencepiece/src -DSentencePiece_LIBRARY_DIR=C:/repos/sentencepiece/build/src -DCMAKE_PREFIX_PATH=C:\lib\SDL2-2.30.11 -DFFmpeg_DIR=D:\lib\ffmpeg-master-latest-win64-lgpl-shared
```
or Ubuntu, change the paths if necessary:
```
cmake .. \
 -DGGML_INCLUDE_DIR=~/repos/ggml/include\
 -DGGML_LIBRARY_DIR=~/repos/ggml/build/src\
 -DSentencePiece_INCLUDE_DIR=~/repos/sentencepiece/include\
 -DSentencePiece_LIBRARY_DIR=~/repos/sentencepiece/lib\
 -DFFmpeg_DIR=~/lib/ffmpeg-master-latest-win64-lgpl-shared
```

And finally build it.
```
cmake --build .
```
That will create a bin directory under build. You will need to copy over ggml libraries, and if needed the ffmpeg libraries. On windows you will need to copy over ffmpeg, sdl2.

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

kyutai/tts-0.75b-en-public


https://huggingface.co/api/models/kyutai/tts-0.75b-en-public/tree/main

https://huggingface.co/api/models/microsoft/VibeVoice-Realtime-0.5B/tree/main


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

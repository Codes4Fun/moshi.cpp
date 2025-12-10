
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

The library depends on SentencePiece (tested with 0.2.0) and GGML.

The tools add additional dependences FFmpeg (8+) and SDL2.

Builds have only been tested under Ubuntu and Windows MSYS2 MinGW x64.

SentencePiece has only been tested using static linking, either built from source or binaries from:
* https://github.com/google/sentencepiece/releases/tag/v0.2.0
I found for MSYS2 MinGW x64, I've had to build it myself because of toolchain differences that show up as undefined references.

GGML can be built from:
* https://github.com/Codes4Fun/ggml
or the offical repository:
* https://github.com/ggml-org/ggml
My custom version of GGML will get vulkan working.

For FFmpeg, older versions of Ubutnu will not have the latest, it can be built from source or binaries from:
* https://github.com/BtbN/FFmpeg-Builds/releases
MSYS2 MinGW x64 has the latest and can be installed via:
```
pacman -S mingw-w64-x86_64-ffmpeg
```

For SDL2, it can be installed using standard package managers, for Ubuntu:
```
sudo apt install libsdl2-dev
```
And for MSYS2 MinGW x64:
```
pacman -S mingw-w64-x86_64-SDL2
```

With most of these in place you can use cmake by first creating a build directory:
```
mkdir build
cd build
```
and then with updated folder locations below you can do something like MYS2 MinGW:
```
/mingw64/bin/cmake .. -G "MinGW Makefiles"\
 -DGGML_INCLUDE_DIR=C:/repos/ggml/include\
 -DGGML_LIBRARY_DIR=C:/repos/ggml/build/src\
 -DSentencePiece_INCLUDE_DIR=C:/repos/sentencepiece/include\
 -DSentencePiece_LIBRARY_DIR=C:/repos/sentencepiece/lib
```
or Ubuntu:
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
That will create a bin directory under build. You will need to copy over ggml libraries, and if needed the ffmpeg libraries. On windows you will need to copy over ffmpeg, sdl2, and other libraries which you can find out by using `ldd` on the binaries.

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

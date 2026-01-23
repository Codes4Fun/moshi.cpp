
# moshi.cpp

A port of Kyutai's Moshi to C++ and ggml.
* https://github.com/kyutai-labs/moshi

## Quick Start (Linux)

Make sure you have relatively recent drivers for linux.

Download a [binary release](https://github.com/Codes4Fun/moshi.cpp/releases) for linux and extract somewhere.

Open a terminal to where the files are extracted.

Install some additional dependencies:
```
sudo apt install aria2 libsdl2-2.0-0
```

Download the models ( about 9.7GB ):
```
aria2c --disable-ipv6 -i moshi-defaults.txt
```

Run moshika, a hallucinating speech-to-speech model, requires microphone, ask her "What are you doing?":
```
./moshi-sts
```

Run speech-to-text, requires microphone:
```
./moshi-stt
```

Run text-to-speech:
```
./moshi-tts "Hello World!"
```

## Quick Start (Windows)

Make sure you have relatively recent drivers and have the latest [msvc runtimes](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170).

Download a [binary release](https://github.com/Codes4Fun/moshi.cpp/releases) for windows and extract somewhere.

Open a command line ( window + r keys, open 'cmd' ) or a PowerShell, and navigate to where the files are extracted.

Download the models ( about 9.7GB ):
```
.\aria2c --disable-ipv6 -i moshi-defaults.txt
```

Run moshika, a hallucinating speech-to-speech model, requires microphone, ask her "What are you doing?":
```
.\moshi-sts
```

Run speech-to-text, requires microphone:
```
.\moshi-stt
```

Run text-to-speech:
```
.\moshi-tts "Hello World!"
```

## Status

There are multiple tools that demonstrate different components:
* mimi-encode - demonstrates using mimi to encode different inputs to a mimi file
* mimi-decode - demonstrates using mimi to decode and output different files
* mimi-play - decodes mimi files and plays them through sdl
* mimi-echo - realtime demo that allows you to hear mimi compression
* moshi-tts - demonstrates text inputs to audio outputs
* moshi-stt - demonstrates audio inputs to text outputs
* moshi-sts - demonstrates audio inputs to audio (and text) outputs

There are aria2c download scripts to make it easier to download tested models.

The tools support quantization of the safetensor models and caching of gguf files via commmand line, `-g` to cache a gguf which is several times faster to load than the safetensors but will consume more drive space. Use `-q q8_0` or `-q q4_k` to quantize, the q4_k can take a while to convert, several minutes for some models, so it's best to use those with `-g` to save gguf versions, they also perform a bit faster. The largest models, moshika and moshiko, can run on 8gb of vram with q4_k, but they may not perform fast enough, though I was able to have a conversation with an rtx 2070 laptop running linux.

### Performance / Optimizations

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

To make downloading models easier, I have provided aria2 input files that will automatically download and verify the downloaded files. You can install aria2 either by downloading from https://github.com/aria2/aria2/releases/tag/release-1.37.0 or using a package manager like apt:
```
sudo apt install aria2
```
or pacman
```
sudo pacman -S aria2
```
For windows you can unzip the aria2c.exe into the moshi directory.

Aftwards you can run the following which will download and verify the minimal files to run moshi-tts and moshi-stt. This requires about 9.7 GB of space:
```
aria2c --disable-ipv6 -i moshi-defaults.txt
```

If you want your models to be located in another directory, ideally set it's path in an environment variable named `MODEL_CACHE` and then add to the command line `-d`, so for example in linux use `-d $MODEL_CACHE` or in windows `-d %MODEL_CACHE%`.

If you wish to download all available voices, 731 MB, run aria command again but change the last part from `-i moshi-defaults.txt` to `-i kyutai_tts-voices.txt`.

These are the available aria2 download scripts:
 * moshi-defaults.txt - 9.7GB downloads files necessary to run all demos.
 * kyutai_tts-voices.txt - 731 MB, all tts-1.6b and tts-0.75b voices
 * Codes4Fun_moshi-common.txt - files shared between models.
 * Codes4Fun_moshika-q4_k-GGUF.txt - Kyutai's Moshika model in quantized gguf format.
 * Codes4Fun_stt-1b-en_fr-GGUF.txt - Kyutai's STT 1B model in gguf format.
 * Codes4Fun_tts-1.6b-en_fr-GGUF.txt - Kyutai's TTS 1.6B model in gguf format.

 These are additional aria2 download scripts, they are here for reference. they can be used but may not performed well unless converted/quantized:
 * kyutai_stt-1b-en_fr-candle.txt - downloaded as part of default.
 * kyutai_stt-2.6b-en.txt - 6 GB, large model without vad but better quality.
 * kyutai_tts-0.75b-en-public.txt - 2 GB, small model that uses audio files for voices.
 * kyutai_tts-1.6b-en_fr.txt - downloaded as part of default.
 * kyutai_moshika-pytorch-bf16.txt - 16 GB female model 
 * kyutai_moshiko-pytorch-bf16.txt - 16 GB male model

# Running Demos

After downloading/building moshicpp , you can see a list of device options with the `-l` option, for example `moshi-tts -l` should output a list of devices. If no output shows up, make sure you have the latest msvc redistributables installed:
* https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170

After downloading the default models (see the Data/Weights section), you should be able start generating speech using the default tts model and voice:
```
moshi-tts "She sells sea shells by the sea shore."
```

If you installed the data to a different directory, you can specify the root location with command line argument `-r` or by setting the environment variable `MODEL_CACHE` to where the models reside, for example if tts model is located at `C:/models/kyutai/tts-1.6b-en_fr` then you could use `-r C:/models` or set `MODEL_CACHE` to `C:\models` and not need the command line option.

If you run into an error about PTX being compiled by an unsupported toolchain, try updating the nvidia drivers.

If for some reason SDL isn't outputing audio or you want to generate a mp3 file, or other media file format that ffmpeg supports, you can use the output option `-o` like so:
```
moshi-tts "She sells sea shells by the sea shore." -o seashells.mp3
```

To demo the stt, using the microphone:
```
moshi-stt
```
or an input media file:
```
moshi-stt -i seashells.mp3
```

If you get an error, make sure the microphone is working, and if in windows make sure desktop apps can access it via the "Microphone privacy settings".

To talk to moshika (not part of the default download), if you have 20gb vram, you can use:
```
moshi-sts
```
If you have less than 20gb and at least 8gb of vram, or performance is a bit low, you can quantize the model down and cache it using this command:
```
moshi-sts -g -q q4_k
```
That will consume about 4gb of additional disk space, and takes several minutes to convert the model, but after the initial creation, starting moshi will take seconds.

If you plan to use these models multiple times, it is recommened to use the `-g` option, it will take up more drive space but will load several times faster. You can experiment with quantization of the other models as well: `-q q8_0` `-q q4_k`.

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

For benchmarking speech-to-speech (sts), you can use the `--bench` option, this will disable sdl audio input/output to run the model as fast as possible for only 125 frames, which can take between 10 to 40 seconds. For the fastest speed with sts it is recommended to use the `-g -q q4_k` options which will take an addition 4gb of disk space and take several minutes the first run, but after the first run it will loads in seconds, and consumes less than 8gb of vram.
```
./moshi-sts --bench -g -q q4_k
```

These commands output frames per second. Although tts also outputs tokens per second, that is for reference since token pronouncation can take variable frames to compute.

Moshi operates at 12.5 frames per second, so anything below that would not work for real time applications.

CUDA benchmarks (beta2):
| make   | name            | gb | driver | os    | tts fps | stt fps | sts q4_k |
|--------|-----------------|----|--------|-------|---------|---------|----------|
| NVIDIA | RTX 4060        |  8 | CUDA   | linux |   19.41 |   76.63 | 🟢 17.85 |
| NVIDIA | RTX 3060        | 12 | CUDA   | linux |   17.98 |   78.02 | 🟢 17.82 |
| NVIDIA | RTX 2070 Laptop |  8 | CUDA   | linux |   18.84 |   83.08 | 🟢 16.89 |
| NVIDIA | RTX 2070 Laptop |  8 | CUDA   | win10 |   16.96 |   59.56 | 🟢 14.75 |
| NVIDIA | RTX 4060        |  8 | CUDA   | win11 |   14.14 |   42.37 | 🟢 13.44 |
| NVIDIA | RTX 3060        | 12 | CUDA   | win11 |   13.80 |   42.44 | 🟢 12.79 |
| NVIDIA | GTX 1070        |  8 | CUDA   | win11 |    8.72 |   41.81 | 🔴  6.94 |

Vulkan benchmarks (beta2):
| make   | name              | gb | driver | os    | tts fps | stt fps | sts q4_k |
|--------|-------------------|----|--------|-------|---------|---------|----------|
|  Intel | ARC B850          | 12 | Vulkan | win11 |   31.43 |   63.88 | 🟢 22.03 |
|    AMD | Radeon RX 6700 XT | 12 | Vulkan | win11 |   22.46 |   56.70 | 🟢 19.17 |
|    AMD | Radeon RX 6700 XT | 12 | Vulkan | linux |   20.35 |   58.32 | 🟢 17.84 |
|  Intel | ARC B850          | 12 | Vulkan | linux |   19.88 |   44.49 | 🟢 16.45 |
|    AMD | Radeon 8060S      | 64 | Vulkan | linux |   13.15 |   43.57 | 🟢 15.47 |
|    AMD | Radeon 8060S      | 64 | Vulkan | win11 |   12.34 |   37.16 | 🟢 15.05 |
|    AMD | Radeon 890M HX370 | 16 | Vulkan | linux |    7.50 |   23.83 | 🔴  6.60 |
|    AMD | Radeon 890M HX370 | 16 | Vulkan | win11 |    7.53 |   21.65 | 🔴  5.80 |

CPU benchmarks (alpha):
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

# Critique of GGML

I would like for the API to decide if it's going to be a data flow graph or a operation dependency graph, as of present, it seems to want to be both which creates confusion. For example many operations are 'inplace', (some are 'inplace' and are not described as 'inplace' such as the clamp but that is besides the point), having 'inplace' operations at all would suggest ggml graphs are more operation dependency graphs, since anything that uses a tensor after an inplace operation will use the modified tensor and not the tensor before the inplace.

If the API is going to be an operation dependency graph, it should go all the way and have targeted outputs instead of returning the output, this way I can create the output tensor ahead of time and the operation on it later. So basically instead of this:

```
auto result = ggml_add( ctx, left, right );
```

```
auto result = ggml_dup_tensor( ctx, left );
ggml_add( ctx, result, left, right );
```

The reason this matters is for the same reason why the 'inplace' operations exist, it allows for more memory reuse, except instead of being limited to single special operations, it can be done to a larger string of operations, if the input to a graph has the same shape as the output then you could reuse the same tensor and save some memory in the process.

This can also allow for a concept of stack memory, tensors can be created temporarily to complete a set of operations and then the memory reused, while the results of the operation are retained by the caller of the operation.

The problem with this idea is what if I don't know what the shape is of my tensor ahead of time, maybe it is too complex to calculate and many input variables will decide the resulting shape. Then maybe the solution is to have a place holder tensor that gets allocated on compute. For example:

```
auto result = ggml_new_tensor_unshaped( ctx, GGML_TYPE_F32 );
my_complex_operations( ctx, result, left, right );
```

I would consider that more of a feature then a requirement, but it also detaches tensors from their data, which happens with backend contexts anyway.



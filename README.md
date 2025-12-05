
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

The next steps would be to setup cmake, and then wrap more code in an api and make an official library. After that in no particular order: finish a download tool, integrate llama.cpp to implement an unmute like program, implement a gui.

### Performance / Optimizations

On an RTX 4090 using the ggml-cuda backend, the 1.6b model performs at a rate of 3 seconds of output audio for about 1 second of generation time. So 15 seconds of audio takes 5 seconds to generate.

Still needs lots of optimization and refactoring. Right now, for example, it rebuilds the graph for each frame.

I left out a lot of unused code, to save time, but also because there was no easy way for me to test everything, as I add more tests, I will port more code and features.

I did create an optimization that does not exist in moshi, and that is instead of generate an attention bias mask each frame, to generate a reusable pattern at initialization, the equivalent of a lookup table. Not only does this reduce the work to just changing an offset in the pattern tensor, but it also allows it to function with vulkan in ggml, which does not support negations that I originally used to replicate what moshi was doing.

## Building

Work in progress.

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

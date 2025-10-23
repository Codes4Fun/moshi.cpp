
HEADERS=\
 src/config.h\
 src/context.h\
 src/ggml_wrap.h\
 src/json.h\
 src/loader.h\
 src/moshi/models/compression.h\
 src/moshi/models/lm.h\
 src/moshi/models/lm_default.h\
 src/moshi/models/lm_utils.h\
 src/moshi/models/tts.h\
 src/moshi/modules/conv.h\
 src/moshi/modules/gating.h\
 src/moshi/modules/rope.h\
 src/moshi/modules/seanet.h\
 src/moshi/modules/transformer.h\
 src/moshi/quantization/core_vq.h\
 src/moshi/quantization/vq.h\
 src/moshi/utils/sampling.h\
 src/safetensor.h\
 src/torch.h\
 src/wav.h\

CPP_FLAGS=\
 -I ggml/include\
 -I sentencepiece-0.2.0-Linux/include

LINK_FLAGS=\
 -L . -lggml -lggml-base -lggml-cpu\
 libsentencepiece.so.0

testd: main.cpp ${HEADERS}
	g++ -ggdb -O0 -std=c++20 -static-libstdc++ -fvisibility-inlines-hidden -o $@ ${CPP_FLAGS} $< ${LINK_FLAGS}

test: main.cpp ${HEADERS}
	g++ -O3 -std=c++20 -static-libstdc++ -fvisibility-inlines-hidden -o $@ ${CPP_FLAGS} $< ${LINK_FLAGS}

debug: testd
	LD_LIBRARY_PATH=$(CURDIR) gdb ./testd

run: test
	LD_LIBRARY_PATH=$(CURDIR) ./test

clean:
	rm test testd

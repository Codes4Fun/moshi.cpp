
all: mimi-encode mimi-decode mimi-play mimi-echo moshi-tts moshi-stt

include config.mk

LIBMOSHI_HEADERS=\
 ../src/config.h\
 ../src/context.h\
 ../src/ggml_cap.h\
 ../src/ggml_wrap.h\
 ../src/loader.h\
 ../src/moshi/models/compression.h\
 ../src/moshi/models/lm.h\
 ../src/moshi/models/lm_default.h\
 ../src/moshi/models/lm_utils.h\
 ../src/moshi/models/tts.h\
 ../src/moshi/modules/conv.h\
 ../src/moshi/modules/gating.h\
 ../src/moshi/modules/rope.h\
 ../src/moshi/modules/seanet.h\
 ../src/moshi/modules/transformer.h\
 ../src/moshi/quantization/core_vq.h\
 ../src/moshi/quantization/vq.h\
 ../src/moshi/utils/sampling.h\
 ../src/torch.h\
 ../src/wav.h\

MOSHI_HEADERS=\
 ../include/moshi/json.h\
 ../include/moshi/moshi.h\
 ../include/moshi/ptrs.h\
 ../include/moshi/safetensor.h\

HEADERS=${MOSHI_HEADERS}\
 ../tools/ffmpeg_helpers.h\
 ../tools/sdl_helper.h\
 ../tools/util.h

MOSHI_CFLAGS=-I../include
MOSHI_LIBS=-L. -lmoshi

CFLAGS=-std=c++20 -static-libstdc++ -fvisibility-inlines-hidden\
 ${SP_CFLAGS} ${GGML_CFLAGS} ${MOSHI_CFLAGS}
LIBS=${SP_LIBS} ${GGML_LIBS}\
 -Wl,-rpath=.-Wl,-rpath=.

TOOL_CFLAGS=-std=c++20 -static-libstdc++\
 ${FFMPEG_CFLAGS} ${SP_CFLAGS} ${GGML_CFLAGS} ${MOSHI_CFLAGS}
TOOL_LIBS=${LIBS} ${MOSHI_LIBS} ${FFMPEG_LIBS} -lSDL2

DL+=${FFMPEG_DL} ${GGML_DL}

# MARK: Library

LIBMOSHI = ${DL_PRE}moshi.${DL_EXT}

${LIBMOSHI}: ../src/moshi.cpp ../src/json.cpp ../src/safetensor.cpp ${LIBMOSHI_HEADERS}
	g++ -ggdb -O0 -fPIC -shared -o $@ ${CFLAGS} ../src/moshi.cpp ../src/json.cpp ../src/safetensor.cpp ${LIBS}

# MARK: Tools

moshi-dl.json: ../tools/moshi-dl.json
	cp $< $@

moshi-dl: ../tools/moshi-dl.cpp ${LIBMOSHI} ${HEADERS} ../tools/validate.h ${DL} moshi-dl.json
	g++ -ggdb -O0 -o $@ ${TOOL_CFLAGS} $< ${TOOL_LIBS} -lcurl -lcrypto

mimi-%: ../tools/mimi-%.cpp ${LIBMOSHI} ${HEADERS} ${DL}
	g++ -ggdb -O0 -o $@ ${TOOL_CFLAGS} $< ${TOOL_LIBS}

moshi-%: ../tools/moshi-%.cpp ${LIBMOSHI} ${HEADERS} ${DL}
	g++ -ggdb -O0 -o $@ ${TOOL_CFLAGS} $< ${TOOL_LIBS}

# MARK: Commands

test-mimi-encode: mimi-encode
	./mimi-encode test.wav test.mimi

debug-mimi-encode: mimi-encode
	gdb --args ./mimi-encode test.wav test.mimi

test-mimi-decode: mimi-decode
	./mimi-decode test.mimi test.mp3

debug-mimi-decode: mimi-decode
	gdb --args ./mimi-decode test.mimi test.mp3

test-mimi-play: mimi-play
	./mimi-play test.mimi

debug-mimi-play: mimi-play
	gdb --args ./mimi-play test.mimi

test-mimi-echo: mimi-echo
	./mimi-echo

debug-mimi-echo: mimi-echo
	gdb ./mimi-echo

test-moshi-tts: moshi-tts
	./moshi-tts "the quick brown fox jumped over the lazy dog"

#debug-moshi-tts: moshi-tts
#	gdb --args ./moshi-tts "the quick brown fox jumped over the lazy dog"
debug-moshi-tts: moshi-tts
	gdb --args ./moshi-tts "done loading. done loading. done loading. done loading."

test-moshi-stt: moshi-stt
	./moshi-stt -i ../audio_test.wav

debug-moshi-stt: moshi-stt
	gdb --args ./moshi-stt -i ../audio_test.wav

clean:
	rm -f ${DL_PRE}*.${DL_EXT}* mimi-* moshi-* test.*

.PRECIOUS: ${DL}

.PHONY:\
 clean\
 all\
 test-mimi-encode\
 debug-mimi-encode\
 test-mimi-decode\
 debug-mimi-decode\
 test-mimi-play\
 debug-mimi-play\
 test-mimi-echo\
 debug-mimi-echo\
 test-moshi-tts\
 debug-moshi-tts\
 test-moshi-stt\
 debug-moshi-stt


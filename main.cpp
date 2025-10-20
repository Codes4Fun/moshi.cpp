#include <stdio.h>
#include <assert.h>
#include <math.h>

#include <sstream>
#include <vector>
#include <string>
#include <deque>
#include <iomanip>
#include <iostream>
#include <functional>

#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>

//#include "replay.h"
//#include "replay_ops.h"

#define CAPTURE(...)
#define CAPTURE_GROUP(...)
#define ONCE(code) {static bool once=false; if (!once) {{code;}; once=true;}}
#define ON_NTH(nth, code) {static int count=0; if (count++ == (nth)) {code;}}

//#define DISABLE_RAND
#include "src/safetensor.h"
#include "src/config.h"
#include "src/context.h"
#include "src/loader.h"
#include "src/torch.h"
#include "src/moshi/modules/transformer.h"
#include "src/moshi/utils/sampling.h"
#include "src/moshi/models/lm_utils.h"
#include "src/moshi/models/lm.h"
#include "src/moshi/quantization/core_vq.h"
#include "src/moshi/quantization/vq.h"
#include "src/moshi/modules/conv.h"
#include "src/moshi/modules/seanet.h"
#include "src/moshi/models/compression.h"
#include "src/moshi/models/lm_default.h"
#include "src/moshi/models/tts.h"


void top_k_cuda_test() {
	ggml_backend_t backend = NULL;

    //backend = ggml_backend_init_by_name("Vulkan0", NULL);
    backend = ggml_backend_init_by_name("CUDA0", NULL);
	assert( backend );

    // create context
    auto ctx = ggml_init({
		/*.mem_size   =*/ ggml_tensor_overhead()*GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead(),
		/*.mem_buffer =*/ NULL,
		/*.no_alloc   =*/ true,
    });

#define TOP_K_PROBS 8000
#define TOP_K 512

	auto probs = ggml_arange( ctx, 0, TOP_K_PROBS, 1 );
	auto result = ggml_top_k( ctx, probs, TOP_K );

    auto gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, result);

    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    ggml_backend_graph_compute(backend, gf);

    std::vector<int> out_data(ggml_nelements(result));
    ggml_backend_tensor_get(result, out_data.data(), 0, ggml_nbytes(result));

	printf("out_data %d %d %d %d ... %d %d %d %d\n",
        out_data[0], out_data[1], out_data[2], out_data[3],
        out_data[TOP_K-4], out_data[TOP_K-3], out_data[TOP_K-2], out_data[TOP_K-1] );

	printf("press enter to continue");
	getchar();

    // release backend memory and free backend
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
}

void test_rms_norm() {
	// init backend
    auto backend = ggml_backend_init_by_name("CUDA0", NULL);
	assert( backend );
    auto ctx = ggml_init({
		/*.mem_size   =*/ ggml_tensor_overhead()*GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead(),
		/*.mem_buffer =*/ NULL,
		/*.no_alloc   =*/ true,
    });
	// create tensors
	std::vector<float> data = { 1.f, 2.f };
	auto input = ggml_new_tensor_1d( ctx, GGML_TYPE_F32, data.size() );
	auto result = ggml_mul( ctx, input, ggml_rms_norm( ctx, input, 1e-8 ) );
	// graph
    auto gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, result);
	// allocate tensors
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
	// initialize tensors
	ggml_backend_tensor_set(input, data.data(), 0, ggml_nbytes(input));
	// compute
    ggml_backend_graph_compute(backend, gf);
	// output results, should be 0.632456 2.529822
	ggml_backend_tensor_get(result, data.data(), 0, ggml_nbytes(result));
	printf("%f %f\npress enter to continue", data[0], data[1]);
	getchar();
	// cleanup
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
}

void test_rms_norm2() {
	// init backend
    auto backend = ggml_backend_init_by_name("CUDA0", NULL);
	assert( backend );
    auto ctx = ggml_init({
		/*.mem_size   =*/ ggml_tensor_overhead()*GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead(),
		/*.mem_buffer =*/ NULL,
		/*.no_alloc   =*/ true,
    });
	// tensors
	auto input = ggml_arange( ctx, 1, 3, 1 );
	auto result = ggml_mul( ctx, input, ggml_rms_norm( ctx, input, 1e-5 ) );
	// graph compute
    auto gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, result);
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    ggml_backend_graph_compute(backend, gf);
	// cleanup
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
}

void tts_test() {
    //auto backend = ggml_backend_init_by_name("Vulkan0", NULL);
    //auto backend = ggml_backend_init_by_name("CUDA0", NULL);
    //auto backend = ggml_backend_init_by_name("CPU", NULL);
    auto backend = ggml_backend_init_best();
    assert( backend );
    auto dev = ggml_backend_get_device( backend );
    auto dev_name = ggml_backend_dev_name( dev );
    printf( "using backend: \"%s\"\n", dev_name );

    auto last = ggml_time_ms();
    printf( "loading...\n");
    auto tts = moshi_ttsmodel( backend, "kyutai/tts-1.6b-en_fr" );
    //auto tts = moshi_ttsmodel( backend, "kyutai/tts-0.75b-en-public" );
    assert( tts );

    if ( tts->needs_voice ) {
        load_voice( "kyutai/tts-voices/expresso/ex03-ex01_happy_001_channel1_334s.wav.1e68beda@240.safetensors",
            tts, backend );
        //load_voice( "kyutai/tts-voices/expresso/ex04-ex02_awe_001_channel1_982s.wav.1e68beda@240.safetensors",
        //    tts, backend );
    }

    auto cur = ggml_time_ms();
    printf( "loading done %ld\n", (cur - last));
    last = cur;
    printf( "generating...\n");

    std::string text = "Hey, how are you?";
    text += " Let's try to blow out our context!";
    text += " She sells sea shells by the sea shore.";
    text += " The quick brown fox jumped over the brown dog.";
    text += " The planes in spain, stay mainly on the plains.";
    text += " And hopefully this exceeds our context!";

    moshi_ttsmodel_generate_wav( tts, text, "audio_test.wav", backend );

    cur = ggml_time_ms();
    printf( "generating done %ld\n", (cur - last));
}

int main(int argc, char **argv)
{
    ggml_time_init();

    auto dev_count = ggml_backend_dev_count();
    printf("available backends:\n");
    for (size_t i = 0; i < dev_count; i++) {
        auto dev = ggml_backend_dev_get( i );
        auto name = ggml_backend_dev_name( dev );
        printf( "  \"%s\"\n", name );
    }

	//top_k_cuda_test();
	//test_rms_norm();

    tts_test();

#if 0
	replay_test();
#endif

	printf("press enter to continue");
	getchar();
	return 0;
}

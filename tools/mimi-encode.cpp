#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <iostream> // tts

#include "../examples/ffmpeg_helpers.h"

// for src/context.h
#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#ifdef ENABLE_REPLAY
#include "replay.h"
#define CAPTURE(...)
#define CAPTURE_GROUP(...)
#else
#ifdef ENABLE_CAPTURE
#include "src/ggml_cap.h"
#else
#define CAPTURE(...)
#define CAPTURE_GROUP(...)
#endif
#endif
#define ONCE(code) {static bool once=false; if (!once) {{code;}; once=true;}}
#define ON_NTH(nth, code) {static int count=0; if (count++ == (nth)) {code;}}

#include "../src/ptrs.h"
#include "../src/safetensor.h"
#include "../src/config.h"
#include "../src/context.h"
#include "../src/wav.h"
#include "../src/loader.h"
#include "../src/torch.h"
#include "../src/moshi/modules/transformer.h"
#include "../src/moshi/utils/sampling.h"
#include "../src/moshi/models/lm_utils.h"
#include "../src/moshi/models/lm.h"
#include "../src/moshi/quantization/core_vq.h"
#include "../src/moshi/quantization/vq.h"
#include "../src/moshi/modules/conv.h"
#include "../src/moshi/modules/seanet.h"
#include "../src/moshi/models/compression.h"
#include "../src/moshi/models/lm_default.h"
#include "../src/moshi/models/tts.h"

const int frame_size = 1920;

static void print_usage(const char * program) {
    fprintf( stderr, "usage: %s [option(s)] input-file output.mimi", program );
    fprintf( stderr, "input-file can also be wav, ogg, flac, and many more formats.\n" );
    fprintf( stderr, "\noption(s):\n" );
    fprintf( stderr, "  -h,       --help          show this help message\n" );
    fprintf( stderr, "  -q N,     --n_q N         compression level. max 32. 32 by default.\n" );
    fprintf( stderr, "  -l,       --list-devices  list devices and exit.\n" );
    fprintf( stderr, "  -d NAME,  --device NAME   use named device.\n" );
    exit(1);
}

static void list_devices() {
    auto dev_count = ggml_backend_dev_count();
    fprintf( stderr, "available devices:\n" );
    for (size_t i = 0; i < dev_count; i++) {
        auto dev = ggml_backend_dev_get( i );
        auto name = ggml_backend_dev_name( dev );
        fprintf( stderr, "  \"%s\"\n", name );
    }
    exit(1);
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        print_usage(argv[0]);
    }

    int n_q = 32;
    const char * device = NULL;
    const char * input_filename = NULL;
    const char * output_filename = NULL;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
        }
        if (arg == "-l" || arg == "--list-devices") {
            list_devices();
        }
        if (arg == "-q" || arg == "--n_q") {
            if (i + 1 >= argc) {
                fprintf( stderr, "error: \"%s\" requires value\n", argv[i] );
                exit(1);
            }
            n_q = std::stoi(argv[++i]);
            if (n_q > 32) {
                fprintf( stderr, "error: value for \"%s\" cannot be more than 32\n", argv[i] );
                exit(1);
            }
            continue;
        }
        if (arg == "-d" || arg == "--device") {
            if (i + 1 >= argc) {
                fprintf( stderr, "error: \"%s\" requires name of device\n", argv[i] );
                exit(1);
            }
            device = argv[++i];
            continue;
        }
        if (arg[0] == '-') {
            fprintf( stderr, "error: unrecognized option \"%s\"\n", argv[i] );
            exit(1);
        }
        if (!input_filename) {
            input_filename = argv[i];
        } else if (!output_filename) {
            output_filename = argv[i];
        } else {
            fprintf( stderr, "error: unexpected extra argument \"%s\"\n", argv[i] );
            exit(1);
        }
    }
    if (!input_filename || !output_filename) {
        print_usage(argv[0]);
    }

    Decoder decoder;
    decoder.init( input_filename );

    ggml_backend * backend;
    if ( device ) {
        backend = ggml_backend_init_by_name( device, NULL );
    } else {
        backend = ggml_backend_init_best();
    }
    if ( ! backend ) {
        fprintf( stderr, "failed to initialize device\n" );
        exit(1);
    }
    auto dev = ggml_backend_get_device( backend );
    auto dev_name = ggml_backend_dev_name( dev );
    printf( "using device: \"%s\"\n", dev_name );

    own_ptr<ScratchContext> scratch_cpu;
    own_ptr<ScratchContext> scratch;
    own_ptr<moshi_mimi_t> mimi;
    own_ptr<WeightLoader> mimi_weights;

    scratch_cpu = new ScratchContext( 256 );
    scratch = new ScratchContext( 256, backend );
    mimi = moshi_mimi_alloc_default( n_q );
    mimi_weights = WeightLoader::from_safetensor(
        "../kyutai/tts-0.75b-en-public/tokenizer-e351c8d8-checkpoint125.safetensors",
        scratch_cpu, backend );
    if ( ! mimi_weights ) {
        fprintf(stderr, "error: mimi weights not found\n" );
        exit(1);
    }
    get_weights( mimi_weights, "mimi.quantizer.", mimi->quantizer );
    get_weights( mimi_weights, "mimi.upsample.convtr.", mimi->upsample );
    get_weights( mimi_weights, "mimi.decoder_transformer.transformer.", mimi->decoder_transformer );
    get_weights( mimi_weights, "mimi.decoder.", mimi->decoder );
    if ( mimi->encoder ) {
        get_weights( mimi_weights, "mimi.downsample.conv.", mimi->downsample );
        get_weights( mimi_weights, "mimi.encoder_transformer.transformer.", mimi->encoder_transformer );
        get_weights( mimi_weights, "mimi.encoder.", mimi->encoder );
    }
    mimi_weights->load();

    StateContext state_ctx( backend );
    own_ptr<moshi_mimi_state_t> mimi_states = moshi_mimi_encoder_states( &state_ctx, mimi );
    ggml_tensor * device_frame = NULL;
    state_ctx.new_tensor( GGML_NE(frame_size), GGML_TYPE_F32, &device_frame );

    state_ctx.alloc();
    state_ctx.init();
    init( mimi_states );

    ScratchContext &ctx = *scratch;

    auto f = fopen( output_filename, "wb" );
    if ( ! f ) {
        fprintf( stderr, "error: failed to open \"%s\"\n", output_filename );
    }
    assert( fwrite( "MIMI", 4, 1, f ) == 1 );
    assert( fwrite( &n_q, 4, 1, f ) == 1 );

    AVChannelLayout mono;
    av_channel_layout_default( &mono, 1 );

    Resampler resampler;
    resampler.set_input( decoder.codec_ctx );
    resampler.set_output( 24000, AV_SAMPLE_FMT_FLT, mono, frame_size );
    resampler.init();
    std::vector<int> tokens;
    std::vector<int16_t> tokens16;

    int frame_count = 0;
    AVFrame * dec_frame;
    while ( ( dec_frame = decoder.frame() ) ) {
        auto frame = resampler.frame( dec_frame );
        while ( frame ) {

            ggml_backend_tensor_set( device_frame, frame->data[0], 0, ggml_nbytes( device_frame ) );
            auto codes = mimi_encode( ctx, mimi, mimi_states, device_frame );
            auto cast = ggml_cast( ctx, codes, GGML_TYPE_I32 );
            tokens.resize( ggml_nelements( cast ) );
            ctx.build_forward_expand( cast, tokens.data() );
            ctx.compute();

            assert( (int)tokens.size() == n_q );
            tokens16.resize( n_q );
            for (int i = 0; i < n_q; i++) {
                assert( tokens[i] < 32768 && tokens[i] >= -32768 );
                tokens16[i] = tokens[i];
            }
            assert( fwrite( tokens16.data(), n_q*2, 1, f ) == 1 );

            frame_count++;
            frame = resampler.frame();
        }
    }
    auto frame = resampler.flush( true ); // inject silence
    if ( frame ) {
        ggml_backend_tensor_set( device_frame, frame->data[0], 0, ggml_nbytes( device_frame ) );
        auto codes = mimi_encode( ctx, mimi, mimi_states, device_frame );
        auto cast = ggml_cast( ctx, codes, GGML_TYPE_I32 );
        tokens.resize( ggml_nelements( cast ) );
        ctx.build_forward_expand( cast, tokens.data() );
        ctx.compute();

        assert( (int)tokens.size() == n_q );
        for (int i = 0; i < n_q; i++) {
            assert( tokens[i] < 32768 && tokens[i] >= -32768 );
            tokens16[i] = tokens[i];
        }
        assert( fwrite( tokens16.data(), n_q*2, 1, f ) == 1 );
    }
    fclose( f );
    printf( "%d\n", frame_count );
    return 0;
}

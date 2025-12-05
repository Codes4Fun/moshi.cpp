#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <iostream> // tts

#include "ffmpeg_helpers.h"
#include "moshi.h"

static void print_usage(const char * program) {
    fprintf( stderr, "usage: %s [option(s)] input.mimi output-file", program );
    fprintf( stderr, "output-file can also be wav, ogg, flac, and many more formats.\n" );
    fprintf( stderr, "\noption(s):\n" );
    fprintf( stderr, "  -h,       --help          show this help message\n" );
    fprintf( stderr, "  -m FNAME, --model FNAME   mimi model.\n" );
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

    const char * device = NULL;
    const char * input_filename = NULL;
    const char * output_filename = NULL;
    const char * model_filename = "tokenizer-e351c8d8-checkpoint125.safetensors";

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
        }
        if (arg == "-m" || arg == "--model") {
            if (i + 1 >= argc) {
                fprintf( stderr, "error: \"%s\" requires filepath to model\n", argv[i] );
                exit(1);
            }
            model_filename = argv[++i];
            continue;
        }
        if (arg == "-l" || arg == "--list-devices") {
            list_devices();
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

    auto f = fopen( input_filename, "rb" );
    if ( ! f ) {
        fprintf( stderr, "error: unable to open \"%s\"\n", input_filename );
        exit( 1 );
    }
    uint32_t marker;
    assert( fread( &marker, 4, 1, f ) == 1 );
    if ( marker != *(uint32_t*)"MIMI" ) {
        fprintf( stderr, "error: invalid mimi input file \"%s\".\n", input_filename );
        exit( 1 );
    }
    int n_q;
    assert( fread( &n_q, 4, 1, f ) == 1 );
    if ( n_q < 1 || n_q > 32 ) {
        fprintf( stderr, "error: n_q in mimi file out of range %d\n", n_q );
        exit( 1 );
    }

    // decoder
    moshi_context_t moshi;
    moshi_alloc( &moshi, device );
    mimi_codec_t codec;
    mimi_alloc( &codec, &moshi, model_filename, n_q );
    mimi_decode_context_t decoder;
    mimi_decode_alloc_context( &decoder, &codec );
    int frame_size = mimi_frame_size( &codec );

    AVChannelLayout mono;
    av_channel_layout_default( &mono, 1 );

    unref_ptr<AVFrame> mimi_frame = av_frame_alloc();
    mimi_frame->nb_samples     = frame_size;
    mimi_frame->ch_layout      = mono;
    mimi_frame->format         = AV_SAMPLE_FMT_FLT;
    mimi_frame->sample_rate    = 24000;
    check_error( av_frame_get_buffer( mimi_frame, 0 ),
        "Error making frame buffer" );

    Encoder encoder;
    encoder.init_from( output_filename, 24000, AV_SAMPLE_FMT_FLT, mono );

    Resampler resampler;
    resampler.set_input( 24000, AV_SAMPLE_FMT_FLT, mono );
    resampler.set_output( encoder.codec_ctx );
    resampler.init();

    // main loop
    std::vector<int16_t> tokens(n_q);
    while ( fread(tokens.data(), n_q*2, 1, f ) == 1 ) {
        mimi_decode_send( &decoder, tokens.data() );
        mimi_decode_receive( &decoder, (float*)mimi_frame->data[0] );

        auto frame = resampler.frame( mimi_frame );
        while ( frame ) {
            encoder.frame( frame );
            frame = resampler.frame();
        }
    }
    encoder.flush();
    fclose( f );

    return 0;
}

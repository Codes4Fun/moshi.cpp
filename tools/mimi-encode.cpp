#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <iostream> // tts

#include "ffmpeg_helpers.h"
#include "moshi.h"

static void print_usage(const char * program) {
    fprintf( stderr, "usage: %s [option(s)] input-file output.mimi", program );
    fprintf( stderr, "input-file can also be wav, ogg, flac, and many more formats.\n" );
    fprintf( stderr, "\noption(s):\n" );
    fprintf( stderr, "  -h,       --help          show this help message\n" );
    fprintf( stderr, "  -m FNAME, --model FNAME   mimi model.\n" );
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

    // encoder
    moshi_context_t moshi;
    moshi_alloc( &moshi, device );
    mimi_codec_t codec;
    mimi_alloc( &codec, &moshi, model_filename, n_q );
    mimi_encode_context_t encoder;
    mimi_encode_alloc_context( &encoder, &codec );
    int frame_size = mimi_frame_size( &codec );

    // output file
    auto f = fopen( output_filename, "wb" );
    if ( ! f ) {
        fprintf( stderr, "error: failed to open \"%s\"\n", output_filename );
    }
    assert( fwrite( "MIMI", 4, 1, f ) == 1 );
    assert( fwrite( &n_q, 4, 1, f ) == 1 );

    // resampler
    AVChannelLayout mono;
    av_channel_layout_default( &mono, 1 );
    Resampler resampler;
    resampler.set_input( decoder.codec_ctx );
    resampler.set_output( 24000, AV_SAMPLE_FMT_FLT, mono, frame_size );
    resampler.init();

    std::vector<int16_t> tokens(n_q);

    // main loop
    int frame_count = 0;
    AVFrame * dec_frame;
    while ( ( dec_frame = decoder.frame() ) ) {
        auto frame = resampler.frame( dec_frame );
        while ( frame ) {
            mimi_encode_send( &encoder, (float*)frame->data[0] );
            mimi_encode_receive( &encoder, tokens.data() );
            assert( fwrite( tokens.data(), n_q*2, 1, f ) == 1 );
            frame_count++;
            frame = resampler.frame();
        }
    }
    auto frame = resampler.flush( true ); // inject silence
    if ( frame ) {
        mimi_encode_send( &encoder, (float*)frame->data[0] );
        mimi_encode_receive( &encoder, tokens.data() );
        assert( fwrite( tokens.data(), n_q*2, 1, f ) == 1 );
        frame_count++;
    }
    fclose( f );
    printf( "%d\n", frame_count );
    return 0;
}

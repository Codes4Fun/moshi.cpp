#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <iostream> // tts

#include "common_ggml.h"
#include <moshi/moshi.h>
#include "common_av.h"
#include "common_utils.h"

static void print_usage(const char * program) {
    fprintf( stderr, R"(usage: %s [option(s)] input.mimi output-file

output-file can also be wav, ogg, flac, and many more formats.

option(s):
  -h,       --help          show this help message
  -m FNAME, --model FNAME   mimi model.
  -l,       --list-devices  list devices and exit.
  -d NAME,  --device NAME   use named device.
  -g,       --gguf-caching  loads gguf if exists, saves gguf if it does not.
                            model is saved alongside the original
                            safetensors file.
)", program );
    exit(1);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
    }

    const char * device = NULL;
    const char * input_filename = NULL;
    const char * output_filename = NULL;
    std::string mimi_filepath = "Codes4Fun/moshi-common/mimi-e351c8d8-125.gguf";
    bool gguf_caching = false;

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
            mimi_filepath = argv[++i];
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
        if (arg == "-g" || arg == "--gguf-caching" ) {
            gguf_caching = true;
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

    bool found = false;
    bool found_dir = false;
    check_arg_path( mimi_filepath, found, found_dir );

    if ( ! found ) {
        const char * model_cache = getenv("MODEL_CACHE");
        std::string model_root = model_cache? model_cache : "";

        std::string program_path = get_program_path(argv[0]);

        // the file is the same for all models
        std::vector<std::string> paths;
        if ( found_dir ) {
            ensure_path( mimi_filepath );
            paths.push_back( mimi_filepath + "Codes4Fun/moshi-common/mimi-e351c8d8-125.gguf" );
        }
        if ( model_root.size() ) {
            ensure_path( model_root );
            paths.push_back( model_root + "Codes4Fun/moshi-common/mimi-e351c8d8-125.gguf" );
        }
        if ( program_path.size() ) {
            ensure_path( program_path );
            paths.push_back( program_path + "Codes4Fun/moshi-common/mimi-e351c8d8-125.gguf" );
        }
        for ( auto & path : paths ) {
            if ( file_exists( path.c_str() ) ) {
                mimi_filepath = path;
                found = true;
                break;
            }
        }
        if ( ! found ) {
            fprintf( stderr, "error: missing mimi model \"%s\"\n", mimi_filepath.c_str() );
            exit(1);
        }
    }
    printf("using %s\n", mimi_filepath.c_str());

    std::string mimi_gguf = "";
    if ( gguf_caching && mimi_filepath.ends_with(".safetensors") ) {
        mimi_gguf = mimi_filepath + ".gguf";
        if ( file_exists( mimi_gguf.c_str() ) ) {
            mimi_filepath = mimi_gguf;
            mimi_gguf = "";
        }
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

    common_ggml_t ggml;
    init_ggml( ggml, device );

    // decoder
    unref_ptr<moshi_context_t> moshi =  moshi_alloc( ggml.backend, ggml.backend_cpu );
    printf("loading %s\n", mimi_filepath.c_str());
    unref_ptr<mimi_codec_t> codec = mimi_alloc( moshi, mimi_filepath.c_str(), n_q );
    printf("done loading\n");
    if ( mimi_gguf.size() ) {
        mimi_save_gguf( codec, mimi_gguf.c_str() );
    }
    unref_ptr<mimi_decode_context_t> decoder = mimi_decode_alloc_context( codec );
    int frame_size = mimi_frame_size( codec );

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
    resampler.set_input( 24000, AV_SAMPLE_FMT_FLT, mono, frame_size );
    resampler.set_output( encoder.codec_ctx );
    resampler.init();

    // main loop
    std::vector<int16_t> tokens(n_q);
    while ( fread(tokens.data(), n_q*2, 1, f ) == 1 ) {
        mimi_decode_send( decoder, tokens.data() );
        mimi_decode_receive( decoder, (float*)mimi_frame->data[0] );

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

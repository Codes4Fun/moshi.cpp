#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <iostream> // tts

#include <moshi/moshi.h>
#include "ffmpeg_helpers.h"
#include "util.h"

static void print_usage(const char * program) {
    fprintf( stderr, "usage: %s [option(s)] input-file output.mimi\n", program );
    fprintf( stderr, "\ninput-file can be wav, ogg, flac, mp4, and many more formats.\n" );
    fprintf( stderr, "\noption(s):\n" );
    fprintf( stderr, "  -h,       --help          show this help message\n" );
    fprintf( stderr, "  -m FNAME, --model FNAME   mimi model.\n" );
    fprintf( stderr, "  -q N,     --n_q N         compression level. max 32. 32 by default.\n" );
    fprintf( stderr, "  -l,       --list-devices  list devices and exit.\n" );
    fprintf( stderr, "  -d NAME,  --device NAME   use named device.\n" );
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
    std::string mimi_filepath = "kyutai/tts-1.6b-en_fr/tokenizer-e351c8d8-checkpoint125.safetensors";

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

    bool found = false;
    bool found_dir = false;
    check_arg_path( mimi_filepath, found, found_dir );

    if ( ! found ) {
        const char * model_cache = getenv("MODEL_CACHE");
        std::string model_root = model_cache? model_cache : "";

        std::string program_path = get_program_path(argv[0]);

        // the file is the same for all models
        std::vector<std::string> paths = {
            "kyutai/tts-1.6b-en_fr/tokenizer-e351c8d8-checkpoint125.safetensors",
            "kyutai/tts-0.75b-en-public/tokenizer-e351c8d8-checkpoint125.safetensors",
            "kyutai/stt-1b-en_fr-candle/mimi-pytorch-e351c8d8@125.safetensors",
            "kyutai/stt-2.6b-en/mimi-pytorch-e351c8d8@125.safetensors",
            "kyutai/stt-1b-en_fr/mimi-pytorch-e351c8d8@125.safetensors",
        };
        if ( found_dir ) {
            ensure_path( mimi_filepath );
            paths.push_back( mimi_filepath + "kyutai/tts-1.6b-en_fr/tokenizer-e351c8d8-checkpoint125.safetensors" );
            paths.push_back( mimi_filepath + "kyutai/tts-0.75b-en-public/tokenizer-e351c8d8-checkpoint125.safetensors" );
            paths.push_back( mimi_filepath + "kyutai/stt-1b-en_fr-candle/mimi-pytorch-e351c8d8@125.safetensors" );
            paths.push_back( mimi_filepath + "kyutai/stt-2.6b-en/mimi-pytorch-e351c8d8@125.safetensors" );
            paths.push_back( mimi_filepath + "kyutai/stt-1b-en_fr/mimi-pytorch-e351c8d8@125.safetensors" );
        }
        if ( model_root.size() ) {
            ensure_path( model_root );
            paths.push_back( model_root + "kyutai/tts-1.6b-en_fr/tokenizer-e351c8d8-checkpoint125.safetensors" );
            paths.push_back( model_root + "kyutai/tts-0.75b-en-public/tokenizer-e351c8d8-checkpoint125.safetensors" );
            paths.push_back( model_root + "kyutai/stt-1b-en_fr-candle/mimi-pytorch-e351c8d8@125.safetensors" );
            paths.push_back( model_root + "kyutai/stt-2.6b-en/mimi-pytorch-e351c8d8@125.safetensors" );
            paths.push_back( model_root + "kyutai/stt-1b-en_fr/mimi-pytorch-e351c8d8@125.safetensors" );
        }
        if ( program_path.size() ) {
            ensure_path( program_path );
            paths.push_back( program_path + "kyutai/tts-1.6b-en_fr/tokenizer-e351c8d8-checkpoint125.safetensors" );
            paths.push_back( program_path + "kyutai/tts-0.75b-en-public/tokenizer-e351c8d8-checkpoint125.safetensors" );
            paths.push_back( program_path + "kyutai/stt-1b-en_fr-candle/mimi-pytorch-e351c8d8@125.safetensors" );
            paths.push_back( program_path + "kyutai/stt-2.6b-en/mimi-pytorch-e351c8d8@125.safetensors" );
            paths.push_back( program_path + "kyutai/stt-1b-en_fr/mimi-pytorch-e351c8d8@125.safetensors" );
        }
        for ( auto & path : paths ) {
            if ( file_exists( path.c_str() ) ) {
                mimi_filepath = path;
                found = true;
                printf("using %s\n", mimi_filepath.c_str());
                break;
            }
        }
        if ( ! found ) {
            fprintf( stderr, "error: missing mimi file \"%s\"\n", mimi_filepath.c_str() );
            exit(1);
        }
    }

    Decoder decoder;
    decoder.init( input_filename );

    // encoder
    unref_ptr<moshi_context_t> moshi =  moshi_alloc( device );
    printf("loading %s\n", mimi_filepath.c_str());
    unref_ptr<mimi_codec_t> codec = mimi_alloc( moshi, mimi_filepath.c_str(), n_q );
    printf("done loading\n");
    unref_ptr<mimi_encode_context_t> encoder = mimi_encode_alloc_context( codec );
    int frame_size = mimi_frame_size( codec );

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
            mimi_encode_send( encoder, (float*)frame->data[0] );
            mimi_encode_receive( encoder, tokens.data() );
            assert( fwrite( tokens.data(), n_q*2, 1, f ) == 1 );
            frame_count++;
            frame = resampler.frame();
        }
    }
    auto frame = resampler.flush( true ); // inject silence
    if ( frame ) {
        mimi_encode_send( encoder, (float*)frame->data[0] );
        mimi_encode_receive( encoder, tokens.data() );
        assert( fwrite( tokens.data(), n_q*2, 1, f ) == 1 );
        frame_count++;
    }
    fclose( f );
    printf( "%d\n", frame_count );
    return 0;
}

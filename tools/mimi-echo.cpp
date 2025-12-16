#include <stdio.h>
#include <assert.h>
#include <iostream> // tts

#include <moshi/moshi.h>
#include "sdl_helper.h"
#include "util.h"

static void print_usage(const char * program) {
    fprintf( stderr, "usage: %s [option(s)]\n", program );
    fprintf( stderr, "\noption(s):\n" );
    fprintf( stderr, "  -h,       --help          show this help message\n" );
    fprintf( stderr, "  -m FNAME, --model FNAME   mimi model.\n" );
    fprintf( stderr, "  -q N,     --n_q N         compression level. max 32. 32 by default.\n" );
    fprintf( stderr, "  -l,       --list-devices  list devices and exit.\n" );
    fprintf( stderr, "  -d NAME,  --device NAME   use named device.\n" );
    exit(1);
}

#include <signal.h>
void signal_handler(int dummy) {
    printf("exit\n");
    exit(1);
}

int main(int argc, char *argv[]) {
    SDL_AudioSpec want, have;

    signal(SIGINT, signal_handler);

    int n_q = 32;
    const char * device = NULL;
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
        fprintf( stderr, "error: unexpected extra argument \"%s\"\n", argv[i] );
        exit(1);
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

    if (SDL_Init(SDL_INIT_AUDIO | SDL_INIT_TIMER) != 0) {
        fprintf(stderr, "Could not initialize SDL: %s\n", SDL_GetError());
        return 1;
    }

    unref_ptr<moshi_context_t> moshi =  moshi_alloc( device );
    printf("loading %s\n", mimi_filepath.c_str());
    unref_ptr<mimi_codec_t> codec = mimi_alloc( moshi, mimi_filepath.c_str(), n_q );
    printf("done loading\n");
    unref_ptr<mimi_encode_context_t> encoder = mimi_encode_alloc_context( codec );
    unref_ptr<mimi_decode_context_t> decoder = mimi_decode_alloc_context( codec );
    int frame_size = mimi_frame_size( codec );

    AudioState input_state;
    sdl_init_frames( input_state, 3, frame_size*4 );

    want.freq = 24000; // Sample rate
#ifdef USE_FLOAT
    want.format = AUDIO_F32; // Audio format
#else
    want.format = AUDIO_S16SYS; // Audio format
#endif
    want.channels = 1; // Mono audio
    want.samples = frame_size;
    want.callback = sdl_capture_callback;
    want.userdata = &input_state;

    SDL_AudioDeviceID cap_dev = SDL_OpenAudioDevice(NULL, 1, &want, &have, 0);
    if (cap_dev <= 0) {
        fprintf(stderr, "Could not open audio: %s\n", SDL_GetError());
        return 1;
    }

    AudioState output_state;
    sdl_init_frames( output_state, 3, frame_size*4 );

    want.callback = sdl_audio_callback;
    want.userdata = &output_state;
    SDL_AudioDeviceID dev = SDL_OpenAudioDevice(NULL, 0, &want, &have, 0);
    if (dev <= 0) {
        fprintf(stderr, "Could not open audio: %s\n", SDL_GetError());
        return 1;
    }

    SDL_PauseAudioDevice(cap_dev, 0);
    SDL_PauseAudioDevice(dev, 0);

    std::vector<int16_t> tokens(n_q);
    sdl_frame_t * input_frame;
    while ((input_frame = sdl_receive_frame( input_state, true ))) {
        mimi_encode_send( encoder, (float*)input_frame->data );
        mimi_encode_receive( encoder, tokens.data() );
        sdl_free_frame( input_state, input_frame );

        sdl_frame_t * output_frame = sdl_get_frame( output_state );
        mimi_decode_send( decoder, tokens.data() );
        mimi_decode_receive( decoder, (float*)output_frame->data );
        sdl_send_frame( output_state, output_frame );
    }

    SDL_CloseAudio();
    SDL_Quit();

    printf("\n");

    return 0;
}

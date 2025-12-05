#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <iostream> // tts
#include <pthread.h>

#include "sdl_helper.h"
#include "moshi.h"

static void print_usage(const char * program) {
    fprintf( stderr, "usage: %s [option(s)] input.mimi\n", program );
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

#include <signal.h>
void signal_handler(int dummy) {
    printf("exit\n");
    exit(1);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
    }

    signal(SIGINT, signal_handler);

    const char * device = NULL;
    const char * input_filename = NULL;
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
        } else {
            fprintf( stderr, "error: unexpected extra argument \"%s\"\n", argv[i] );
            exit(1);
        }
    }
    if (!input_filename) {
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

    if (SDL_Init(SDL_INIT_AUDIO | SDL_INIT_TIMER) < 0) {
        fprintf(stderr, "Could not initialize SDL: %s\n", SDL_GetError());
        return 1;
    }

    int sample_rate = 24000;
    int format = AUDIO_F32;
    int nb_samples = 1920;
    int nb_bytes = nb_samples * 4;

    AudioState state;

    SDL_AudioSpec want, have;
    SDL_zero(want);
    want.freq = sample_rate;
    want.format = format;
    want.channels = 1;
    want.samples = nb_samples; // Buffer size
    want.callback = sdl_audio_callback;
    want.userdata = &state;

    state.device_id = SDL_OpenAudioDevice(NULL, 0, &want, &have, 0);
    if (state.device_id == 0) {
        fprintf(stderr, "Failed to open SDL audio device: %s\n", SDL_GetError());
        SDL_Quit();
        return 1;
    }

    // do we need a resampler?
    if (have.freq != sample_rate) {
        fprintf(stderr, "error: sample_rate %d\n", have.freq);
        return 1;
    }
    if (have.format != format) {
        fprintf(stderr, "error: format %d\n", have.format);
        return 1;
    }
    if (have.channels != 1) {
        fprintf(stderr, "error: channels %d\n", have.channels);
        return 1;
    }
    if (have.samples != nb_samples) {
        fprintf(stderr, "error: samples %d\n", have.samples);
        return 1;
    }

    sdl_init_frames( state, 3, nb_bytes );

    SDL_PauseAudioDevice(state.device_id, 0);

    // main loop
    std::vector<int16_t> tokens(n_q);
    while ( fread(tokens.data(), n_q*2, 1, f ) == 1 ) {
        sdl_frame_t * frame = sdl_get_frame( state );
        mimi_decode_send( &decoder, tokens.data() );
        mimi_decode_receive( &decoder, (float*)frame->data );
        sdl_send_frame( state, frame );
    }
    fclose( f );
    SDL_Delay(1);
    SDL_CloseAudioDevice(state.device_id);
    SDL_Quit();

    return 0;
}

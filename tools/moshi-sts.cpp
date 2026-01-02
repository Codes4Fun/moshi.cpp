#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include <moshi/moshi.h>
#include "ffmpeg_helpers.h"
#include "sdl_helper.h"
#include "util.h"

static void print_usage(const char * program) {
    fprintf( stderr, R"(usage: %s [option(s)]

uses sdl to listen and respond to audio i/o.

options:
  -h,       --help              show this help message
  -c PATH,  --model-cache PATH  path to where all kyutai models are stored and
                                replaces MODEL_CACHE environment variable.
  -m PATH,  --model PATH        path to where model is, can be relative to the
                                MODEL_CACHE environment variable, or program
                                directory, or working directory.
  -l,       --list-devices      list hardware and exits.
  -d NAME,  --device NAME       use named hardware.
  -q QUANT, --quantize QUANT    convert weights to: q8_0, q4_0, q4_k
  -g,       --gguf-caching      loads gguf if exists, saves gguf if it does not.
  -s N,     --seed N            seed value.
  -t N,     --temperature N     consistency vs creativity, default 0.8
            --threads N         number of CPU threads to use during generation.
)", program);
    exit(1);
}

int64_t lm_start = 0;
int64_t lm_delta_time = 0;
int64_t lm_frames = 0;

#include <signal.h>
void signal_handler(int dummy) {
    printf( "\nrun time: %.3f s\n", lm_delta_time / 1000.f );
    printf( "run frames: %d\n", (int)lm_frames );
    printf( "\nframe rate:  %f frames/s\n", lm_frames * 1000.f / lm_delta_time );
    exit(1);
}

int main(int argc, char *argv[]) {
    signal(SIGINT, signal_handler);

    const char * model_cache = getenv("MODEL_CACHE");
    std::string model_root = model_cache? model_cache : "";
    std::string model_path = "kyutai/moshika-pytorch-bf16";
    const char * device = NULL;
    const char * quant = NULL;
    bool gguf_caching = false;
    int n_threads = 4;
    int seed = (int)time(NULL);
    float depth_temperature = 0.8f;
    float text_temperature = 0.7f;

    //////////////////////
    // MARK: Parse Args
    //////////////////////

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
        }
        if (arg == "-c" || arg == "--model-cache") {
            if (i + 1 >= argc) {
                fprintf( stderr, "error: \"%s\" requires path to models\n", argv[i] );
                exit(1);
            }
            model_root = argv[++i];
            continue;
        }
        if (arg == "-m" || arg == "--model") {
            if (i + 1 >= argc) {
                fprintf( stderr, "error: \"%s\" requires filepath to model\n", argv[i] );
                exit(1);
            }
            model_path = argv[++i];
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
        if (arg == "-q" || arg == "--quantize") {
            if (i + 1 >= argc) {
                fprintf( stderr, "error: \"%s\" requires type\n", argv[i] );
                exit(1);
            }
            quant = argv[++i];
            continue;
        }
        if (arg == "-g" || arg == "--gguf-caching" ) {
            gguf_caching = true;
            continue;
        }
        if (arg == "-s" || arg == "--seed") {
            if (i + 1 >= argc) {
                fprintf( stderr, "error: \"%s\" requires value\n", argv[i] );
                exit(1);
            }
            seed = std::stoi(argv[++i]);
            continue;
        }
        if (arg == "-t" || arg == "--temperature") {
            if (i + 1 >= argc) {
                fprintf( stderr, "error: \"%s\" requires value\n", argv[i] );
                exit(1);
            }
            text_temperature = (float) std::stod(argv[++i]);
            depth_temperature = text_temperature;
            continue;
        }
        if (arg == "--threads") {
            if (i + 1 >= argc) {
                fprintf( stderr, "error: \"%s\" requires value\n", argv[i] );
                exit(1);
            }
            n_threads = std::stoi(argv[++i]);
            continue;
        }
        if (arg[0] == '-') {
            fprintf( stderr, "error: unrecognized option \"%s\"\n", argv[i] );
            exit(1);
        }
        fprintf( stderr, "error: unexpected extra argument \"%s\"\n", argv[i] );
        exit(1);
    }

    /////////////////////////
    // MARK: Initialize
    /////////////////////////

    std::string program_path = get_program_path(argv[0]);
    ensure_path( program_path );
    ensure_path( model_root );
    ensure_path( model_path );

    // default config
    moshi_config_t config;
    if ( moshi_get_config( &config, "moshi-config.json" ) != 0 ) {
        if ( moshi_get_config( &config, (program_path + "moshi-config.json").c_str() ) != 0 ) {
            fprintf( stderr, "error: reading config\n");
            exit(1);
        }
    }

    srand( seed );
    printf( "seed: %d\n", seed );

    // context
    unref_ptr<moshi_context_t> moshi =  moshi_alloc( device );
    if ( n_threads > 0 && n_threads < 512 ) {
        moshi_set_n_threads( moshi, n_threads );
        printf( "set threads to %d\n", n_threads );
    }

    auto load_start = ggml_time_ms();

    // check model-path, absolute path or relative to current working directory
    std::string model_filepath = model_path + "model.safetensors";
    std::string mimi_filepath = model_path + "tokenizer-e351c8d8-checkpoint125.safetensors";
    std::string tokenizer_filepath = model_path + "tokenizer_spm_32k_3.model";
    if ( ! file_exists( model_filepath.c_str() ) ) {
        // check model-cache directory
        model_filepath = model_root + model_path + "model.safetensors";
        if ( ! file_exists( model_filepath.c_str() ) ) {
            model_filepath = program_path + model_path + "model.safetensors";
            if ( ! file_exists( model_filepath.c_str() ) ) {
                fprintf( stderr, "unable to find model, set MODEL_CACHE to root of models. %s\n", model_path.c_str() );
                exit(1);
            } else {
                mimi_filepath = program_path + model_path + "tokenizer-e351c8d8-checkpoint125.safetensors";
                tokenizer_filepath = program_path + model_path + "tokenizer_spm_32k_3.model";
            }
        } else {
            mimi_filepath = model_root + model_path + "tokenizer-e351c8d8-checkpoint125.safetensors";
            tokenizer_filepath = model_root + model_path + "tokenizer_spm_32k_3.model";
        }
    }

    printf( "loading...\n" );

    if ( quant ) {
        uint32_t uquant = *(uint32_t*)quant;
        switch (uquant) {
        case 0x305f3471: // "q4_0"
            break;
        case 0x6b5f3471: // "q4_k"
            break;
        case 0x305f3871: // "q8_0"
            break;
        default:
            fprintf( stderr, "error: invalid quant %s\n", quant );
            exit(-1);
        }
    }

    std::string model_gguf = "";
    if ( gguf_caching ) {
        if ( quant ) {
            uint32_t uquant = *(uint32_t*)quant;
            switch (uquant) {
            case 0x305f3471: // "q4_0"
                break;
            case 0x6b5f3471: // "q4_k"
                break;
            case 0x305f3871: // "q8_0"
                break;
            default:
                fprintf( stderr, "error: invalid quant %s\n", quant );
                exit(-1);
            }
            model_gguf = model_filepath + "." + quant + ".gguf";
            if ( file_exists( model_gguf.c_str() ) ) {
                model_filepath = model_gguf;
                model_gguf = "";
                quant = NULL;
            }
        } else {
            model_gguf = model_filepath + ".gguf";
            if ( file_exists( model_gguf.c_str() ) ) {
                model_filepath = model_gguf;
                model_gguf = "";
            }
        }
    }

    // model
    unref_ptr<moshi_lm_t> lm = moshi_lm_from_files( moshi, &config,
        model_filepath.c_str() );
    if ( quant ) {
        if ( ! moshi_lm_quantize( lm, quant ) ) {
            fprintf( stderr, "error: unknown quant %s\n", quant );
            exit(-1);
        }
    }

    // generator
    unref_ptr<moshi_lm_gen_t> gen = moshi_lm_generator( lm );

    // tokenizer
    unref_ptr<tokenizer_t> tok = tokenizer_alloc(
        tokenizer_filepath.c_str(),
        config.cross_attention );

    // codec
    int num_codebooks = (int)( config.n_q - config.dep_q );
    if ( config.dep_q > config.n_q )
        num_codebooks = (int) config.dep_q;
    unref_ptr<mimi_codec_t> codec = mimi_alloc( moshi,
        mimi_filepath.c_str(),
        num_codebooks );
    float frame_rate = mimi_frame_rate( codec );
    int frame_size = mimi_frame_size( codec );

    //const int delay_steps = (int)( tts_config.tts_config.audio_delay * frame_rate );
    //assert( delay_steps == 16 );
     //we invasively put the on_audio_hook in lm, so we need to copy delay_steps
    //moshi_lm_set_delay_steps( lm, delay_steps );

    // model
    moshi_lm_load( lm );
    if ( model_gguf.size() ) {
        moshi_lm_save_gguf( lm, model_gguf.c_str() );
    }

    // encoder
    unref_ptr<mimi_encode_context_t> encoder;
    encoder = mimi_encode_alloc_context( codec );

    // decoder
    unref_ptr<mimi_decode_context_t> decoder;
    decoder = mimi_decode_alloc_context( codec );

    auto load_end = ggml_time_ms();
    printf("done loading. %f\n", (load_end - load_start) / 1000.f);

    /////////////////////////
    // MARK: SDL
    /////////////////////////

    if (SDL_Init(SDL_INIT_AUDIO | SDL_INIT_TIMER) != 0) {
        fprintf(stderr, "Could not initialize SDL: %s\n", SDL_GetError());
        return 1;
    }

    AudioState input_state;
    sdl_init_frames( input_state, 3, frame_size*4 );

    SDL_AudioSpec want, have;

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

    /////////////////////////
    // MARK: Loop
    /////////////////////////

    // model
    moshi_lm_start( moshi, gen, depth_temperature, text_temperature );

    // consumes about   18400 MiB   17.969 GiB  18 GiB
    //printf("made it.\n");
    //getchar();

    std::vector<int16_t> tokens(num_codebooks);
    int text_token;

    // warmup
    std::vector<float> blank(frame_size);
    memset(blank.data(), 0, blank.size() * sizeof(blank[0]));
    for ( int i = 0; i < 4; i++ ) {
        mimi_encode_send( encoder, blank.data() );
        mimi_encode_receive( encoder, tokens.data() );
        moshi_lm_send2( gen, tokens );
        if ( ! moshi_lm_receive( gen, text_token, tokens ) )
            continue;
        mimi_decode_send( decoder, tokens.data() );
        mimi_decode_receive( decoder, blank.data() );
        memset(blank.data(), 0, blank.size() * sizeof(blank[0]));
    }

    int64_t lm_start = ggml_time_ms();

    SDL_PauseAudioDevice(cap_dev, 0);
    SDL_PauseAudioDevice(dev, 0);
    while ( true ) {
        lm_frames++;

        // sdl_receive_frame will block, don't include in frame rate
        lm_delta_time += ggml_time_ms() - lm_start;
        sdl_frame_t * input_frame = sdl_receive_frame( input_state, true );
        lm_start = ggml_time_ms();

        mimi_encode_send( encoder, (float*)input_frame->data );
        //mimi_encode_send( encoder, blank.data() );
        sdl_free_frame( input_state, input_frame );

        mimi_encode_receive( encoder, tokens.data() );
        moshi_lm_send2( gen, tokens );

        if ( moshi_lm_receive( gen, text_token, tokens ) ) {
            // audio out
            mimi_decode_send( decoder, tokens.data() );

            // sdl_get_frame will block, don't include in frame rate
            lm_delta_time += ggml_time_ms() - lm_start;
            sdl_frame_t * frame = sdl_get_frame( output_state );
            lm_start = ggml_time_ms();

            mimi_decode_receive( decoder, (float*)frame->data );
            sdl_send_frame( output_state, frame ); // this can block

            // text out
            if ( text_token != 0 && text_token != 3 /*&& text_token > 0*/ ) {
                auto piece = tokenizer_id_to_piece( tok, text_token );
                std::string _text;
                for ( size_t ci = 0; ci < piece.size(); ci++ ) {
                    if ( piece.c_str()[ci] == -30 ) {
                        _text += ' ';
                        ci += 2;
                        continue;
                    }
                    _text += piece[ci];
                }
                fprintf( stdout, "%s", _text.c_str() );
                fflush( stdout );
            }
        }
    }

    lm_delta_time += ggml_time_ms() - lm_start;
    printf( "frame rate:  %f frames/s\n", lm_frames * 1000.f / lm_delta_time );

    return 0;
}



#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "common_ggml.h"
#include <moshi/moshi.h>
#include "ffmpeg_helpers.h"
#include "sdl_helper.h"
#include "util.h"

static void print_usage(const char * program) {
    fprintf( stderr, R"(usage: %s [option(s)]

uses sdl to listen and respond to audio i/o.

options:
  -h,       --help             shows this help message

  -l,       --list-devices     list hardware and exits.
  -d NAME,  --device NAME      use named hardware.
            --threads N        number of CPU threads.

  -r PATH,  --model-root PATH  path to where all kyutai models are stored and
                               replaces MODEL_CACHE environment variable. the
                               models at root are in subdirectories of
                               'organization/model'
  -m PATH,  --model PATH       path to where model is, can be relative to the
                               MODEL_CACHE environment variable, or program
                               directory, or working directory. by default is
                               'Codes4Fun/moshika-q4_k-GGUF'
  -q QUANT, --quantize QUANT   convert weights to: q8_0, q4_0, q4_k
  -g,       --gguf-caching     loads gguf if exists, saves gguf if it does not.
                               model is saved alongside the original
                               safetensors file.

  -c N,     --context N        default: 3000, lowering reduces vram usage but
                               reduces effective conversation time. higher does
                               not improve effective conversation time.
  -s N,     --seed N           seed value.
  -t N,     --temperature N    consistency vs creativity, default 0.8
  -b        --bench            benchmark mode that disables sdl io and ends
                               after a few seconds.
  -i FNAME                     talk to moshi from an audio file.
            --delay            delay the audio file in frames (12.5 fps)

personaplex options:
  -v NAME,  --voice NAME       either a filepath to a safetensor or one of:
                                    NATF0
                                    NATF1
                                    NATF2
                                    NATF3
                                    NATM0
                                    NATM1
                                    NATM2
                                    NATM3
                                    VARF0
                                    VARF1
                                    VARF2
                                    VARF3
                                    VARF4
                                    VARM0
                                    VARM1
                                    VARM2
                                    VARM3
                                    VARM4

)", program);
    exit(1);
}

bool shutdown = false;
int64_t lm_delta_time = 0;
int64_t lm_frames = 0;

void log_metrics() {
    printf( "\n\nrun frames: %d\n", (int)lm_frames );
    printf( "run time: %.3f s\n", lm_delta_time / 1000000.f );
    printf( "\nframe rate:  %f frames/s\n", lm_frames * 1000000.f / lm_delta_time );
}

#include <signal.h>
void signal_handler(int dummy) {
    shutdown = true;
}

int main(int argc, char *argv[]) {
    signal(SIGINT, signal_handler);

    const char * device = NULL;
    int n_threads = 0;

    const char * model_cache = getenv("MODEL_CACHE");
    std::string model_root = model_cache? model_cache : "";
    std::string model_path = "Codes4Fun/moshika-q4_k-GGUF/";
    bool model_path_set = false;
    bool personaplex = false;
    const char * quant = NULL;
    bool gguf_caching = false;

    int context = -1;
    int seed = (int)time(NULL);
    float depth_temperature = 0.8f;
    float text_temperature = 0.7f;
    bool bench = false;

    const char * input = NULL;
    int input_delay = 0;

    std::string personaplex_voice_filepath = "";

    //////////////////////
    // MARK: Parse Args
    //////////////////////

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
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
        if (arg == "--threads") {
            if (i + 1 >= argc) {
                fprintf( stderr, "error: \"%s\" requires value\n", argv[i] );
                exit(1);
            }
            n_threads = std::stoi(argv[++i]);
            continue;
        }
        if (arg == "-r" || arg == "--model-root") {
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
            model_path_set = true;
            personaplex = model_path.find("personaplex") != std::string::npos;
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
        if (arg == "-c" || arg == "--context") {
            if (i + 1 >= argc) {
                fprintf( stderr, "error: \"%s\" requires value\n", argv[i] );
                exit(1);
            }
            context = std::stoi(argv[++i]);
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
        if (arg == "-i") {
            if (i + 1 >= argc) {
                fprintf( stderr, "error: \"%s\" requires filepath to audio file\n", argv[i] );
                exit(1);
            }
            input = argv[++i];
            continue;
        }
        if (arg == "--delay") {
            if (i + 1 >= argc) {
                fprintf( stderr, "error: \"%s\" requires value\n", argv[i] );
                exit(1);
            }
            input_delay = std::stoi(argv[++i]);
            continue;
        }
        if (arg == "-b" || arg == "--bench")  {
            bench = true;
            continue;
        }
        if (arg == "-v" || arg == "--voice") {
            if (i + 1 >= argc) {
                fprintf( stderr, "error: \"%s\" must be followed by filepath or voice name\n", argv[i] );
                exit(1);
            }
            personaplex_voice_filepath = argv[++i];
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

    if ( input && ! file_exists( input ) ) {
        fprintf( stderr, "error: input file does not exist: %s\n", input );
        exit(1);
    }

    // find model path
    bool found_file, found_dir;
    if ( is_abs_or_rel( model_path ) ) {
        check_arg_path( model_path, found_file, found_dir );
        if ( ! found_dir ) {
            if ( found_file ) {
                fprintf( stderr, "error: expected directory but found file: %s\n",
                     model_path.c_str() );
                exit(1);
            } else {
                fprintf( stderr, "error: could not find directory: %s\n",
                    model_path.c_str() );
                exit(1);
            }
        }
    } else if ( ! model_path_set) {
        // check defaults
        std::vector<std::string> paths;
        paths.push_back( model_root + "Codes4Fun/moshika-q4_k-GGUF/" );
        paths.push_back( program_path + "Codes4Fun/moshika-q4_k-GGUF/" );
        paths.push_back( "Codes4Fun/moshika-q4_k-GGUF/" );
        paths.push_back( model_root + "kyutai/moshika-pytorch-bf16/" );
        paths.push_back( program_path + "kyutai/moshika-pytorch-bf16/" );
        paths.push_back( "kyutai/moshika-pytorch-bf16/" );
        for ( auto & path : paths ) {
            check_arg_path( path, found_file, found_dir );
            if ( found_dir ) {
                model_path = path;
                break;
            }
        }
        if ( ! found_dir ) {
            fprintf( stderr, "error: could not find a default model directory\n" );
            exit(1);
        }
    } else {
        std::string full_path = model_root + model_path;
        check_arg_path( full_path, found_file, found_dir );
        if ( found_dir ) {
            model_path = full_path;
        } else {
            full_path = program_path + model_path;
            check_arg_path( full_path, found_file, found_dir );
            if ( found_dir ) {
                model_path = full_path;
            } else {
                check_arg_path( model_path, found_file, found_dir );
                if ( ! found_dir ) {
                    fprintf( stderr, "error: could not find directory: %s\n",
                        model_path.c_str() );
                    exit(1);
                }
            }
        }
    }
    printf( "found model path: %s\n", model_path.c_str() );

    // default config
    moshi_config_t config;
    std::string config_filepath;
    if ( personaplex ) {
        config_filepath = model_path + "personaplex-config.json";
        if ( ! file_exists( config_filepath.c_str() ) ) {
            config_filepath = program_path + "personaplex-config.json";
            if ( ! file_exists( config_filepath.c_str() ) ) {
                fprintf( stderr, "error: failed to find a config.json\n" );
                exit(1);
            }
        }
    } else {
        config_filepath = model_path + "config.json";
        if ( ! file_exists( config_filepath.c_str() ) ) {
            config_filepath = program_path + "moshi-config.json";
            if ( ! file_exists( config_filepath.c_str() ) ) {
                fprintf( stderr, "error: failed to find a config.json\n" );
                exit(1);
            }
        }
    }

    if ( moshi_get_config( &config, config_filepath.c_str() ) != 0 ) {
        fprintf( stderr, "error: reading config\n");
        exit(1);
    }

    if ( context > 0 ) {
        config.context = context;
    }

    std::string model_filepath = model_path + config.moshi_name;
    std::string mimi_filepath = model_path + config.mimi_name;
    std::string tokenizer_filepath = model_path + config.tokenizer_name;

    if ( ! file_exists( model_filepath.c_str() ) ) {
        fprintf( stderr, "error: missing moshi file \"%s\"\n", model_filepath.c_str() );
        exit(1);
    }

    if ( ! file_exists( mimi_filepath.c_str() ) ) {
        // files can be deleted or not downloaded to save memory
        bool found = false;
        std::vector<std::string> paths = {
            "kyutai/tts-1.6b-en_fr/tokenizer-e351c8d8-checkpoint125.safetensors",
            "kyutai/tts-0.75b-en-public/tokenizer-e351c8d8-checkpoint125.safetensors",
            "kyutai/stt-1b-en_fr-candle/mimi-pytorch-e351c8d8@125.safetensors",
            "kyutai/stt-2.6b-en/mimi-pytorch-e351c8d8@125.safetensors",
            "kyutai/stt-1b-en_fr/mimi-pytorch-e351c8d8@125.safetensors",
        };
        if ( model_root.size() ) {
            paths.push_back( model_root + "kyutai/tts-1.6b-en_fr/tokenizer-e351c8d8-checkpoint125.safetensors" );
            paths.push_back( model_root + "kyutai/tts-0.75b-en-public/tokenizer-e351c8d8-checkpoint125.safetensors" );
            paths.push_back( model_root + "kyutai/stt-1b-en_fr-candle/mimi-pytorch-e351c8d8@125.safetensors" );
            paths.push_back( model_root + "kyutai/stt-2.6b-en/mimi-pytorch-e351c8d8@125.safetensors" );
            paths.push_back( model_root + "kyutai/stt-1b-en_fr/mimi-pytorch-e351c8d8@125.safetensors" );
        }
        if ( program_path.size() ) {
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
                break;
            }
        }

        if ( ! found ) {
            fprintf( stderr, "error: missing mimi file \"%s\"\n", mimi_filepath.c_str() );
            exit(1);
        }
    }

    if ( ! file_exists( tokenizer_filepath.c_str() ) ) {
        // files can be deleted or not downloaded to save memory
        bool found = false;
        if ( config.tokenizer_name == "tokenizer_spm_32k_3.model" ) {
            // the file is the same for several models
            std::vector<std::string> paths = {
                "kyutai/moshika-pytorch-bf16/tokenizer_spm_32k_3.model",
                "kyutai/moshiko-pytorch-bf16/tokenizer_spm_32k_3.model"
            };
            if ( model_root.size() ) {
                paths.push_back( model_root + "kyutai/moshika-pytorch-bf16/tokenizer_spm_32k_3.model" );
                paths.push_back( model_root + "kyutai/moshiko-pytorch-bf16/tokenizer_spm_32k_3.model" );
            }
            if ( program_path.size() ) {
                paths.push_back( program_path + "kyutai/moshika-pytorch-bf16/tokenizer_spm_32k_3.model" );
                paths.push_back( program_path + "kyutai/moshiko-pytorch-bf16/tokenizer_spm_32k_3.model" );
            }
            for ( auto & path : paths ) {
                if ( file_exists( path.c_str() ) ) {
                    tokenizer_filepath = path;
                    found = true;
                    break;
                }
            }
        }
        if ( ! found ) {
            fprintf( stderr, "error: missing tokenizer file \"%s\"\n", tokenizer_filepath.c_str() );
            exit(1);
        }
    }

    if ( personaplex && personaplex_voice_filepath.size()
    && ! file_exists( personaplex_voice_filepath.c_str() ) ) {
        std::vector<std::string> paths;
        if ( personaplex_voice_filepath.size() == 5 ) {
            std::string expanded_filepath = model_path + "voices/" + personaplex_voice_filepath;
            paths.push_back( expanded_filepath + ".gguf" );
            paths.push_back( expanded_filepath + ".safetensors" );
        }
        paths.push_back( model_path + personaplex_voice_filepath );
        if ( model_root.size() ) {
            paths.push_back( model_root + personaplex_voice_filepath );
        }
        if ( program_path.size() ) {
            paths.push_back( program_path + personaplex_voice_filepath );
        }

        bool found = false;
        for ( auto & path : paths ) {
            if ( file_exists( path.c_str() ) ) {
                personaplex_voice_filepath = path;
                found = true;
                break;
            }
        }

        if ( ! found ) {
            fprintf( stderr, "error: failed to find voice file \"%s\"\n", personaplex_voice_filepath.c_str() );
            exit(1);
        }
    }

    // MARK: Loading

    srand( seed );
    printf( "seed: %d\n", seed );

    common_ggml_t ggml;
    init_ggml( ggml, device, n_threads );

    // context
    unref_ptr<moshi_context_t> moshi =  moshi_alloc( ggml.backend, ggml.backend_cpu );

    printf( "loading...\n" );
    auto load_start = ggml_time_ms();

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
    if ( config.dep_q >= config.n_q )
        num_codebooks = (int) config.dep_q;
    if ( personaplex )
        num_codebooks = 8;
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

    Decoder input_decoder;
    Resampler resampler;
    if ( input ) {
        input_decoder.init( input );
        AVChannelLayout mono;
        av_channel_layout_default( &mono, 1 );
        resampler.set_input( input_decoder.codec_ctx );
        resampler.set_output( 24000, AV_SAMPLE_FMT_FLT, mono, frame_size );
        resampler.init();
    }

    if ( personaplex && personaplex_voice_filepath.size() ) {
        moshi_lm_personaplex_load_voice( moshi, gen, personaplex_voice_filepath.c_str() );
    }

    /////////////////////////
    // MARK: SDL
    /////////////////////////

    AudioState input_state, output_state;
    SDL_AudioDeviceID cap_dev, dev;

    if ( ! bench ) {
        if (SDL_Init(SDL_INIT_AUDIO | SDL_INIT_TIMER) != 0) {
            fprintf(stderr, "Could not initialize SDL: %s\n", SDL_GetError());
            return 1;
        }

        sdl_init_frames( input_state, 3, frame_size*4 );

        SDL_AudioSpec want, have;

        want.freq = 24000; // Sample rate
        want.format = AUDIO_F32; // Audio format
        want.channels = 1; // Mono audio
        want.samples = frame_size;
        want.callback = sdl_capture_callback;
        want.userdata = &input_state;

        cap_dev = SDL_OpenAudioDevice(NULL, 1, &want, &have, 0);
        if (cap_dev <= 0) {
            fprintf(stderr, "Could not open audio: %s\n", SDL_GetError());
            return 1;
        }
        assert( want.freq == have.freq );
        assert( want.format == have.format );
        assert( want.channels == have.channels );
        assert( want.samples == have.samples );

        sdl_init_frames( output_state, 3, frame_size*4 );

        want.callback = sdl_audio_callback;
        want.userdata = &output_state;
        dev = SDL_OpenAudioDevice(NULL, 0, &want, &have, 0);
        if (dev <= 0) {
            fprintf(stderr, "Could not open audio: %s\n", SDL_GetError());
            return 1;
        }
        assert( want.freq == have.freq );
        assert( want.format == have.format );
        assert( want.channels == have.channels );
        assert( want.samples == have.samples );
    }

    /////////////////////////
    // MARK: Loop
    /////////////////////////

    // model
    moshi_lm_start( moshi, gen, depth_temperature, text_temperature );

    std::vector<int16_t> tokens(num_codebooks);
    int text_token;

    std::vector<float> blank(frame_size);

    if ( ! bench ) {
        SDL_PauseAudioDevice(cap_dev, 0);
        SDL_PauseAudioDevice(dev, 0);
    }

    AVFrame * dec_frame = input? dec_frame = input_decoder.frame() : NULL;
    AVFrame * res_frame = NULL;

    uint64_t lm_start = ggml_time_us();
    while ( ! shutdown ) {
        if ( input ) {
            if ( input_delay > 0 ) {
                input_delay--;
                memset(blank.data(), 0, blank.size() * sizeof(blank[0]));
                lm_start = ggml_time_us();
                mimi_encode_send( encoder, blank.data() );
                if ( input_delay == 0 ) {
                    printf(" | ");
                    fflush( stdout );
                }
            } else {
                if ( res_frame ) {
                    // drain resampler
                    res_frame = resampler.frame();
                }
                while ( ! res_frame ) { // fill resampler if needed
                    dec_frame = input_decoder.frame();
                    if ( ! dec_frame ) { // we are done
                        break;
                    } else {
                        res_frame = resampler.frame( dec_frame );
                    }
                }
                if ( res_frame ) {
                    lm_start = ggml_time_us();
                    mimi_encode_send( encoder, (float*)res_frame->data[0] );
                    res_frame = resampler.frame();
                } else {
                    // no more decoder frames
                    input = NULL;
                    memset(blank.data(), 0, blank.size() * sizeof(blank[0]));
                    lm_start = ggml_time_us();
                    mimi_encode_send( encoder, blank.data() );
                    printf(" | ");
                    fflush( stdout );
                }
            }
        } else if ( bench ) {
            memset(blank.data(), 0, blank.size() * sizeof(blank[0]));
            lm_start = ggml_time_us();
            mimi_encode_send( encoder, blank.data() );
        } else {
            // sdl_receive_frame can block, don't include in frame rate
            sdl_frame_t * input_frame = sdl_receive_frame( input_state, true );

            lm_start = ggml_time_us();
            mimi_encode_send( encoder, (float*)input_frame->data );
            lm_delta_time += ggml_time_us() - lm_start;

            sdl_free_frame( input_state, input_frame );
            lm_start = ggml_time_us();
        }

        mimi_encode_receive( encoder, tokens.data() );
        moshi_lm_send2( gen, tokens );

        if ( moshi_lm_receive( gen, text_token, tokens ) ) {
            // audio out
            mimi_decode_send( decoder, tokens.data() );

            if ( bench ) {
                mimi_decode_receive( decoder, blank.data() );
            } else {
                // sdl_get_frame can block, don't include in frame rate
                lm_delta_time += ggml_time_us() - lm_start;
                sdl_frame_t * frame = sdl_get_frame( output_state );
                lm_start = ggml_time_us();

                mimi_decode_receive( decoder, (float*)frame->data );
                sdl_send_frame( output_state, frame ); // this can block
            }
            lm_delta_time += ggml_time_us() - lm_start;
            lm_frames++;
            if ( bench && lm_frames >= 125 ) {
                break;
            }

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

    log_metrics();

    return 0;
}



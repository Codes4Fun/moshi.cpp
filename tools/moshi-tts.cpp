#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include <limits.h>

#include <moshi/moshi.h>
#include "ffmpeg_helpers.h"
#include "sdl_helper.h"
#include "util.h"

static void print_usage(const char * program) {
    fprintf( stderr, R"(usage: %s [option(s)] \"hello world\"

plays using sdl if output not specified.

option(s):
  -h,       --help             show this help message

  -l,       --list-devices     list hardware and exit.
  -d NAME,  --device NAME      use named hardware.
            --threads N        number of CPU threads.

  -r PATH,  --model-root PATH  path to where all kyutai models are stored and
                               replaces MODEL_CACHE environment variable. the
                               models at root are in subdirectories of
                               'organization/model'
  -m PATH,  --model PATH       path to where model is, can be relative to the
                               MODEL_CACHE environment variable, or program
                               directory, or working directory. by default is
                               'Codes4Fun/tts-1.6b-en_fr-GGUF'
  -q QUANT, --quantize QUANT   convert weights to: q8_0, q4_0, q4_k
  -g,       --gguf-caching     loads gguf if exists, saves gguf if it does not.
                               model is saved alongside the original
                               safetensors file.
  -v FNAME, --voice FNAME      path to voice model/prefix.

  -o FNAME, --output FNAME     output to file, can be mimi, wav, mp3, ogg, etc.
  -i FNAME, --input FNAME      input text file.

  -s N,     --seed N           seed value.
  -t N,     --temperature N    consistency vs creativity, default 0.6
            --bench            sets defaults for benching.
)", program );
    exit(1);
}

SDL_mutex * stdin_mutex;
SDL_cond * stdin_ready;
std::string stdin_text;

int stdin_thread_func( void * arg ) {
    char buffer[1024];
    while (true) {
        char * read = fgets( buffer, sizeof(buffer) - 1, stdin );
        if (! read ) {
            printf("fgets returned NULL\n");
            break;
        }
        SDL_LockMutex( stdin_mutex );
        stdin_text += buffer;
        SDL_CondSignal( stdin_ready );
        SDL_UnlockMutex( stdin_mutex );
    }
    return 0;
}

bool get_text( std::string & text, bool block ) {
    bool ready = false;
    SDL_LockMutex( stdin_mutex );
    if ( block ) {
        while ( ! stdin_text.size() ) {
            SDL_CondWait( stdin_ready, stdin_mutex );
        }
    }
    if ( stdin_text.size() ) {
        text = stdin_text;
        stdin_text = "";
        ready = true;
    }
    SDL_UnlockMutex( stdin_mutex );
    return ready;
}


#include <signal.h>
void signal_handler(int dummy) {
    printf("exit\n");
    exit(1);
}

////////////////
// MARK: Main
////////////////

int main(int argc, char *argv[]) {
    signal(SIGINT, signal_handler);

    const char * device = NULL;
    int n_threads = 0;

    const char * model_cache = getenv("MODEL_CACHE");
    std::string model_root = model_cache? model_cache : "";
    std::string tts_path = "Codes4Fun/tts-1.6b-en_fr-GGUF/";
    bool tts_path_set = false;
    const char * quant = NULL;
    bool gguf_caching = false;
    std::string voice_filename;

    const char * input_filename = NULL;
    const char * output_filename = NULL;

    bool seed_set = false;
    int seed = (int)time(NULL);
    bool temperature_set = false;
    float text_temperature = 0.6f;
    float depth_temperature = 0.6f;
    bool bench = false;

    const char * text = NULL;

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
            tts_path = argv[++i];
            tts_path_set = true;
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
        if (arg == "-v" || arg == "--voice") {
            if (i + 1 >= argc) {
                fprintf( stderr, "error: \"%s\" requires filepath to voice\n", argv[i] );
                exit(1);
            }
            voice_filename = argv[++i];
            continue;
        }
        if (arg == "-o" || arg == "--output") {
            if (i + 1 >= argc) {
                fprintf( stderr, "error: \"%s\" requires filepath to output file\n", argv[i] );
                exit(1);
            }
            output_filename = argv[++i];
            continue;
        }
        if (arg == "-i" || arg == "--input") {
            if (i + 1 >= argc) {
                fprintf( stderr, "error: \"%s\" requires filepath to input file\n", argv[i] );
                exit(1);
            }
            input_filename = argv[++i];
            continue;
        }
        if (arg == "-s" || arg == "--seed") {
            if (i + 1 >= argc) {
                fprintf( stderr, "error: \"%s\" requires value\n", argv[i] );
                exit(1);
            }
            seed_set = true;
            seed = std::stoi(argv[++i]);
            continue;
        }
        if (arg == "-t" || arg == "--temperature") {
            if (i + 1 >= argc) {
                fprintf( stderr, "error: \"%s\" requires value\n", argv[i] );
                exit(1);
            }
            temperature_set = true;
            text_temperature = (float) std::stod(argv[++i]);
            depth_temperature = text_temperature;
            continue;
        }
        if (arg == "--bench") {
            bench = true;
            continue;
        }
        if (arg[0] == '-') {
            fprintf( stderr, "error: unrecognized option \"%s\"\n", argv[i] );
            exit(1);
        }
        if (!text) {
            text = argv[i];
        } else {
            fprintf( stderr, "error: unexpected extra argument \"%s\"\n", argv[i] );
            exit(1);
        }
    }

    bool use_sdl = ! output_filename;
    if ( bench ) {
        if ( ! text && !input_filename )
            text = "The quick brown fox jumped over the sleeping dog.";
        if ( ! seed_set ) seed = 0;
        if ( ! temperature_set ) {
            text_temperature = 0;
            depth_temperature = 0;
        }
        use_sdl = false;
    }

    /////////////////////////
    // MARK: Validate Args
    /////////////////////////

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

    const char * ext = NULL;
    if ( output_filename ) {
        ext = get_ext( output_filename );
        if ( ! ext ) {
            fprintf( stderr, "unable to determine output file type without ext.\n" );
            print_usage(argv[0]);
        }
    }

    std::string program_path = get_program_path(argv[0]);
    ensure_path( program_path );
    ensure_path( model_root );
    ensure_path( tts_path );

    std::string tts_config_path = tts_path + "config.json";
    if ( ! file_exists( tts_config_path.c_str() ) ) {
        // is path specific (aka absolute or relative)
        if ( is_abs_or_rel( tts_config_path ) ) {
            fprintf( stderr, "error: failed to find config.json from path: \"%s\"\n", tts_path.c_str() );
            exit(1);
        }
        std::vector<std::string> paths;
        if ( tts_path_set ) {
            paths.push_back( "kyutai/" + tts_path );
            if ( model_root.size() ) {
                paths.push_back( model_root + tts_path );
                paths.push_back( model_root + "kyutai/" + tts_path );
            }
            if ( program_path.size() ) {
                paths.push_back( program_path + tts_path );
                paths.push_back( program_path + "kyutai/" + tts_path );
            }
        } else {
            // try default paths
            paths.push_back( "kyutai/tts-1.6b-en_fr/" );
            if ( model_root.size() ) {
                paths.push_back( model_root + tts_path );
                paths.push_back( model_root + "kyutai/tts-1.6b-en_fr/" );
            }
            if ( program_path.size() ) {
                paths.push_back( program_path + tts_path );
                paths.push_back( program_path + "kyutai/tts-1.6b-en_fr/" );
            }
        }
        bool found = false;
        for ( auto & path : paths ) {
            tts_config_path = path + "config.json";
            if ( file_exists( tts_config_path.c_str() ) ) {
                tts_path = path;
                found = true;
                break;
            }
        }
        if ( ! found ) {
            fprintf( stderr, "error: failed to find config.json from path: \"%s\"\n", tts_path.c_str() );
            exit(1);
        }
    }
    printf( "found model path: %s\n", tts_path.c_str() );

    if ( input_filename ) {
        if ( ! file_exists( input_filename ) ) {
            fprintf( stderr, "error: failed to find input file: \"%s\"\n", input_filename );
            exit(1);
        }
    }

    moshi_config_t tts_config;
    if ( moshi_get_config( &tts_config, tts_config_path.c_str() ) != 0 ) {
        fprintf( stderr, "error: reading tts config\n");
        exit(1);
    }

    // find/check files in the config
    std::string tokenizer_filepath = tts_path + tts_config.tokenizer_name;
    if ( ! file_exists( tokenizer_filepath.c_str() ) ) {
        bool found = false;
        if ( tts_config.tokenizer_name == "tokenizer_spm_8k_en_fr_audio.model"
          || tts_config.tokenizer_name == "tokenizer_en_fr_audio_8000.model"
        ) {
            // the file is the same for several models
            std::vector<std::string> paths = {
                "kyutai/tts-1.6b-en_fr/tokenizer_spm_8k_en_fr_audio.model",
                "kyutai/tts-0.75b-en-public/tokenizer_spm_8k_en_fr_audio.model",
                "stt-1b-en_fr-candle/tokenizer_en_fr_audio_8000.model"
            };
            if ( model_root.size() ) {
                paths.push_back( model_root + "kyutai/tts-1.6b-en_fr/tokenizer_spm_8k_en_fr_audio.model" );
                paths.push_back( model_root + "kyutai/tts-0.75b-en-public/tokenizer_spm_8k_en_fr_audio.model" );
                paths.push_back( model_root + "stt-1b-en_fr-candle/tokenizer_en_fr_audio_8000.model" );
            }
            if ( program_path.size() ) {
                paths.push_back( program_path + "kyutai/tts-1.6b-en_fr/tokenizer_spm_8k_en_fr_audio.model" );
                paths.push_back( program_path + "kyutai/tts-0.75b-en-public/tokenizer_spm_8k_en_fr_audio.model" );
                paths.push_back( program_path + "stt-1b-en_fr-candle/tokenizer_en_fr_audio_8000.model" );
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

    std::string moshi_filepath = tts_path + tts_config.moshi_name;
    if ( ! file_exists( moshi_filepath.c_str() ) ) {
        fprintf( stderr, "error: missing moshi file \"%s\"\n", moshi_filepath.c_str() );
        exit(1);
    }

    std::string mimi_filepath = tts_path + tts_config.mimi_name;
    if ( ! file_exists( mimi_filepath.c_str() ) ) {
        bool found = false;
        // the file is the same for all models
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

    // default voice filepaths
    if ( ! voice_filename.size() ) {
        if ( tts_config.cross_attention )
            voice_filename = "kyutai/tts-voices/expresso/ex03-ex01_happy_001_channel1_334s.wav.1e68beda@240.safetensors";
        else
            voice_filename = "kyutai/tts-voices/expresso/ex03-ex01_happy_001_channel1_334s.wav";
    } else {
        auto voice_ext = get_ext( voice_filename.c_str() );
        if ( ! voice_ext ) {
            fprintf( stderr, "error: unable to determine voice file type without ext.\n" );
            print_usage(argv[0]);
        }
        if ( tts_config.cross_attention ) {
            if ( strcmp( ".safetensors", voice_ext ) ) {
                fprintf( stderr, "error: expected safetensor for voice file.\n" );
                print_usage(argv[0]);
            }
        }
    }

    if ( ! file_exists( voice_filename.c_str() ) ) {
        // is path specific (aka absolute or relative)
        if ( is_abs_or_rel( voice_filename ) ) {
            fprintf( stderr, "error: failed to find voice file: \"%s\"\n", voice_filename.c_str() );
            exit(1);
        }
        std::vector<std::string> paths = { "kyutai/tts-voices/" + voice_filename };
        if ( model_root.size() ) {
            paths.push_back( model_root + voice_filename );
            paths.push_back( model_root + "kyutai/tts-voices/" + voice_filename );
        }
        if ( program_path.size() ) {
            paths.push_back( program_path + voice_filename );
            paths.push_back( program_path + "kyutai/tts-voices/" + voice_filename );
        }
        bool found = false;
        for ( auto & path : paths ) {
            if ( file_exists( path.c_str() ) ) {
                voice_filename = path;
                found = true;
                break;
            }
        }
        if ( ! found ) {
            fprintf( stderr, "error: failed to find voice file: \"%s\"\n", voice_filename.c_str() );
            exit(1);
        }
    }

    std::string model_gguf = "";
    if ( gguf_caching ) {
        if ( quant ) {
            model_gguf = moshi_filepath + "." + quant + ".gguf";
            if ( file_exists( model_gguf.c_str() ) ) {
                moshi_filepath = model_gguf;
                model_gguf = "";
                quant = NULL;
            }
        } else {
            model_gguf = moshi_filepath + ".gguf";
            if ( file_exists( model_gguf.c_str() ) ) {
                moshi_filepath = model_gguf;
                model_gguf = "";
            }
        }
    }

    ///////////////////////////////////////////////
    // MARK: Open / Allocate
    ///////////////////////////////////////////////

    stdin_mutex = SDL_CreateMutex();
    stdin_ready = SDL_CreateCond();

    // context
    unref_ptr<moshi_context_t> moshi =  moshi_alloc( device );
    if ( n_threads > 0 && n_threads < 512 ) {
        moshi_set_n_threads( moshi, n_threads );
        printf( "set threads to %d\n", n_threads );
    }

    // model
    unref_ptr<moshi_lm_t> lm = moshi_lm_from_files( moshi, &tts_config,
        moshi_filepath.c_str() );
    if ( quant ) {
        if ( ! moshi_lm_quantize( lm, quant ) ) {
            fprintf( stderr, "error: unknown quant %s\n", quant );
            exit(-1);
        }
    }

    // generator
    unref_ptr<moshi_lm_gen_t> gen = moshi_lm_generator( lm );

    // voice
    own_ptr<Decoder> voice_decoder;
    std::string voice_ext = get_ext( voice_filename.c_str() );
    if ( tts_config.cross_attention ) {
        if ( moshi_lm_set_voice_condition( moshi, gen, voice_filename.c_str() ) != 0 ) {
            if ( voice_ext != ".safetensors" || voice_ext != ".sft" ) {
                fprintf( stderr, "error: not a safetensors file: \"%s\"\n", voice_filename.c_str() );
            } else {
                fprintf( stderr, "error: failed to open safetensors file: \"%s\"\n", voice_filename.c_str() );
            }
            exit(1);
        }
    } else {
        voice_decoder = new Decoder;
        assert( voice_decoder );
        try {
            voice_decoder->init( voice_filename.c_str() );
        } catch (const std::exception& e) {
            fprintf( stderr, "error: decoder failed to open: %s\n", voice_filename.c_str() );
            fprintf( stderr, "error: %s\n", e.what() );
            exit( 1 );
        }
    }

    // output
    AVChannelLayout mono;
    av_channel_layout_default( &mono, 1 );
    unref_ptr<FILE> mimi_file;
    own_ptr<Encoder> encoder;
    if ( output_filename ) {
        if ( strcmp( ext, ".mimi" ) == 0 ) {
            mimi_file = fopen( output_filename, "wb" );
            if ( ! mimi_file ) {
                fprintf( stderr, "error: unable to open file for writing: %s\n", output_filename );
                exit( 1 );
            }
            auto n = fwrite( "MIMI", 4, 1, mimi_file );
            assert( n == 1 );
            n = fwrite( &tts_config.n_q, 4, 1, mimi_file );
            assert( n == 1 );
        } else {
            encoder = new Encoder();
            encoder->init_from( output_filename, 24000, AV_SAMPLE_FMT_FLT, mono );
        }
    }
    if ( use_sdl ) {
        if ( SDL_Init(SDL_INIT_AUDIO | SDL_INIT_TIMER) < 0 ) {
            fprintf( stderr, "error: Could not initialize SDL: %s\n", SDL_GetError() );
            exit( 1 );
        }
    }

    printf("done preparing loads.\n");

    ///////////////////////
    // MARK: Load / Read
    ///////////////////////

    auto load_start = ggml_time_ms();

    // maybe ordered from dependency and quickest to fail

    // tokenizer
    unref_ptr<tokenizer_t> tts_tok = tokenizer_alloc( tokenizer_filepath.c_str(),
        tts_config.cross_attention );

    // codec
    unref_ptr<mimi_codec_t> codec = mimi_alloc( moshi, mimi_filepath.c_str(), (int) tts_config.n_q );
    float frame_rate = mimi_frame_rate( codec );
    int frame_size = mimi_frame_size( codec );

    const int delay_steps = (int)( tts_config.tts_config.audio_delay * frame_rate );
    assert( delay_steps == 16 );
    // we invasively put the on_audio_hook in lm, so we need to copy delay_steps
    moshi_lm_set_delay_steps( lm, delay_steps );

    // mimi codec dependents
    unref_ptr<AVFrame> mimi_frame;
    own_ptr<Resampler> resampler;
    AudioState state;
    if ( encoder ) {
        mimi_frame = av_frame_alloc();
        mimi_frame->nb_samples     = frame_size;
        mimi_frame->ch_layout      = mono;
        mimi_frame->format         = AV_SAMPLE_FMT_FLT;
        mimi_frame->sample_rate    = 24000;
        check_error( av_frame_get_buffer( mimi_frame, 0 ),
            "Error making frame buffer" );

        resampler = new Resampler;
        resampler->set_input( 24000, AV_SAMPLE_FMT_FLT, mono, frame_size );
        resampler->set_output( encoder->codec_ctx );
        resampler->init();
    }
    if ( use_sdl ) {
        int sample_rate = 24000;
        int format = AUDIO_F32;
        int nb_samples = 1920;
        int nb_bytes = nb_samples * 4;

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
    }

    // model
    moshi_lm_load( lm );
    if ( model_gguf.size() ) {
        moshi_lm_save_gguf( lm, model_gguf.c_str() );
    }

    // conditions for some models
    if ( tts_config.cross_attention ) {
        moshi_lm_load_voice_condition( moshi, gen );
    } else {
        std::deque<int> text_prefixes;
        std::deque<std::vector<int>> audio_prefixes;
        load_voice_prefix(
            text_prefixes,
            audio_prefixes,
            voice_decoder,
            codec,
            lm,
            (int) tts_config.n_q
        );
        moshi_lm_voice_prefix( gen, text_prefixes, audio_prefixes );
    }

    auto load_end = ggml_time_ms();
    printf("done loading. %f\n", (load_end - load_start) / 1000.f);

    /////////////////////////////
    // MARK: Initialize States
    /////////////////////////////

    srand( seed );
    printf( "seed: %d\n", seed );

    // decoder
    unref_ptr<mimi_decode_context_t> decoder;
    if ( ! mimi_file ) {
        decoder = mimi_decode_alloc_context( codec );
    }

    // model
    moshi_lm_start( moshi, gen, depth_temperature, text_temperature );
    int int_text_token;
    std::vector<int16_t> int16_audio_tokens;

    // read in text file
    // TODO: load and tokenize it in chunks instead of the whole thing
    own_ptr<char> input_file_text;
    if ( input_filename ) {
        auto f = fopen( input_filename, "rb" );
        if ( ! f ) {
            fprintf( stderr, "error: unable to open \"%s\"\n", input_filename );
            exit( 1 );
        }
        auto e = fseek( f, 0, SEEK_END );
        assert( e == 0 );
        auto size = ftell( f );
        assert( size > 0 );
        e = fseek( f, 0, SEEK_SET );
        assert( e == 0 );
        input_file_text = new char[size];
        assert( input_file_text );
        auto n = fread( input_file_text, size, 1, f );
        assert( n == 1 );
        fclose( f );
        text = input_file_text;
    }
    if ( text ) {
        tokenizer_send( tts_tok, text );
        tokenizer_send( tts_tok, "" ); // flush
    } else {
        tokenizer_send( tts_tok, "done loading " );
    }

    // tokens
    Entry entry;

    if ( use_sdl )
        SDL_PauseAudioDevice(state.device_id, 0);

    SDL_Thread *stdin_thread = NULL;
    if ( ! text ) {
        stdin_thread = SDL_CreateThread( stdin_thread_func, "stdin_thread",  NULL );
        if ( ! stdin_thread ) {
            fprintf( stderr, "error: failed to create thread: %s\n", SDL_GetError() );
            exit(1);
        }
    }

    /////////////////////
    // MARK: Main Loop
    /////////////////////

    int64_t gen_start = ggml_time_ms();
    int64_t lm_start = gen_start;
    int64_t lm_delta_time = 0;
    int64_t lm_tokens = 0;
    int64_t lm_frames = 0;

    bool machine_clean = moshi_lm_is_empty( gen ) && tokenizer_empty( tts_tok );
    bool active = true;
    while (active) {
        if ( ! text ) { // stdin
            active = true;

            std::string text;
            if ( get_text( text, machine_clean ) ) {
                tokenizer_send( tts_tok, text.c_str() );
            }
        } else {
            active = false;
        }

        // add at least 4 tokens for the look ahead
        for ( int i = 0; i < 4 && tokenizer_receive( tts_tok, &entry ); ++i ) {
            if ( machine_clean ) {
                moshi_lm_machine_reset( gen );
                lm_start = ggml_time_ms();
            }
            lm_tokens++;
            moshi_lm_send( gen, &entry );
            active = true;
            machine_clean = false;
        }

        if ( moshi_lm_receive( gen, int_text_token, int16_audio_tokens ) ) {
            lm_frames++;
            if ( mimi_file ) {
                auto n = fwrite( int16_audio_tokens.data(), tts_config.n_q*2, 1, mimi_file );
                assert( n == 1 );
            }

            if ( decoder ) {
                mimi_decode_send( decoder, int16_audio_tokens.data() );
                // TODO: look into possibly supporting parallel sdl and file out
                if ( encoder ) {
                    mimi_decode_receive( decoder, (float*)mimi_frame->data[0] );
                    auto frame = resampler->frame( mimi_frame );
                    while ( frame ) {
                        encoder->frame( frame );
                        frame = resampler->frame();
                    }
                } else if ( use_sdl ) {
                    sdl_frame_t * frame = sdl_get_frame( state );
                    mimi_decode_receive( decoder, (float*)frame->data );
                    sdl_send_frame( state, frame );
                } else {
                    mimi_decode_receive( decoder, NULL );
                }
            }
        }
        if ( moshi_lm_is_active( gen ) ) {
            active = true;
        } else if ( moshi_lm_is_empty( gen ) ) {
            // reset?
            if ( ! machine_clean ) {
                auto lm_end = ggml_time_ms();
                lm_delta_time += lm_end - lm_start;
                lm_start = lm_end;

                printf("rest\n");
                machine_clean = true;
            }
        }
    }

    auto gen_end = ggml_time_ms();
    printf("done generating. %f\n", (gen_end - gen_start) / 1000.f);

    printf("token count: %4d tokens\n", (int)lm_tokens);
    printf("frame count: %4d frames\n", (int)lm_frames);
    printf("token rate:  %f tokens/s\n", lm_tokens * 1000.f / lm_delta_time);
    printf("frame rate:  %f frames/s\n", lm_frames * 1000.f / lm_delta_time );

    ////////////////
    // MARK: Exit
    ////////////////

    if ( encoder )
        encoder->flush();

    if ( use_sdl ) {
        SDL_Delay(1);
        SDL_CloseAudioDevice(state.device_id);
        SDL_Quit();
    }

    return 0;
}

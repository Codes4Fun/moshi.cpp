#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <iostream> // tts
#include <pthread.h>

#include <limits.h>
#include <unistd.h>
#include <libgen.h>

#include <moshi/moshi.h>
#include "ffmpeg_helpers.h"
#include "sdl_helper.h"
#include "util.h"

//#define DEFAULT_BIG
#define SDL_OUT

static void print_usage(const char * program) {
    fprintf( stderr, "usage: %s [option(s)]\n", program );
    fprintf( stderr, "\nlistens to sdl audio capture if input not specified.\n" );
    fprintf( stderr, "outputs to console if output not specified.\n" );
    fprintf( stderr, "\noption(s):\n" );
    fprintf( stderr, "  -h,       --help             show this help message\n" );
    fprintf( stderr, "  -sm PATH, --stt-model PATH   path to stt model.\n" );
    fprintf( stderr, "  -l,       --list-devices     list hardware and exit.\n" );
    fprintf( stderr, "  -d NAME,  --device NAME      use named hardware.\n" );
    fprintf( stderr, "  -o FNAME, --output FNAME     output to text file.\n");
    fprintf( stderr, "  -i FNAME, --input FNAME      input file can be wav, mp3, ogg, etc.\n");
    fprintf( stderr, "            --debug            outputs each frames vad and token.\n");
    //fprintf( stderr, "  -s N,     --seed N           seed value.\n" );
    exit(1);
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
    const char * input_filename = NULL;
    const char * output_filename = NULL;
    bool output_debug = false;
#ifdef DEFAULT_BIG
    std::string stt_path = "kyutai/stt-2.6b-en";
#else
    std::string stt_path = "kyutai/stt-1b-en_fr-candle";
#endif
    int seed = (int)time(NULL);
    bool use_sdl = false;

    //////////////////////
    // MARK: Parse Args
    //////////////////////

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
        }
        if (arg == "-sm" || arg == "--stt_model") {
            if (i + 1 >= argc) {
                fprintf( stderr, "error: \"%s\" requires filepath to model\n", argv[i] );
                exit(1);
            }
            stt_path = argv[++i];
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
            seed = std::stoi(argv[++i]);
            continue;
        }
        if (arg == "--debug") {
            output_debug = true;
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
    // MARK: Validate Args
    /////////////////////////

    const char * ext = NULL;
    if ( input_filename ) {
        if ( access( input_filename, F_OK | R_OK ) != 0 ) {
            fprintf( stderr, "error: failed to find or access input file: \"%s\"\n", input_filename );
            exit(1);
        }
        ext = get_ext( input_filename );
        if ( ! ext ) {
            fprintf( stderr, "unable to determine input file type without ext.\n" );
            print_usage(argv[0]);
        }
    } else {
        use_sdl = true;
    }

    std::string program_path = get_program_path(argv[0]);

    auto stt_path_size = stt_path.size();
    if ( stt_path_size > 1 && stt_path[stt_path_size - 1] != '/' ) {
        stt_path += "/";
    }

    std::string stt_config_path = stt_path + "config.json";
    if ( access( stt_config_path.c_str(), F_OK | R_OK ) != 0 ) {
        // is path specific (aka absolute or relative)
        if ( is_abs_or_rel( stt_config_path ) ) {
            fprintf( stderr, "error: failed to find config.json from path: \"%s\"\n", stt_path.c_str() );
            exit(1);
        }
        std::vector<std::string> paths = {
            program_path + stt_path,
            "../" + stt_path,
            program_path + "../" + stt_path,
            "kyutai/" + stt_path,
            program_path + "kyutai/" + stt_path,
            "../kyutai/" + stt_path,
            program_path + "../kyutai/" + stt_path,
        };
        bool found = false;
        for ( auto & path : paths ) {
            stt_config_path = path + "config.json";
            if ( access( stt_config_path.c_str(), F_OK | R_OK ) == 0 ) {
                stt_path = path;
                found = true;
                break;
            }
        }
        if ( ! found ) {
            fprintf( stderr, "error: failed to find config.json from path: \"%s\"\n", stt_path.c_str() );
            exit(1);
        }
    }

    moshi_config_t stt_config;
    if ( moshi_get_config( &stt_config, stt_config_path.c_str() ) != 0 ) {
        fprintf( stderr, "error: reading stt config\n");
        exit(1);
    }

    // find/check files in the config
    std::string tokenizer_filepath = stt_path + stt_config.tokenizer_name;
    if ( access( tokenizer_filepath.c_str(), F_OK | R_OK ) != 0 ) {
        bool found = false;
        if ( stt_config.tokenizer_name == "tokenizer_spm_8k_en_fr_audio.model"
          || stt_config.tokenizer_name == "tokenizer_en_fr_audio_8000.model"
        ) {
            // the file is the same for all models, it can be at a shared location
            std::vector<std::string> paths = {
                "kyutai/tokenizer_spm_8k_en_fr_audio.model",
                "../kyutai/tokenizer_spm_8k_en_fr_audio.model",
                program_path + "kyutai/tokenizer_spm_8k_en_fr_audio.model",
                program_path + "../kyutai/tokenizer_spm_8k_en_fr_audio.model",
            };
            for ( auto & path : paths ) {
                if ( access( path.c_str(), F_OK | R_OK ) == 0 ) {
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

    std::string moshi_filepath = stt_path + stt_config.moshi_name;
    if ( access( moshi_filepath.c_str(), F_OK | R_OK ) != 0 ) {
        fprintf( stderr, "error: missing moshi file \"%s\"\n", moshi_filepath.c_str() );
        exit(1);
    }

    std::string mimi_filepath = stt_path + stt_config.mimi_name;
    if ( access( mimi_filepath.c_str(), F_OK | R_OK ) != 0 ) {
        bool found = false;
        // the file is the same for all models, it can be at a shared location
        std::vector<std::string> paths = {
            "kyutai/mimi-pytorch-e351c8d8@125.safetensors",
            "../kyutai/mimi-pytorch-e351c8d8@125.safetensors",
            program_path + "kyutai/mimi-pytorch-e351c8d8@125.safetensors",
            program_path + "../kyutai/mimi-pytorch-e351c8d8@125.safetensors",
        };
        for ( auto & path : paths ) {
            if ( access( path.c_str(), F_OK | R_OK ) == 0 ) {
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

    ///////////////////////////////////////////////
    // MARK: Open / Allocate
    ///////////////////////////////////////////////

    // context
    unref_ptr<moshi_context_t> moshi =  moshi_alloc( device );

    // model
    unref_ptr<moshi_lm_t> lm = moshi_lm_from_files( moshi, &stt_config, moshi_filepath.c_str() );

    // generator
    unref_ptr<moshi_lm_gen_t> gen = moshi_lm_generator( lm );

    // input
    own_ptr<Decoder> decoder;
    if ( input_filename ) {
        decoder = new Decoder;
        decoder->init( input_filename );
    }
    if ( use_sdl ) {
        if ( SDL_Init(SDL_INIT_AUDIO | SDL_INIT_TIMER) != 0 ) {
            fprintf( stderr, "error: Could not initialize SDL: %s\n", SDL_GetError() );
            exit( 1 );
        }
    }

    // output
    unref_ptr<FILE> output_file;
    FILE * out = stdout;
    bool output_srt = false;
    if ( output_filename ) {
        output_file = fopen( output_filename, "wb" );
        if ( ! output_file ) {
            fprintf( stderr, "error: unable to open file for writing: %s\n", output_filename );
            exit( 1 );
        }
        out = output_file;
        auto ext = get_ext( output_filename );
        if ( ext && strcmp( ext, ".srt" ) == 0 ) {
            output_srt = true;
        }
    }

    printf("done preparing loads.\n");

    ///////////////////////
    // MARK: Load / Read
    ///////////////////////

    // maybe ordered from dependency and quickest to fail

    // tokenizer
    sentencepiece::SentencePieceProcessor sp;
    sp.Load( tokenizer_filepath );

    // codec
    unref_ptr<mimi_codec_t> codec = mimi_alloc( moshi, mimi_filepath.c_str(), stt_config.n_q );
    float frame_rate = mimi_frame_rate( codec );
    int frame_size = mimi_frame_size( codec );
    int stt_frame_delay = stt_config.stt_config.audio_delay_seconds * frame_rate;

    // input (decoder/sdl)
    AVChannelLayout mono;
    av_channel_layout_default( &mono, 1 );
    own_ptr<Resampler> resampler;
    SDL_AudioDeviceID cap_dev;
    AudioState input_state;
#ifdef SDL_OUT
    AudioState output_state;
#endif
    if ( decoder ) {
        resampler = new Resampler;
        resampler->set_input( decoder->codec_ctx );
        resampler->set_output( 24000, AV_SAMPLE_FMT_FLT, mono, frame_size );
        resampler->init();
    }
    if ( use_sdl ) {
        int sample_rate = 24000;
        int format = AUDIO_F32;
        int nb_samples = frame_size;
        int nb_bytes = nb_samples * 4;

        SDL_AudioSpec want, have;
        SDL_zero( want );
        want.freq = sample_rate;
        want.format = format;
        want.channels = 1;
        want.samples = nb_samples;
        want.callback = sdl_capture_callback;
        want.userdata = &input_state;
        //input_state.log = true;
        sdl_init_frames( input_state, 3, nb_bytes );
        input_state.device_id = SDL_OpenAudioDevice(NULL, 1, &want, &have, 0);
        if (input_state.device_id <= 0) {
            fprintf(stderr, "Could not open audio: %s\n", SDL_GetError());
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

#ifdef SDL_OUT
        sdl_init_frames( output_state, 3, nb_bytes );
        want.callback = sdl_audio_callback;
        want.userdata = &output_state;
        output_state.device_id = SDL_OpenAudioDevice(NULL, 0, &want, &have, 0);
        if (output_state.device_id <= 0) {
            fprintf(stderr, "Could not open audio: %s\n", SDL_GetError());
            return 1;
        }
#endif
    }

    // model
    moshi_lm_load( lm );

    printf("done loading.\n");

    /////////////////////////////
    // MARK: Initialize States
    /////////////////////////////

    srand( seed );

    // mimi encoder
    unref_ptr<mimi_encode_context_t> encoder = mimi_encode_alloc_context( codec );

    // model
    moshi_lm_start( moshi, gen,
        stt_config.lm_gen_config.temp,
        stt_config.lm_gen_config.temp_text
    );

    if ( use_sdl ) {
        SDL_PauseAudioDevice(input_state.device_id, 0);
#ifdef SDL_OUT
        SDL_PauseAudioDevice(output_state.device_id, 0);
#endif
    }

    std::vector<int16_t> tokens(stt_config.n_q);
    std::vector<int> tokens2(stt_config.n_q);

    /////////////////////
    // MARK: Main Loop
    /////////////////////

    int vad_count = 0;
    bool last_print_was_vad = false;
    int extra = 8;
    int frame_count = 0;
    AVFrame * dec_frame, * frame = NULL;
    while ( true ) {
        if ( use_sdl ) {
            // NOTE: this blocks until a frame is ready
            sdl_frame_t * input_frame = sdl_receive_frame( input_state, true );
#ifdef SDL_OUT
            sdl_frame_t * output_frame = sdl_get_frame( output_state );
            memcpy( output_frame->data, input_frame->data, output_frame->nb_bytes );
            sdl_send_frame( output_state, output_frame );
#endif
            mimi_encode_send( encoder, (float*)input_frame->data );
            sdl_free_frame( input_state, input_frame );
        } else {
            if ( frame ) { // get next frame
                frame = resampler->frame();
            }
            if ( ! frame ) { // start of loop or resampler empty
                dec_frame = decoder->frame();
                frame = dec_frame ?
                    resampler->frame( dec_frame ) :
                    resampler->flush( true );
            }
            if ( ! dec_frame ) { // decoder empty
                if ( extra-- < 0) // needs extra frames to complete the tail
                    break;
            }
            if ( ! frame ) { // not enough decoded frames for the resampler
                continue;
            }
            mimi_encode_send( encoder, (float*)frame->data[0] );
        }

        mimi_encode_receive( encoder, tokens.data() );

        moshi_lm_send2( gen, tokens );

        int text_token;
        float vad = 0;
        moshi_lm_receive2( gen, text_token, vad );
        if ( output_debug ) {
            if ( vad > 0.5 ) {
                fprintf(out, "*%f", vad);
            } else {
                fprintf(out, "%f", vad);
            }
            if ( text_token != 0 ) {
                auto piece = sp.IdToPiece(text_token);
                std::string _text;
                for ( size_t ci = 0; ci < piece.size(); ci++ ) {
                    if ( piece.c_str()[ci] == -30 ) {
                        _text += ' ';
                        ci += 2;
                        continue;
                    }
                    _text += piece[ci];
                }
                fprintf(out, " %s\n", _text.c_str());
            } else {
                fprintf(out, "\n");
            }
        } else if ( output_srt ) {
            static int64_t start;
            static std::string acc_text;
            if ( ! last_print_was_vad && vad > 0.45 ) {
                if ( vad_count == 0 ) {
                    vad_count = 6;
                }
            }
            if ( text_token != 0 && text_token != 3 ) {
                if ( last_print_was_vad )
                    start = frame_count - stt_frame_delay;
                auto piece = sp.IdToPiece(text_token);
                std::string _text;
                for ( size_t ci = 0; ci < piece.size(); ci++ ) {
                    if ( piece.c_str()[ci] == -30 ) {
                        _text += ' ';
                        ci += 2;
                        continue;
                    }
                    _text += piece[ci];
                }
                //fprintf(out, "%s", _text.c_str());
                //if ( out == stdout )
                //    fflush( stdout );
                acc_text += _text;
                last_print_was_vad = false;
            }
            if ( vad_count > 0 ) {
                if ( --vad_count == 0 ) {
                    last_print_was_vad = true;
                    if ( acc_text.size() ) {
                        //fprintf(out, " [end of turn detected]\n");
                        int64_t end = frame_count - stt_frame_delay;
                        int64_t start_ms = start * 1000 / frame_rate;
                        int64_t end_ms = end * 1000 / frame_rate;
                        //int sh, sm, ss, sms, eh, em, es, ems;
                        int sh =  start_ms / 1000 / 60 / 60;
                        int sm =  start_ms / 1000 / 60 % 60;
                        int ss =  start_ms / 1000 % 60;
                        int sms = start_ms % 1000;
                        int eh =  end_ms / 1000 / 60 / 60;
                        int em =  end_ms / 1000 / 60 % 60;
                        int es =  end_ms / 1000 % 60;
                        int ems = end_ms % 1000;
                        fprintf(out, "%02d:%02d:%02d.%03d --> %02d:%02d:%02d.%03d\n",
                            sh, sm, ss, sms, eh, em, es, ems
                        );
                        const char * s = acc_text.c_str();
                        if ( *s == ' ' ) s++;
                        fprintf(out, "%s\n\n", s);
                        acc_text = "";
                    }
                }
            }
        } else {
#ifdef OLD_WAY
            if ( !last_print_was_vad && vad > 0.5 ) {
                fprintf(out, " [end of turn detected %f]\n", vad);
                last_print_was_vad = true;
            }
            if ( text_token != 0 && text_token != 3 ) {
                auto piece = sp.IdToPiece(text_token);
                std::string _text;
                for ( size_t ci = 0; ci < piece.size(); ci++ ) {
                    if ( piece.c_str()[ci] == -30 ) {
                        _text += ' ';
                        ci += 2;
                        continue;
                    }
                    _text += piece[ci];
                }
                fprintf(out, "%s", _text.c_str());
                if ( out == stdout )
                    fflush( stdout );
                last_print_was_vad = false;
            }
#else
            if ( ! last_print_was_vad && vad > 0.5 ) {
                if ( vad_count == 0 ) {
                    vad_count = 5;
                }
            }
            if ( text_token != 0 && text_token != 3 ) {
                auto piece = sp.IdToPiece(text_token);
                std::string _text;
                for ( size_t ci = 0; ci < piece.size(); ci++ ) {
                    if ( piece.c_str()[ci] == -30 ) {
                        _text += ' ';
                        ci += 2;
                        continue;
                    }
                    _text += piece[ci];
                }
                fprintf(out, "%s", _text.c_str());
                if ( out == stdout )
                    fflush( stdout );
                last_print_was_vad = false;
            }
            if ( vad_count > 0 ) {
                if ( --vad_count == 0 ) {
                    fprintf(out, " [end of turn detected]\n");
                    last_print_was_vad = true;
                }
            }
#endif
        }
        ++frame_count;
    }
    printf( "\n" );

    ////////////////
    // MARK: Exit
    ////////////////

    if ( use_sdl ) {
        SDL_CloseAudio();
        SDL_Quit();
    }

    return 0;
}

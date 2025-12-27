#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include <moshi/moshi.h>
#include "ffmpeg_helpers.h"
#include "sdl_helper.h"
#include "util.h"

#include <signal.h>
void signal_handler(int dummy) {
    printf("exit\n");
    exit(1);
}

int main(int argc, char *argv[]) {
    signal(SIGINT, signal_handler);

    const char * model_cache = getenv("MODEL_CACHE");
    std::string model_root = model_cache? model_cache : "";
    std::string sts_path = "kyutai/moshika-pytorch-bf16";
    const char * device = NULL;
    int n_threads = 4;

    auto load_start = ggml_time_ms();

    std::string program_path = get_program_path(argv[0]);
    ensure_path( program_path );
    ensure_path( model_root );
    ensure_path( sts_path );

    // default config
    moshi_config_t config;
    if ( moshi_get_config( &config, "moshi-config.json" ) != 0 ) {
        if ( moshi_get_config( &config, (program_path + "moshi-config.json").c_str() ) != 0 ) {
            fprintf( stderr, "error: reading config\n");
            exit(1);
        }
    }

    // context
    unref_ptr<moshi_context_t> moshi =  moshi_alloc( device );
    if ( n_threads > 0 && n_threads < 512 ) {
        moshi_set_n_threads( moshi, n_threads );
        printf( "set threads to %d\n", n_threads );
    }

    std::string model_path = sts_path + "model.safetensors";
    std::string mimi_path = sts_path + "tokenizer-e351c8d8-checkpoint125.safetensors";
    std::string tokenizer_path = sts_path + "tokenizer_spm_32k_3.model";
    if ( ! file_exists( model_path.c_str() ) ) {
        model_path = model_root + model_path;
        if ( ! file_exists( model_path.c_str() ) ) {
            fprintf( stderr, "unable to find model, set MODEL_CACHE to root of models. %s\n", model_path.c_str() );
            exit(1);
        }
        mimi_path = model_root + mimi_path;
        tokenizer_path = model_root + tokenizer_path;
    }

    // model
    unref_ptr<moshi_lm_t> lm = moshi_lm_from_files( moshi, &config,
        model_path.c_str() );

    // generator
    unref_ptr<moshi_lm_gen_t> gen = moshi_lm_generator( lm );

    // tokenizer
    unref_ptr<tokenizer_t> tok = tokenizer_alloc(
        tokenizer_path.c_str(),
        config.cross_attention );

    // codec
    int num_codebooks = (int)( config.n_q - config.dep_q );
    if ( config.dep_q > config.n_q )
        num_codebooks = (int) config.dep_q;
    printf( "num_codebooks: %d\n", num_codebooks );
    unref_ptr<mimi_codec_t> codec = mimi_alloc( moshi,
        mimi_path.c_str(),
        num_codebooks );
    float frame_rate = mimi_frame_rate( codec );
    int frame_size = mimi_frame_size( codec );

    //const int delay_steps = (int)( tts_config.tts_config.audio_delay * frame_rate );
    //assert( delay_steps == 16 );
     //we invasively put the on_audio_hook in lm, so we need to copy delay_steps
    //moshi_lm_set_delay_steps( lm, delay_steps );

    // model
    moshi_lm_load( lm );

    // encoder
    unref_ptr<mimi_encode_context_t> encoder;
    encoder = mimi_encode_alloc_context( codec );

    // decoder
    unref_ptr<mimi_decode_context_t> decoder;
    decoder = mimi_decode_alloc_context( codec );

    auto load_end = ggml_time_ms();
    printf("done loading. %f\n", (load_end - load_start) / 1000.f);



    // MARK: SDL

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



    // MARK: Loop

    // model
    float depth_temperature = 0.8f;
    float text_temperature = 0.7f;
    //float depth_temperature = 0.f;
    //float text_temperature = 0.f;
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

    SDL_PauseAudioDevice(cap_dev, 0);
    SDL_PauseAudioDevice(dev, 0);
    while ( true ) {
        // this blocks until a frame is available
        sdl_frame_t * input_frame = sdl_receive_frame( input_state, true );
        mimi_encode_send( encoder, (float*)input_frame->data );
        //mimi_encode_send( encoder, blank.data() );
        sdl_free_frame( input_state, input_frame );

        mimi_encode_receive( encoder, tokens.data() );
        moshi_lm_send2( gen, tokens );

        if ( moshi_lm_receive( gen, text_token, tokens ) ) {
            // audio out
            mimi_decode_send( decoder, tokens.data() );
            sdl_frame_t * frame = sdl_get_frame( output_state );
            mimi_decode_receive( decoder, (float*)frame->data );
            sdl_send_frame( output_state, frame );

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

    return 0;
}



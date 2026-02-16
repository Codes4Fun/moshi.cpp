#pragma once

#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>

#include <deque>

#include <sentencepiece_processor.h>

#include "ptrs.h"
#include "safetensor.h"

#if defined(_WIN32) && !defined(__MINGW32__)
#    ifdef MOSHI_BUILD
#        define MOSHI_API __declspec(dllexport) extern
#    else
#        define MOSHI_API __declspec(dllimport) extern
#    endif
#else
#    define MOSHI_API __attribute__ ((visibility ("default"))) extern
#endif

// MARK: Moshi Context

struct moshi_context_t;

MOSHI_API moshi_context_t * moshi_alloc( ggml_backend * backend, ggml_backend * backend_cpu );
MOSHI_API void unref( moshi_context_t * moshi );

// MARK: Mimi Codec

struct mimi_codec_t;

MOSHI_API mimi_codec_t * mimi_alloc( moshi_context_t * moshi, const char * filename, int n_q );
MOSHI_API void unref( mimi_codec_t * codec );
MOSHI_API float mimi_frame_rate( mimi_codec_t * codec );
MOSHI_API int mimi_frame_size( mimi_codec_t * codec );
MOSHI_API void mimi_save_gguf( mimi_codec_t * codec, const char * filepath );

// MARK: Mimi Encode

struct mimi_encode_context_t;

MOSHI_API mimi_encode_context_t * mimi_encode_alloc_context( mimi_codec_t * codec );
MOSHI_API void unref( mimi_encode_context_t * context );
MOSHI_API void mimi_encode_reset( mimi_encode_context_t * context );
MOSHI_API void mimi_encode_send( mimi_encode_context_t * context, float * frame );
MOSHI_API void mimi_encode_receive( mimi_encode_context_t * context, int16_t * tokens );

// MARK: Mimi Decode

struct mimi_decode_context_t;

MOSHI_API mimi_decode_context_t * mimi_decode_alloc_context( mimi_codec_t * codec );
MOSHI_API void unref( mimi_decode_context_t * context );
MOSHI_API void mimi_decode_reset( mimi_decode_context_t * context );
MOSHI_API void mimi_decode_send( mimi_decode_context_t * context, int16_t * tokens );
MOSHI_API void mimi_decode_receive( mimi_decode_context_t * context, float * frame );

// MARK: Tokenizer

struct Entry {
    std::vector<int> tokens;
    std::string text;
    int padding;
    int64_t time = 0;
};

struct tokenizer_t;

MOSHI_API tokenizer_t * tokenizer_alloc( const char * filepath, bool insert_bos = true );
MOSHI_API void unref( tokenizer_t * tok );
MOSHI_API bool tokenizer_empty( tokenizer_t * tok );
MOSHI_API int tokenizer_send( tokenizer_t * tok, std::string text );
MOSHI_API int tokenizer_receive( tokenizer_t * tok, Entry * entry );
MOSHI_API std::string tokenizer_id_to_piece( tokenizer_t * tok, int token );

// MARK: Config

struct config_fuser_t {
    bool cross_attention_pos_emb; // true
    float cross_attention_pos_emb_scale; // 1
    std::vector<std::string> sum; // [ "control", "cfg" ]
    // ? "prepend": [],
    std::vector<std::string> cross; // [ "speaker_wavs" ]
};

struct config_tts_t {
    float audio_delay; // 1.28
    int64_t second_stream_ahead; // 2
};

struct config_stt_t {
    float audio_delay_seconds; // 0.5
    float audio_silence_prefix_seconds; // 0.0
};

struct config_model_id_t {
    std::string sig;
    int64_t epoch;
};

struct config_lm_gen_t {
    float temp; // 0.6
    float temp_text; // 0.6
    int64_t top_k; // 250
    int64_t top_k_text; // 50
};

struct moshi_config_t {
    int64_t card; // 2048
    int64_t n_q; // 32 || 16
    int64_t dep_q; // 32 || 16
    std::vector<int64_t> delays; // 32 || 16
    int64_t dim; // 2048 || 1024
    int64_t text_card; // 8000
    int64_t existing_text_padding_id; // 3
    int64_t num_heads; // 16
    int64_t num_layers; // 16 || 24
    float hidden_scale; // 4.125
    bool causal; // true
    // layer_scale = NULL;
    int64_t context; // 500
    int64_t max_period; // 10000
    std::string gating;// "silu"
    std::string norm; // "rms_norm_f32"
    std::string positional_embedding; // "rope"
    int64_t depformer_dim; // 1024
    int64_t depformer_num_heads; // 16
    int64_t depformer_num_layers; // 4
    //int64_t depformer_dim_feedforward; // 3072   no needed, it's in the weight files
    bool depformer_multi_linear; // true
    int64_t depformer_context; // 0
    int64_t depformer_max_period; // 0
    std::string depformer_gating; // ""
    std::string depformer_pos_emb; // "none"
    bool depformer_weights_per_step; // true
    int64_t depformer_low_rank_embeddings; // 128
    bool demux_second_stream; // true
    // text_card_out = null || 5
    // conditioners_t * conditioners; // NULL
    config_fuser_t fuser;
    bool cross_attention; // true || false
    int64_t extra_heads_num_heads;
    //int extra_heads_dim; // this will come from weights
    config_tts_t tts_config;
    config_stt_t stt_config;
    config_model_id_t model_id;
    std::vector<int64_t> depformer_weights_per_step_schedule; // 32 || 16
    std::string model_type;
    config_lm_gen_t lm_gen_config;
    std::string tokenizer_name; // "tokenizer_spm_8k_en_fr_audio.model"
    std::string mimi_name; // "tokenizer-e351c8d8-checkpoint125.safetensors",
    std::string moshi_name; // "dsm_tts_1e68beda@240.safetensors" || "dsm_tts_d6ef30c7@1000.safetensors"
};

MOSHI_API int moshi_get_config( moshi_config_t * config, const char * filename );

// MARK: LM

struct moshi_lm_t;

MOSHI_API moshi_lm_t * moshi_lm_from_files(
    moshi_context_t * moshi,
    moshi_config_t * config,
    const char * filepath
);
MOSHI_API void unref( moshi_lm_t * lm );
MOSHI_API void moshi_lm_set_delay_steps( moshi_lm_t * lm, int delay_steps );
MOSHI_API int moshi_lm_get_max_delay( moshi_lm_t * lm );
MOSHI_API int moshi_lm_get_delay_steps( moshi_lm_t * lm );
MOSHI_API bool moshi_lm_quantize( moshi_lm_t * lm, const char * quant );
MOSHI_API int moshi_lm_load( moshi_lm_t * lm );
MOSHI_API void moshi_lm_save_gguf( moshi_lm_t * lm, const char * filepath );

// MARK: Generator

struct moshi_lm_gen_t;

MOSHI_API moshi_lm_gen_t * moshi_lm_generator( moshi_lm_t * lm );
MOSHI_API void unref( moshi_lm_gen_t * gen );

MOSHI_API int moshi_lm_set_voice_condition( moshi_context_t * moshi, moshi_lm_gen_t * gen, const char * filepath );
MOSHI_API int moshi_lm_load_voice_condition( moshi_context_t * moshi, moshi_lm_gen_t * gen );
MOSHI_API int moshi_lm_voice_prefix( moshi_lm_gen_t * gen, std::deque<int> & text_prefix, std::deque<std::vector<int>> & audio_prefix );
MOSHI_API int moshi_lm_personaplex_load_voice( moshi_context_t * moshi, moshi_lm_gen_t * gen, const char * filename );

MOSHI_API void moshi_lm_start( moshi_context_t * moshi, moshi_lm_gen_t * gen, float depth_temperature, float text_temperature, bool logging = false );
MOSHI_API void moshi_lm_send( moshi_lm_gen_t * gen, Entry * entry );
MOSHI_API int moshi_lm_receive( moshi_lm_gen_t * gen, int & text_token, std::vector<int16_t> & audio_tokens );
MOSHI_API void moshi_lm_send2( moshi_lm_gen_t * gen, std::vector<int16_t> & audio_tokens );
MOSHI_API void moshi_lm_receive2( moshi_lm_gen_t * gen, int & text_token, float & vad );
MOSHI_API int moshi_lm_is_active( moshi_lm_gen_t * gen );
MOSHI_API int moshi_lm_is_empty( moshi_lm_gen_t * gen );
MOSHI_API void moshi_lm_machine_reset( moshi_lm_gen_t * gen );

// MARK: Misc


/*int moshi_lm_n_q( moshi_lmmodel_t * lm );
int moshi_lm_max_delay( moshi_lmmodel_t * lm );
int moshi_lm_delay_steps( moshi_lmmodel_t * lm );*/

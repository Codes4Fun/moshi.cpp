#pragma once

#include <sentencepiece_processor.h>

struct conditioners_t {
    ggml_tensor * cfg_embed_weight;
    ggml_tensor * cfg_learnt_padding;
    ggml_tensor * cfg_output_proj_weight;
    ggml_tensor * control_embed_weight;
    ggml_tensor * control_learnt_padding;
    ggml_tensor * control_output_proj_weight;
    ggml_tensor * speaker_wavs_learnt_padding;
    ggml_tensor * speaker_wavs_output_proj_weight;
};

void get_weights( WeightLoader * loader, conditioners_t * cond ) {
    // TODO: remove prefix "lm"
    loader->fetch( &cond->cfg_embed_weight, "lm.condition_provider.conditioners.cfg.embed.weight" );
    loader->fetch( &cond->cfg_learnt_padding, "lm.condition_provider.conditioners.cfg.learnt_padding" );
    loader->fetch( &cond->cfg_output_proj_weight, "lm.condition_provider.conditioners.cfg.output_proj.weight" );
    loader->fetch( &cond->control_embed_weight, "lm.condition_provider.conditioners.control.embed.weight" );
    loader->fetch( &cond->control_learnt_padding, "lm.condition_provider.conditioners.control.learnt_padding" );
    loader->fetch( &cond->control_output_proj_weight, "lm.condition_provider.conditioners.control.output_proj.weight" );
    loader->fetch( &cond->speaker_wavs_learnt_padding, "lm.condition_provider.conditioners.speaker_wavs.learnt_padding" );
    loader->fetch( &cond->speaker_wavs_output_proj_weight, "lm.condition_provider.conditioners.speaker_wavs.output_proj.weight" );
}

struct voice_t {
    ggml_context * ctx;
    ggml_backend_buffer * buffer;
    ggml_tensor * sum;
    ggml_tensor * cross;
    std::deque<int> text_prefixes;
    std::deque<std::vector<int>> audio_prefixes;
};

struct moshi_ttsmodel_t {
    ScratchContext * scratch_cpu;
    ScratchContext * scratch;

    moshi_lmmodel_t * lm;
    WeightLoader * weights;

    moshi_mimi_t * mimi;
    WeightLoader * mimi_weights;

    sentencepiece::SentencePieceProcessor sp;

    int delay_steps;

    bool uses_cross;
    conditioners_t cond;
    voice_t voice;
};

moshi_ttsmodel_t * moshi_ttsmodel( ggml_backend * backend, std::string path = "kyutai/tts-1.6b-en_fr" ) {
    auto config = get_config( (path + "/config.json").c_str() );
    assert( config );
    assert( config->positional_embedding == "rope" );
    assert( config->norm == "rms_norm_f32" );
    std::string lm_path = path + "/" + config->moshi_name;
    std::string mimi_path = path + "/" + config->mimi_name;
    std::string tokenizer_path = path + "/" + config->tokenizer_name;

    auto tts = new moshi_ttsmodel_t;

    tts->scratch_cpu = new ScratchContext( 256 );
    tts->scratch = new ScratchContext( 256, backend );

    auto lm_safetensor = SafeTensorFile::from_file( lm_path.c_str() );
    tts->lm = moshi_lmmodel_alloc_default( config );
    tts->weights = new WeightLoader( lm_safetensor, tts->scratch_cpu, backend );
    get_weights( tts->weights, "lm.", tts->lm );
    get_weights( tts->weights, &tts->cond );
    {CAPTURE_GROUP("lm");
    tts->weights->load();}

    auto mimi_safetensor = SafeTensorFile::from_file( mimi_path.c_str() );
    tts->mimi = moshi_mimi_alloc_default( config->n_q );
    tts->mimi_weights = new WeightLoader( mimi_safetensor, tts->scratch_cpu, backend );
    get_weights( tts->mimi_weights, "mimi.quantizer.", tts->mimi->quantizer );
    get_weights( tts->mimi_weights, "mimi.upsample.convtr.", tts->mimi->upsample );
    get_weights( tts->mimi_weights, "mimi.decoder_transformer.transformer.", tts->mimi->decoder_transformer );
    get_weights( tts->mimi_weights, "mimi.decoder.", tts->mimi->decoder );
    if ( tts->mimi->encoder ) {
        get_weights( tts->mimi_weights, "mimi.downsample.conv.", tts->mimi->downsample );
        get_weights( tts->mimi_weights, "mimi.encoder_transformer.transformer.", tts->mimi->encoder_transformer );
        get_weights( tts->mimi_weights, "mimi.encoder.", tts->mimi->encoder );
    }
    {CAPTURE_GROUP("mimi");
    tts->mimi_weights->load();}

    tts->sp.Load( tokenizer_path );

    tts->delay_steps = config->tts_config.audio_delay * tts->mimi->frame_rate;
    assert( tts->delay_steps == 16 );
    // we invasively put the on_audio_hook in lm, so we need to copy delay_steps
    tts->lm->delay_steps = tts->delay_steps;

    tts->uses_cross = config->cross_attention;

    tts->voice.ctx = NULL;
    tts->voice.buffer = NULL;
    tts->voice.sum = NULL;
    tts->voice.cross = NULL;

    return tts;
}

bool load_voice(
        std::string filename,
        moshi_ttsmodel_t * tts,
        ggml_backend * backend ) {
    SafeTensorFile * stf = SafeTensorFile::from_file( filename.c_str() );
    auto loader = new WeightLoader( stf, tts->scratch_cpu, backend );
    // TODO: remove prefix "voice"
    ggml_tensor * speaker_wavs;
    loader->fetch( &speaker_wavs, "voice.speaker_wavs" );
    {
        CAPTURE_GROUP("voice");
        loader->load();
    }

    CAPTURE_GROUP("load_voice");
    ScratchContext &b_voice_ctx = *tts->scratch;
    auto cond = &tts->cond;
    auto voice = &tts->voice;

    // cfg {'1.0': 0, '1.5': 1, '2.0': 2, '2.5': 3, '3.0': 4, '3.5': 5, '4.0': 6}
    auto cfg_val = b_voice_ctx.constant( 2 );
    auto cfg_emb = ggml_get_rows(b_voice_ctx, cond->cfg_embed_weight, cfg_val);
    auto cfg_cond = ggml_mul_mat(b_voice_ctx, cond->cfg_output_proj_weight, cfg_emb);

    // control {'ok': 0}
    auto control_val = b_voice_ctx.constant( 0 );
    auto control_emb = ggml_get_rows(b_voice_ctx, cond->control_embed_weight, control_val);
    auto control_cond = ggml_mul_mat(b_voice_ctx, cond->control_output_proj_weight, control_emb);

    auto condition_sum = ggml_add(b_voice_ctx, cfg_cond, control_cond);

    // speaker_wavs
    auto speaker_wavs_a = ggml_cont(b_voice_ctx, ggml_transpose(b_voice_ctx, speaker_wavs));
    auto speaker_wavs_b = ggml_mul_mat(b_voice_ctx, cond->speaker_wavs_output_proj_weight,
        speaker_wavs_a);
    //
    //auto speaker_wavs_cond = ggml_new_tensor_2d(b_voice_ctx, GGML_TYPE_F32,
    //    speaker_wavs_b->ne[0], speaker_wavs_b->ne[1] * 5);
    // fill with learnt padding
    //speaker_wavs_cond = ggml_scale_inplace(b_voice_ctx, speaker_wavs_cond, 0);
    //speaker_wavs_cond = ggml_add_inplace(b_voice_ctx, speaker_wavs_cond,
    //    speaker_wavs_learnt_padding);
    auto speaker_wavs_cond = ggml_repeat_4d( b_voice_ctx,
        cond->speaker_wavs_learnt_padding,
        speaker_wavs_b->ne[0], speaker_wavs_b->ne[1] * 5, 1, 1 );
    // set first speaker
    auto speaker_0 = ggml_view_2d(b_voice_ctx, speaker_wavs_cond,
        speaker_wavs_b->ne[0], speaker_wavs_b->ne[1],
        speaker_wavs_cond->nb[1], 0);
    speaker_0 = ggml_cpy(b_voice_ctx, speaker_wavs_b, speaker_0);
    speaker_wavs_cond = ggml_view_2d(b_voice_ctx, speaker_0,
        speaker_wavs_cond->ne[0], speaker_wavs_cond->ne[1],
        speaker_wavs_cond->nb[1], 0);

    float cross_attention_pos_emb_scale = 1;
    //auto positions = ggml_arange(b_voice_ctx, 0, speaker_wavs_cond->ne[1], 1);
    auto positions = b_voice_ctx.arange(0, speaker_wavs_cond->ne[1], 1);
    auto pos_emb = ggml_timestep_embedding(b_voice_ctx, positions, 2048, 10000);
    auto condition_cross = ggml_add(b_voice_ctx, speaker_wavs_cond, ggml_scale(b_voice_ctx, pos_emb, cross_attention_pos_emb_scale));

    size_t mem_size = 2 * ggml_tensor_overhead();
    bool no_alloc = true;
    if (! backend ) {
        mem_size += ggml_nbytes( condition_sum );
        mem_size += ggml_nbytes( condition_cross );
        no_alloc = false;
    }
    voice->ctx = ggml_init({
        /*.mem_size   =*/ mem_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ no_alloc,
    });
    voice->sum = ggml_dup_tensor( voice->ctx, condition_sum );
    voice->cross = ggml_dup_tensor( voice->ctx, condition_cross );
    ggml_set_name( voice->sum, "sum" );
    ggml_set_name( voice->cross, "cross" );
    voice->buffer = ggml_backend_alloc_ctx_tensors( voice->ctx, backend );
    b_voice_ctx.build_forward_expand( condition_sum, voice->sum );
    b_voice_ctx.build_forward_expand( condition_cross, voice->cross );
    ONCE( b_voice_ctx.set_name("voice") );
    b_voice_ctx.compute();

    return true;
}

void get_prefix( 
        std::string audiopath,
        moshi_ttsmodel_t * tts,
        ggml_backend * backend
) {
    std::vector<short> data;
    auto sample_rate = load_wav( audiopath, data );
    printf( "sample_rate %d\n", sample_rate );
    std::vector<float> samples;
    if ( sample_rate == 24000 ) {
        assert( data.size() == 240000 ); // only support 10 second clips currently
        samples.resize( data.size() );
        for ( size_t s = 0; s < samples.size(); s++ )
            samples[s] = data[s] / 32768.f;
    } else if (sample_rate == 48000 ) {
        assert( data.size() == 480000 ); // only support 10 second clips currently
        samples.resize( data.size() / 2 );
        for ( size_t s = 0; s < samples.size(); s++ ) {
            int a = data[s*2];
            int b = data[s*2+1];
            samples[s] = (a + b) * 0.5f / 32768.f;
        }
    } else {
        assert( false ); // other sample rates not yet supported
    }

    const int frame_size = 1920;
    const int nframes = (int)samples.size() / frame_size;

    StateContext state_ctx( backend );
    auto states = moshi_mimi_encoder_states( &state_ctx, tts->mimi );
    // we don't need to pad because 24khz at 10 seconds fits perfectly
    //auto x = ctx.input( GGML_NE(samples.size()), samples );
    ggml_tensor * x = NULL;
    state_ctx.new_tensor( GGML_NE(samples.size()), GGML_TYPE_F32, &x );
    //auto x = ggml_new_tensor_1d( state_ctx.ctx, GGML_TYPE_F32, samples.size() );
    //int frame_size = 1920; // TODO: setup mimi.sample_rate / mimi.frame_rate
    //x = pad_for_conv1d( x, frame_size, frame_size );
    state_ctx.alloc();
    state_ctx.init();
    init( states );
    assert( x );
    ggml_backend_tensor_set( x, samples.data(), 0, ggml_nbytes( x ) );
    
    auto voice = &tts->voice;
    ScratchContext &ctx = *tts->scratch;
    const int n_q = tts->lm->n_q;
    const int max_delay = tts->lm->max_delay;
    const int delay_steps = tts->delay_steps;

    for ( int i = 0; i < nframes; i++ )
        voice->text_prefixes.push_back( -1 ); // zero_id

    for ( int i = 0; i < max_delay + delay_steps; i++ ) {
        voice->audio_prefixes.push_back({});
        auto & codes = voice->audio_prefixes.back();
        codes.resize( n_q );
        for ( int j = 0; j < n_q; j++ ) {
            codes[j] = lm_ungenerated_token_id;
        }
    }

    // encode one frame at a time for now
    const auto x_nb = 1920 * x->nb[0];
    for ( int i = 0; i < nframes; i++ ) {
        auto x_view = ggml_view_1d( ctx, x, 1920, i * x_nb );
        auto codes = mimi_encode( ctx, tts->mimi, states, x_view );
        auto cast = ggml_cast( ctx, codes, GGML_TYPE_I32 );
        voice->audio_prefixes.push_back({});
        std::vector<int> & audio_codes = voice->audio_prefixes.back();
        audio_codes.resize( ggml_nelements( cast ) );
        ctx.build_forward_expand( cast, audio_codes.data() );
        ctx.compute();
        // HACK: moving known delayed code from config
        auto nprefixes = voice->audio_prefixes.size();
        voice->audio_prefixes[nprefixes-3][0] = audio_codes[0];
        audio_codes[0] = lm_ungenerated_token_id;
    }
}




void moshi_ttsmodel_generate_wav(
        moshi_ttsmodel_t * tts,
        std::string text,
        std::string filename,
        ggml_backend * backend = NULL,
        int seed = -1) {
    assert( !tts->uses_cross || tts->voice.cross ); // need to load a voice, maybe automate this?

    auto time_start = ggml_time_ms();

    if (seed == -1)
        seed = (int)time(NULL);
    srand(seed);

    const int max_padding = 8;
    const int initial_padding = 2;
    const int second_stream_ahead = 2; // checkpoint_info.tts_config.get('second_stream_ahead', 0)
    auto machine = new StateMachine(tts->lm->text_card + 1, second_stream_ahead, max_padding, initial_padding);

    g_last_token_time = ggml_time_ms();
    std::deque<Entry> entries_;
    TokenIds token_ids;
    float frame_rate = 12.5f; // mimi.frame_rate
    std::vector<std::string> script_ = {text};
    bool multi_speaker = tts->uses_cross;
    int padding_between = 1;
    script_to_entries(
        entries_,
        tts->sp,
        token_ids,
        frame_rate,
        script_,
        multi_speaker,
        padding_between
    );
    auto machine_state = machine->new_state( entries_ );

    StateContext state_ctx( backend );
    auto lm_states = moshi_lmmodel_states( &state_ctx, tts->lm, tts->voice.cross );
    auto lmgen_state = moshi_lmgen_state( tts->lm );
    NE upsample_ne = {1, 512, 1, 1};
    NE decoder_ne = {2, 512, 1, 1};
    auto mimi_states = moshi_mimi_states( &state_ctx, tts->mimi, upsample_ne, decoder_ne );

    state_ctx.alloc();
    state_ctx.init();
    init( lm_states );
    init( mimi_states );

    ScratchContext ctx( 256, backend );
    std::vector<std::vector<float>> pcms2;
    std::vector<int> int_audio_tokens( tts->lm->num_audio_codebooks );
    const int final_padding = 4;
    const int delay_steps = tts->delay_steps;
    do {
        bool depformer_replace_tokens = (lmgen_state->offset < delay_steps);
        auto audio_tokens = moshi_lmgen_step(
            ctx,
            lmgen_state,
            tts->lm,
            lm_states,
            true, 0.6, 0.6, 250, 25,
            depformer_replace_tokens,
            machine, machine_state,
            tts->voice.sum, tts->voice.cross,
            tts->voice.text_prefixes, tts->voice.audio_prefixes,
            int_audio_tokens
        );
        if (audio_tokens) {
            std::vector<float> pcm2;
            mimi_decode(
                ctx,
                tts->mimi,
                mimi_states,
                int_audio_tokens,
                pcm2
            );
            pcms2.push_back(pcm2);
        }
    } while(lmgen_state->offset < (machine_state->end_step + delay_steps + final_padding) || machine_state->end_step == -1);

    auto generate_time_end = ggml_time_ms();

    std::vector<short> pcm2;
    for (auto pcm : pcms2) {
        for (auto value : pcm) {
            float v = value;
            v *= 32768;
            if (v > 32767.f) v = 32767.f;
            else if (v < -32768.f) v = -32768.f;
            pcm2.push_back((short)v);
        }
    }

    //int sample_rate = tts_model.attr("mimi").attr("sample_rate").cast<int>();
    int sample_rate = tts->mimi->sample_rate;
    printf( "sample_rate %d\n", sample_rate );
    save_wav( filename.c_str(), pcm2, sample_rate );

    auto save_time_end = ggml_time_ms();

    printf("generate time %f s\n", (generate_time_end - time_start) / 1000.f);
    printf("save     time %f s\n", (save_time_end - generate_time_end) / 1000.f);
    printf("seed was %0x\n", seed);
}



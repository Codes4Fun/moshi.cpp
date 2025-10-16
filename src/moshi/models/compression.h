#pragma once

struct moshi_mimi_t {
    bool initialized;
    moshi_split_rvq_t * quantizer;
    moshi_streaming_conv_transpose_1d_t * upsample;
    moshi_streaming_transformer_t * decoder_transformer;
    moshi_seanet_decoder_t * decoder;
    int sample_rate;
};

struct moshi_mimi_state_t {
    ggml_tensor * upsample;
    moshi_streaming_transformer_state_t * decoder_transformer;
    moshi_seanet_decoder_states_t * decoder;
};

moshi_mimi_state_t * moshi_mimi_states( StateContext * state_ctx,
        moshi_mimi_t * mimi, NE upsample_ne, NE decoder_ne ) {
    auto states = new moshi_mimi_state_t;
    moshi_streaming_conv_transpose_1d_state( state_ctx,
            mimi->upsample, upsample_ne, states->upsample );
    states->decoder_transformer = moshi_streaming_transformer_state( state_ctx,
            mimi->decoder_transformer, NULL );
    states->decoder = create_moshi_seanet_decoder_states( state_ctx,
            mimi->decoder, decoder_ne );
    return states;
}

void get_weights( WeightLoader * loader, moshi_mimi_t * mimi ) {
    get_weights( loader, "mimi.quantizer.", mimi->quantizer );
    get_weights( loader, "mimi.upsample.convtr.", mimi->upsample );
    get_weights( loader, "mimi.decoder_transformer.transformer.", mimi->decoder_transformer );
    get_weights( loader, "mimi.decoder.", mimi->decoder );
}

void init( moshi_mimi_state_t * states ) {
    init( states->decoder_transformer );
}

ggml_tensor * mimi_decode_latent(
        ScratchContext & ctx,
        moshi_split_rvq_t * quantizer,
        ggml_tensor * codes ) {
    auto emb = moshi_split_rvq_decode( ctx, quantizer, codes );
    return emb;
}

ggml_tensor * moshi_mimi_to_encoder_framerate(
        ggml_context * ctx,
        ggml_tensor * prev_y,
        moshi_streaming_conv_transpose_1d_t * upsample,
        ggml_tensor * x ) {
    auto y = moshi_streaming_conv_transpose_1d( ctx, prev_y, upsample, x );
    return y;
}

void mimi_decode(
        ScratchContext & ctx,
        moshi_mimi_t * mimi,
        moshi_mimi_state_t * states,
        std::vector<int> & int_codes,
        std::vector<float> & results ) {
    //ProfileScope profile(time_mimi_decode_us);

    // decode latent

    auto codes = ctx.input( GGML_NE(1, int_codes.size()), int_codes );
    auto emb = mimi_decode_latent( ctx, mimi->quantizer, codes );

    // to encoder framerate

    emb = moshi_mimi_to_encoder_framerate( ctx,
        states->upsample,
        mimi->upsample , emb );

    // decoder transformer

    emb = moshi_projected_transformer( ctx,
        states->decoder_transformer,
        mimi->decoder_transformer,
        emb );

    // decode

    auto out = moshi_seanet_decoder( ctx,
        states->decoder,
        mimi->decoder, emb );

    results.resize( ggml_nelements( out ) );
    ctx.build_forward_expand( out, results.data() );
    ON_NTH( 4, ctx.set_name( "decode_4" ) );
    ctx.compute();
}



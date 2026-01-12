#pragma once

struct moshi_mimi_t {
    bool initialized;
    own_ptr<moshi_split_rvq_t> quantizer;
    // decoder
    own_ptr<moshi_streaming_conv_transpose_1d_t> upsample;
    own_ptr<moshi_streaming_transformer_t> decoder_transformer;
    own_ptr<moshi_seanet_decoder_t> decoder;
    // encoder
    own_ptr<moshi_streaming_conv_1d_t> downsample;
    own_ptr<moshi_streaming_transformer_t> encoder_transformer;
    own_ptr<moshi_seanet_encoder_t> encoder;
    float frame_rate;
    int sample_rate;
};

struct moshi_mimi_state_t {
    // decoder
    ggml_tensor * upsample;
    own_ptr<moshi_streaming_transformer_state_t> decoder_transformer;
    own_ptr<moshi_seanet_decoder_states_t> decoder;
    // encoder
    ggml_tensor * downsample;
    own_ptr<moshi_streaming_transformer_state_t> encoder_transformer;
    own_ptr<moshi_seanet_encoder_states_t> encoder;
};

moshi_mimi_state_t * moshi_mimi_states( StateContext * state_ctx,
        moshi_mimi_t * mimi, NE upsample_ne, NE decoder_ne ) {
    auto states = new moshi_mimi_state_t;
    moshi_streaming_conv_transpose_1d_state( state_ctx,
            mimi->upsample, upsample_ne, states->upsample );
    states->decoder_transformer = moshi_streaming_transformer_state( state_ctx,
            mimi->decoder_transformer, NULL );
    /*if ( mimi->encoder ) { // we currently don't use the encoder for streaming
        moshi_streaming_conv_1d_state( state_ctx,
            mimi->downsample, states->downsample );
        states->encoder_transformer = moshi_streaming_transformer_state( state_ctx,
                mimi->encoder_transformer, NULL );
        states->encoder = create_moshi_seanet_encoder_states( state_ctx,
            mimi->encoder );
    } else*/ {
        states->downsample = NULL;
    }
    states->decoder = create_moshi_seanet_decoder_states( state_ctx,
            mimi->decoder, decoder_ne );
    return states;
}

moshi_mimi_state_t * moshi_mimi_encoder_states( StateContext * state_ctx,
        moshi_mimi_t * mimi ) {
    auto states = new moshi_mimi_state_t;
    states->upsample = NULL;
    moshi_streaming_conv_1d_state( state_ctx,
        mimi->downsample, states->downsample );
    states->encoder_transformer = moshi_streaming_transformer_state( state_ctx,
            mimi->encoder_transformer, NULL );
    states->encoder = create_moshi_seanet_encoder_states( state_ctx,
        mimi->encoder );
    return states;
}

/*
void get_weights( WeightLoader * loader, moshi_mimi_t * mimi ) {
    get_weights( loader, "mimi.quantizer.", mimi->quantizer );
    get_weights( loader, "mimi.upsample.convtr.", mimi->upsample );
    get_weights( loader, "mimi.decoder_transformer.transformer.", mimi->decoder_transformer );
    get_weights( loader, "mimi.decoder.", mimi->decoder );
    if ( mimi->encoder ) {
        //get_weights( loader, "mimi.encoder_transformer.transformer.", mimi->encoder_transformer );
        get_weights( loader, "mimi.encoder.", mimi->encoder );
    }
}
*/

void init( ScratchContext * ctx, moshi_mimi_state_t * states, moshi_mimi_t * mimi ) {
    if ( states->decoder_transformer )
        init( ctx, states->decoder_transformer, mimi->decoder_transformer , NULL );
    if ( states->encoder_transformer )
        init( ctx, states->encoder_transformer, mimi->encoder_transformer , NULL );
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


ggml_tensor * moshi_mimi_to_framerate(
        ScratchContext & ctx,
        ggml_tensor * prev_y,
        moshi_streaming_conv_1d_t * downsample,
        ggml_tensor * x ) {
    auto y = moshi_streaming_conv_1d( ctx, prev_y, downsample, x );
    return y;
}

ggml_tensor * mimi_quantizer_encode(
        ScratchContext & ctx,
        moshi_split_rvq_t * quantizer,
        ggml_tensor * emb ) {
    auto codes = moshi_split_rvq_encode( ctx, quantizer, emb );
    return codes;
}

ggml_tensor * mimi_encode(
        ScratchContext & ctx,
        moshi_mimi_t * mimi,
        moshi_mimi_state_t * states,
        ggml_tensor * x ) {

    auto emb = moshi_seanet_encoder( ctx, states->encoder, mimi->encoder, x );

    //std::vector<uint8_t> dst( ggml_nbytes( emb ) );
    //ctx.build_forward_expand( emb, dst.data() );
    //ctx.compute();

    emb = moshi_projected_transformer( ctx,
        states->encoder_transformer, mimi->encoder_transformer, emb );

    // TODO: add support for replicate
    emb = moshi_mimi_to_framerate( ctx,
        states->downsample,
        mimi->downsample , emb );
    
    auto codes = mimi_quantizer_encode( ctx, mimi->quantizer, emb );

    return codes;
}

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

struct moshi_mimi_graph_t {
    own_ptr<GraphContext> ctx;
    ggml_tensor * codes;
    ggml_tensor * frame;
    int T;
};

struct moshi_mimi_state_t {
    // decoder
    ggml_tensor * upsample;
    own_ptr<moshi_streaming_transformer_state_t> decoder_transformer;
    own_ptr<moshi_seanet_decoder_states_t> decoder;
    moshi_mimi_graph_t decoder_graph;
    // encoder
    ggml_tensor * downsample;
    own_ptr<moshi_streaming_transformer_state_t> encoder_transformer;
    own_ptr<moshi_seanet_encoder_states_t> encoder;
    moshi_mimi_graph_t encoder_graph;
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
        ggml_context * ctx,
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

#ifdef USE_SCRATCH
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
#else
void mimi_decode(
        ScratchContext & ctx,
        moshi_mimi_t * mimi,
        moshi_mimi_state_t * states,
        std::vector<int> & int_codes,
        std::vector<float> & results ) {
    //ProfileScope profile(time_mimi_decode_us);

    if ( ! states->decoder_graph.ctx ) {
        states->decoder_graph.ctx = new GraphContext( 256, ctx.backend );
        GraphContext & gctx = *states->decoder_graph.ctx;

        auto codes = ggml_new_tensor_2d( gctx, GGML_TYPE_I32, 1, int_codes.size() );
        states->decoder_graph.codes = codes;

        // decode latent
        auto emb = mimi_decode_latent( gctx, mimi->quantizer, codes );

        // to encoder framerate
        emb = moshi_mimi_to_encoder_framerate( gctx,
            states->upsample,
            mimi->upsample , emb );

        // decoder transformer
        states->decoder_graph.T = (int) emb->ne[1];
        emb = moshi_projected_transformer_graph_build( gctx,
            states->decoder_transformer,
            mimi->decoder_transformer,
            emb );

        // decode
        auto frame = moshi_seanet_decoder( gctx,
            states->decoder,
            mimi->decoder, emb );
        states->decoder_graph.frame = frame;

        states->decoder_graph.ctx->build_forward_expand( frame );
        states->decoder_graph.ctx->alloc();
    }

    assert( ggml_nelements( states->decoder_graph.codes ) == int_codes.size() );
    ggml_backend_tensor_set( states->decoder_graph.codes, int_codes.data(),
        0, ggml_nbytes( states->decoder_graph.codes ) );

    moshi_projected_transformer_graph_step( ctx, 
        states->decoder_transformer,
        mimi->decoder_transformer,
        states->decoder_graph.T );

    ctx.compute();
    states->decoder_graph.ctx->compute();

    results.resize( ggml_nelements( states->decoder_graph.frame ) );
    ggml_backend_tensor_get( states->decoder_graph.frame,
        results.data(), 0, results.size() * sizeof(results[0]) );
}
#endif

ggml_tensor * moshi_mimi_to_framerate(
        GraphContext & ctx,
        ggml_tensor * prev_y,
        moshi_streaming_conv_1d_t * downsample,
        ggml_tensor * x ) {
    auto y = moshi_streaming_conv_1d( ctx, prev_y, downsample, x );
    return y;
}

ggml_tensor * mimi_quantizer_encode(
        GraphContext & ctx,
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

#ifdef USE_SCRATCH
void mimi_encode(
        ScratchContext & ctx,
        moshi_mimi_t * mimi,
        moshi_mimi_state_t * states,
        std::vector<float> & frame,
        std::vector<int> & int_codes ) {

    auto x = ctx.input( GGML_NE(frame.size()), frame );

    auto emb = moshi_seanet_encoder( ctx, states->encoder, mimi->encoder, x );

    emb = moshi_projected_transformer( ctx,
        states->encoder_transformer, mimi->encoder_transformer, emb );

    emb = moshi_mimi_to_framerate( ctx,
        states->downsample,
        mimi->downsample , emb );
    
    auto codes = mimi_quantizer_encode( ctx, mimi->quantizer, emb );

    codes = ggml_cast( ctx, codes, GGML_TYPE_I32 );

    int_codes.resize( ggml_nelements( codes ) );
    ctx.build_forward_expand( codes, int_codes.data() );
    ctx.compute();
}
#else
void mimi_encode(
        ScratchContext & ctx,
        moshi_mimi_t * mimi,
        moshi_mimi_state_t * states,
        std::vector<float> & frame,
        std::vector<int> & int_codes ) {

    if ( ! states->encoder_graph.ctx ) {
        states->encoder_graph.ctx = new GraphContext( 256, ctx.backend );
        GraphContext & gctx = *states->encoder_graph.ctx;

        auto x = ggml_new_tensor_1d( gctx, GGML_TYPE_F32, frame.size() );
        states->encoder_graph.frame = x;

        auto emb = moshi_seanet_encoder( gctx, states->encoder, mimi->encoder, x );

        states->encoder_graph.T = emb->ne[1];
        emb = moshi_projected_transformer_graph_build( gctx,
            states->encoder_transformer, mimi->encoder_transformer, emb );

        emb = moshi_mimi_to_framerate( gctx,
            states->downsample,
            mimi->downsample , emb );
        
        auto codes = mimi_quantizer_encode( gctx, mimi->quantizer, emb );

        codes = ggml_cast( gctx, codes, GGML_TYPE_I32 );
        states->encoder_graph.codes = codes;

        states->encoder_graph.ctx->build_forward_expand( codes );
        states->encoder_graph.ctx->alloc();
    }

    assert( ggml_nelements( states->encoder_graph.frame ) == frame.size() );
    ggml_backend_tensor_set( states->encoder_graph.frame, frame.data(),
        0, ggml_nbytes( states->encoder_graph.frame ) );

    moshi_projected_transformer_graph_step( ctx, 
        states->encoder_transformer,
        mimi->encoder_transformer,
        states->encoder_graph.T );

    ctx.compute();
    states->encoder_graph.ctx->compute();

    int_codes.resize( ggml_nelements( states->encoder_graph.codes ) );
    ggml_backend_tensor_get( states->encoder_graph.codes,
        int_codes.data(), 0, int_codes.size() * sizeof(int_codes[0]) );
}
#endif

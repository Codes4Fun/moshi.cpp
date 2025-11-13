#pragma once

// for src/context.h
#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#ifdef ENABLE_REPLAY
#include "replay.h"
#define CAPTURE(...)
#define CAPTURE_GROUP(...)
#else
#ifdef ENABLE_CAPTURE
#include "src/ggml_cap.h"
#else
#define CAPTURE(...)
#define CAPTURE_GROUP(...)
#endif
#endif
#define ONCE(code) {static bool once=false; if (!once) {{code;}; once=true;}}
#define ON_NTH(nth, code) {static int count=0; if (count++ == (nth)) {code;}}

#include "../src/ptrs.h"
#include "../src/safetensor.h"
#include "../src/config.h"
#include "../src/context.h"
#include "../src/wav.h"
#include "../src/loader.h"
#include "../src/torch.h"
#include "../src/moshi/modules/transformer.h"
#include "../src/moshi/utils/sampling.h"
#include "../src/moshi/models/lm_utils.h"
#include "../src/moshi/models/lm.h"
#include "../src/moshi/quantization/core_vq.h"
#include "../src/moshi/quantization/vq.h"
#include "../src/moshi/modules/conv.h"
#include "../src/moshi/modules/seanet.h"
#include "../src/moshi/models/compression.h"
#include "../src/moshi/models/lm_default.h"
#include "../src/moshi/models/tts.h"

struct moshi_context_t {
    ggml_backend * backend;
    own_ptr<ScratchContext> scratch_cpu;
    own_ptr<ScratchContext> scratch;
};

void moshi_alloc( moshi_context_t * moshi, ggml_backend * backend ) {
    assert( backend );
    moshi->backend = backend;
    moshi->scratch_cpu = new ScratchContext( 256 );
    moshi->scratch = new ScratchContext( 256, backend );
}

void moshi_alloc( moshi_context_t * moshi, const char * device = NULL ) {
    ggml_backend * backend;
    if ( device ) {
        backend = ggml_backend_init_by_name( device, NULL );
    } else {
        backend = ggml_backend_init_best();
    }
    if ( ! backend ) {
        fprintf( stderr, "failed to initialize device\n" );
        exit(1);
    }
    auto dev = ggml_backend_get_device( backend );
    auto dev_name = ggml_backend_dev_name( dev );
    printf( "using device: \"%s\"\n", dev_name );
    moshi_alloc( moshi, backend );
}

struct mimi_codec_t {
    int n_q;
    moshi_context_t * moshi;
    own_ptr<moshi_mimi_t> mimi;
    own_ptr<WeightLoader> mimi_weights;
};

void mimi_alloc( mimi_codec_t * codec, moshi_context_t * moshi, const char * filename, int n_q ) {
    auto mimi = moshi_mimi_alloc_default( n_q );
    auto mimi_weights = WeightLoader::from_safetensor( filename,
        moshi->scratch_cpu, moshi->backend );
    if ( ! mimi_weights ) {
        fprintf(stderr, "error: mimi weights not found\n" );
        exit(1);
    }
    get_weights( mimi_weights, "mimi.quantizer.", mimi->quantizer );
    get_weights( mimi_weights, "mimi.upsample.convtr.", mimi->upsample );
    get_weights( mimi_weights, "mimi.decoder_transformer.transformer.", mimi->decoder_transformer );
    get_weights( mimi_weights, "mimi.decoder.", mimi->decoder );
    if ( mimi->encoder ) {
        get_weights( mimi_weights, "mimi.downsample.conv.", mimi->downsample );
        get_weights( mimi_weights, "mimi.encoder_transformer.transformer.", mimi->encoder_transformer );
        get_weights( mimi_weights, "mimi.encoder.", mimi->encoder );
    }
    mimi_weights->load();

    codec->n_q = n_q;
    codec->moshi = moshi;
    codec->mimi = mimi;
    codec->mimi_weights = mimi_weights;
}

int mimi_frame_size( mimi_codec_t * mimi ) {
    return 1920;
}

struct mimi_encode_context_t {
    mimi_codec_t * codec;
    own_ptr<StateContext> state_ctx;
    own_ptr<moshi_mimi_state_t> states;
    ggml_tensor * device_frame;
    std::vector<int> tokens;
};

void mimi_encode_alloc_context( mimi_encode_context_t * context, mimi_codec_t * codec ) {
    auto state_ctx = new StateContext( codec->moshi->backend );
    auto mimi_states = moshi_mimi_encoder_states( state_ctx, codec->mimi );
    int frame_size = mimi_frame_size( codec );
    ggml_tensor * device_frame = NULL;
    state_ctx->new_tensor( GGML_NE(frame_size), GGML_TYPE_F32, &device_frame );

    state_ctx->alloc();
    state_ctx->init();
    init( mimi_states );

    context->codec = codec;
    context->state_ctx = state_ctx;
    context->states = mimi_states;
    context->device_frame = device_frame;
}

void mimi_encode_send( mimi_encode_context_t * context, float * frame ) {
    auto device_frame = context->device_frame;
    ggml_backend_tensor_set( device_frame, frame, 0, ggml_nbytes( device_frame ) );
}

void mimi_encode_receive( mimi_encode_context_t * context, int16_t * tokens ) {
    auto & ctx = *context->codec->moshi->scratch;
    auto mimi = context->codec->mimi.ptr;
    auto states = context->states.ptr;
    auto codes = mimi_encode( ctx, mimi, states, context->device_frame );
    auto cast = ggml_cast( ctx, codes, GGML_TYPE_I32 );
    context->tokens.resize( ggml_nelements( cast ) );
    ctx.build_forward_expand( cast, context->tokens.data() );
    ctx.compute();
    for ( int i = 0; i < context->tokens.size(); i++ )
        tokens[i] = context->tokens[i];
}

struct mimi_decode_context_t {
    mimi_codec_t * codec;
    own_ptr<StateContext> state_ctx;
    own_ptr<moshi_mimi_state_t> states;
    ggml_tensor * device_frame;
    std::vector<int> tokens;
    std::vector<float> frame;
};

void mimi_decode_alloc_context( mimi_decode_context_t * context, mimi_codec_t * codec ) {
    auto state_ctx = new StateContext( codec->moshi->backend );
    NE upsample_ne = {1, 512, 1, 1};
    NE decoder_ne = {2, 512, 1, 1};
    auto mimi_states = moshi_mimi_states( state_ctx, codec->mimi, upsample_ne, decoder_ne );
    int frame_size = mimi_frame_size( codec );
    ggml_tensor * device_frame = NULL;
    state_ctx->new_tensor( GGML_NE(frame_size), GGML_TYPE_F32, &device_frame );

    state_ctx->alloc();
    state_ctx->init();
    init( mimi_states );

    context->codec = codec;
    context->state_ctx = state_ctx;
    context->states = mimi_states;
    context->device_frame = device_frame;
    context->tokens.resize( codec->n_q );
    context->frame.resize( frame_size );
}

void mimi_decode_send( mimi_decode_context_t * context, int16_t * tokens ) {
    auto n_q = context->codec->n_q;
    for ( int i = 0; i < n_q; i++ )
        context->tokens[i] = tokens[i];
}

void mimi_decode_receive( mimi_decode_context_t * context, float * frame ) {
    auto & ctx = *context->codec->moshi->scratch;
    auto mimi = context->codec->mimi.ptr;
    auto states = context->states.ptr;
    mimi_decode(
        ctx,
        mimi,
        states,
        context->tokens,
        context->frame
    );
    memcpy( frame, context->frame.data(), context->frame.size() * 4 );
}

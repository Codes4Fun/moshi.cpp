#pragma once

/*************************************************************\
 *  moshi.models.lm_utils.ScaledEmbedding
 *
 * notes:
 * this is split between two separate functions, one that
 * demuxes input and one that does not.
 * because of integer logic, math, and use of modulus, they 
 * have to be done on the cpu, but can be the start of a 
 * graph
\*************************************************************/

struct moshi_scaled_embedding_demux_t {
    int num_embeddings;
    own_ptr<torch_nn_linear_t> out1;
    own_ptr<torch_nn_linear_t> out2;
    ggml_tensor * weight;
};

void get_weights( WeightLoader * loader, std::string path,
        moshi_scaled_embedding_demux_t * m ) {
    if ( loader->quantize ) {
        if ( loader->qtype == GGML_TYPE_Q4_K ) {
            auto n = loader->fetch( &m->weight, path + "weight", GGML_TYPE_Q4_0 );
            assert( n );
        } else if ( loader->qtype == GGML_TYPE_Q8_K ) {
            auto n = loader->fetch( &m->weight, path + "weight", GGML_TYPE_Q8_0 );
            assert( n );
        } else {
            auto n = loader->fetch( &m->weight, path + "weight", loader->qtype );
            assert( n );
        }
    } else {
        auto n = loader->fetch( &m->weight, path + "weight", (void*)ggml_get_rows );
        assert( n );
    }
    get_weights( loader, path + "out1.", m->out1 );
    get_weights( loader, path + "out2.", m->out2 );
}

struct embedding_demux_t {
    ggml_tensor * left;
    ggml_tensor * right;
    ggml_tensor * right_scale;
};

ggml_tensor * moshi_scaled_embedding_demux_build(
        GraphContext & ctx,
        moshi_scaled_embedding_demux_t * m,
        embedding_demux_t * emb ) {
    auto left = ctx.new_tensor( GGML_TYPE_I32, GGML_NE( 1 ) );
    auto right = ctx.new_tensor( GGML_TYPE_I32, GGML_NE( 1 ) );
    auto right_scale = ctx.new_tensor( GGML_TYPE_F32, GGML_NE( 1 ) );
    emb->left = left;
    emb->right = right;
    emb->right_scale = right_scale;
    left = ggml_get_rows( ctx, m->weight, left );
    right = ggml_get_rows( ctx, m->weight, right );
    auto right_y = torch_nn_linear( ctx, m->out2, right );
    auto left_y = torch_nn_linear( ctx, m->out1, left );
    right_y = ggml_mul( ctx, right_y, right_scale );
    auto y = ggml_add( ctx, left_y, right_y );
    return y;
}

void moshi_scaled_embedding_demux_step(
        GraphContext & ctx,
        moshi_scaled_embedding_demux_t * m,
        embedding_demux_t * emb,
        int input ) {
    if ( input < 0 )
        input = 0;
    auto left_idx = input % m->num_embeddings;

    auto right_idx = input / m->num_embeddings;
    right_idx = right_idx - 1;
    bool right_zero = right_idx < 0;
    if ( right_idx < 0 )
        right_idx = 0;

    ctx.tensor_set( emb->left, left_idx );
    ctx.tensor_set( emb->right, right_idx );
    ctx.tensor_set( emb->right_scale, right_zero? 0.f : 1.f );
}

ggml_tensor * moshi_scaled_embedding_demux(
        ScratchContext & ctx,
        moshi_scaled_embedding_demux_t * m,
        ggml_tensor * left, ggml_tensor * right, ggml_tensor * right_scale ) {

    left = ggml_get_rows( ctx, m->weight, left );
    right = ggml_get_rows( ctx, m->weight, right );
    auto right_y = torch_nn_linear( ctx, m->out2, right );
    auto left_y = torch_nn_linear( ctx, m->out1, left );
    right_y = ggml_mul( ctx, right_y, right_scale );
    auto y = ggml_add( ctx, left_y, right_y );
    return y;
}

ggml_tensor * moshi_scaled_embedding_demux(
        ScratchContext & ctx,
        moshi_scaled_embedding_demux_t * m,
        int input ) {

    if ( input < 0 )
        input = 0;
    auto left_idx = input % m->num_embeddings;
    auto right_idx = input / m->num_embeddings;
    right_idx = right_idx - 1;

    auto left = ctx.constant( left_idx );

    bool right_zero = right_idx < 0;

    if ( right_idx < 0 )
        right_idx = 0;

    auto right = ctx.constant( right_idx );

    auto right_scale = ctx.constant( right_zero? 0.f : 1.f );

    return moshi_scaled_embedding_demux( ctx, m, left, right, right_scale );
}

struct moshi_scaled_embedding_t {
    own_ptr<torch_nn_linear_t> low_rank;
    ggml_tensor * weight;
};

void get_weights( WeightLoader * loader, std::string path,
        moshi_scaled_embedding_t * m ) {
    if ( loader->quantize ) {
        if ( loader->qtype == GGML_TYPE_Q4_K ) {
            auto n = loader->fetch( &m->weight, path + "weight", GGML_TYPE_Q4_0 );
            assert( n );
        } else if ( loader->qtype == GGML_TYPE_Q8_K ) {
            auto n = loader->fetch( &m->weight, path + "weight", GGML_TYPE_Q8_0 );
            assert( n );
        } else {
            auto n = loader->fetch( &m->weight, path + "weight", loader->qtype );
            assert( n );
        }
    } else {
        auto n = loader->fetch( &m->weight, path + "weight", (void*)ggml_get_rows );
        assert( n );
    }
    if ( m->low_rank )
        get_weights( loader, path + "low_rank.", m->low_rank );
}

struct embedding_t {
    ggml_tensor * input;
    ggml_tensor * scale;
};

ggml_tensor * moshi_scaled_embedding_build(
        GraphContext & ctx,
        moshi_scaled_embedding_t * m,
        embedding_t * emb ) {
    auto input = ctx.new_tensor( GGML_TYPE_I32, GGML_NE( 1 ) );
    auto scale = ctx.new_tensor( GGML_TYPE_F32, GGML_NE( 1 ) );
    emb->input = input;
    emb->scale = scale;
    auto y = ggml_get_rows( ctx, m->weight, input );
    y = ggml_mul(ctx, y, scale );
    if ( m->low_rank )
        y = torch_nn_linear( ctx, m->low_rank, y );
    return y;
}

void moshi_scaled_embedding_step(
        GraphContext & ctx,
        moshi_scaled_embedding_t * m,
        embedding_t * emb,
        int input ) {
    bool is_zero = input == -1;
    if ( input < 0 )
        input = 0;
    ctx.tensor_set( emb->input, input );
    ctx.tensor_set( emb->scale, is_zero? 0.f : 1.f );
}

ggml_tensor * moshi_scaled_embedding(
        ScratchContext & ctx,
        moshi_scaled_embedding_t * m,
        ggml_tensor * input, ggml_tensor * scale ) {
    auto y = ggml_get_rows( ctx, m->weight, input );
    y = ggml_mul(ctx, y, scale );
    if ( m->low_rank )
        y = torch_nn_linear( ctx, m->low_rank, y );
    return y;
}

ggml_tensor * moshi_scaled_embedding(
        ScratchContext & ctx,
        moshi_scaled_embedding_t * m,
        int input ) {
    bool is_zero = input == -1;
    if ( input < 0 )
        input = 0;
    return moshi_scaled_embedding( ctx, m,
        ctx.constant( input), 
        ctx.constant( is_zero? 0.f : 1.f ) );
}

// this should only be used after the first moshi_scaled_embedding
// where the input value is guaranteed to not be negative
ggml_tensor * moshi_scaled_embedding_chained(
        GraphContext & ctx,
        moshi_scaled_embedding_t * m,
        ggml_tensor * input ) {
    auto y = ggml_get_rows( ctx, m->weight, input );
    if ( m->low_rank )
        y = torch_nn_linear( ctx, m->low_rank, y );
    return y;
}



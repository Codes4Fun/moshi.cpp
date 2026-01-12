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
    left = ggml_get_rows( ctx, m->weight, left );

    bool right_zero = right_idx < 0;

    if ( right_idx < 0 )
        right_idx = 0;

    auto right = ctx.constant( right_idx );
    right = ggml_get_rows( ctx, m->weight, right );

    auto right_y = torch_nn_linear( ctx, m->out2, right );

    auto left_y = torch_nn_linear( ctx, m->out1, left );

    if ( right_zero )
        right_y = ggml_scale( ctx, right_y, 0 );

    auto y = ggml_add( ctx, left_y, right_y );

    return y;
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

ggml_tensor * moshi_scaled_embedding(
        ScratchContext & ctx,
        moshi_scaled_embedding_t * m,
        int input ) {
    bool is_zero = input == -1;
    if ( input < 0 )
        input = 0;
    auto y = ggml_get_rows( ctx, m->weight, ctx.constant( input ) );
    if ( is_zero )
        y = ggml_scale(ctx, y, 0);
    if ( m->low_rank )
        y = torch_nn_linear( ctx, m->low_rank, y );
    return y;
}

// this should only be used after the first moshi_scaled_embedding
// where the input value is guaranteed to not be negative
ggml_tensor * moshi_scaled_embedding_chained(
        ScratchContext & ctx,
        moshi_scaled_embedding_t * m,
        ggml_tensor * input ) {
    auto y = ggml_get_rows( ctx, m->weight, input );
    if ( m->low_rank )
        y = torch_nn_linear( ctx, m->low_rank, y );
    return y;
}



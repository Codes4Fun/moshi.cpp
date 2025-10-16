#pragma once

/*****************************************************************************\
 *   moshi.modules.seanet.SEANetResnetBlock
 * custom version of StreamingConv1d that did not use it's state variables.
 * located in models:
 *   mimi.decoder.models.3,6,9,12
\*****************************************************************************/

struct moshi_seanet_resnet_block_t {
    moshi_streaming_conv_1d_t * block_1;
    moshi_stateless_conv_1d_t * block_3;
};

ggml_tensor * moshi_seanet_resnet_block(
        ggml_context * ctx,
        ggml_tensor * prev,
        moshi_seanet_resnet_block_t * resnet,
        ggml_tensor * x ) {
    auto u = x;
    auto v = ggml_elu( ctx, x );

    v = moshi_streaming_conv_1d( ctx, prev, resnet->block_1, v );
    v = ggml_elu( ctx, v );
    v = moshi_stateless_conv_1d( ctx, resnet->block_3, v );
    auto y = ggml_add( ctx, u, v );
    return y;
}

bool calc_out_dim( const moshi_seanet_resnet_block_t * resnet,
        const NE x_ne, NE &y_ne ) {
    calc_out_dim( resnet->block_1, x_ne, y_ne );
    calc_out_dim( resnet->block_3, y_ne, y_ne );
    return true;
}

void get_weights( WeightLoader * loader, std::string path,
        moshi_seanet_resnet_block_t * resnet ) {
    get_weights( loader, path + "block.1." ,resnet->block_1 );
    get_weights( loader, path + "block.3." ,resnet->block_3 );
}

void moshi_seanet_resnet_block_state( StateContext * state_ctx,
        moshi_seanet_resnet_block_t * resnet, const NE x_ne,
        ggml_tensor * &prev ) {
    moshi_streaming_conv_1d_state( state_ctx, resnet->block_1, x_ne, prev );
}

/*****************************************************************************\
 *   moshi.modules.seanet.SEANetDecoder
 * custom version of StreamingConv1d that did not use it's state variables.
 * located in models:
 *   mimi.decoder
\*****************************************************************************/

struct moshi_seanet_decoder_t {
    moshi_streaming_conv_1d_t * model_0;
    moshi_streaming_conv_transpose_1d_t * model_2;
    moshi_seanet_resnet_block_t * model_3;
    moshi_streaming_conv_transpose_1d_t * model_5;
    moshi_seanet_resnet_block_t * model_6;
    moshi_streaming_conv_transpose_1d_t * model_8;
    moshi_seanet_resnet_block_t * model_9;
    moshi_streaming_conv_transpose_1d_t * model_11;
    moshi_seanet_resnet_block_t * model_12;
    moshi_streaming_conv_1d_t * model_14;
};

struct moshi_seanet_decoder_states_t {
    ggml_tensor * model_0;
    ggml_tensor * model_2;
    ggml_tensor * model_3;
    ggml_tensor * model_5;
    ggml_tensor * model_6;
    ggml_tensor * model_8;
    ggml_tensor * model_9;
    ggml_tensor * model_11;
    ggml_tensor * model_12;
    ggml_tensor * model_14;
};

ggml_tensor * moshi_seanet_decoder(
        ggml_context * ctx,
        moshi_seanet_decoder_states_t * states,
        moshi_seanet_decoder_t * decoder,
        ggml_tensor * x) {

    x = moshi_streaming_conv_1d( ctx, states->model_0, decoder->model_0, x );
    x = ggml_elu( ctx, x );

    x = moshi_streaming_conv_transpose_1d( ctx, states->model_2, decoder->model_2, x );
    x = moshi_seanet_resnet_block( ctx, states->model_3, decoder->model_3, x );
    x = ggml_elu( ctx, x );

    x = moshi_streaming_conv_transpose_1d( ctx, states->model_5, decoder->model_5, x );
    x = moshi_seanet_resnet_block( ctx, states->model_6, decoder->model_6, x );
    x = ggml_elu( ctx, x );

    x = moshi_streaming_conv_transpose_1d( ctx, states->model_8, decoder->model_8, x );
    x = moshi_seanet_resnet_block( ctx, states->model_9, decoder->model_9, x );
    x = ggml_elu( ctx, x );

    x = moshi_streaming_conv_transpose_1d( ctx, states->model_11, decoder->model_11, x );
    x = moshi_seanet_resnet_block( ctx, states->model_12, decoder->model_12, x );
    x = ggml_elu( ctx, x );

    x = moshi_streaming_conv_1d( ctx, states->model_14, decoder->model_14, x );

    return x;
}

void get_weights( WeightLoader * loader, std::string path, moshi_seanet_decoder_t * decoder ) {
    get_weights( loader, path + "model.0.", decoder->model_0 );
    get_weights( loader, path + "model.2.", decoder->model_2 );
    get_weights( loader, path + "model.3.", decoder->model_3 );
    get_weights( loader, path + "model.5.", decoder->model_5 );
    get_weights( loader, path + "model.6.", decoder->model_6 );
    get_weights( loader, path + "model.8.", decoder->model_8 );
    get_weights( loader, path + "model.9.", decoder->model_9 );
    get_weights( loader, path + "model.11.", decoder->model_11 );
    get_weights( loader, path + "model.12.", decoder->model_12 );
    get_weights( loader, path + "model.14.", decoder->model_14 );
}

moshi_seanet_decoder_states_t * create_moshi_seanet_decoder_states(
    StateContext * state_ctx,
    moshi_seanet_decoder_t * decoder,
    const NE x_ne )
{
    auto states = new moshi_seanet_decoder_states_t;

    NE out_ne;
    moshi_streaming_conv_1d_state( state_ctx, decoder->model_0, x_ne, states->model_0 );
    calc_out_dim( decoder->model_0, x_ne, out_ne );

    moshi_streaming_conv_transpose_1d_state( state_ctx, decoder->model_2, out_ne, states->model_2 );
    calc_out_dim( decoder->model_2, out_ne, out_ne );
    moshi_seanet_resnet_block_state( state_ctx, decoder->model_3, out_ne, states->model_3 );
    calc_out_dim( decoder->model_3, out_ne, out_ne );

    moshi_streaming_conv_transpose_1d_state( state_ctx, decoder->model_5, out_ne, states->model_5 );
    calc_out_dim( decoder->model_5, out_ne, out_ne );
    moshi_seanet_resnet_block_state( state_ctx, decoder->model_6, out_ne, states->model_6 );
    calc_out_dim( decoder->model_6, out_ne, out_ne );

    moshi_streaming_conv_transpose_1d_state( state_ctx, decoder->model_8, out_ne, states->model_8 );
    calc_out_dim( decoder->model_8, out_ne, out_ne );
    moshi_seanet_resnet_block_state( state_ctx, decoder->model_9, out_ne, states->model_9 );
    calc_out_dim( decoder->model_9, out_ne, out_ne );

    moshi_streaming_conv_transpose_1d_state( state_ctx, decoder->model_11, out_ne, states->model_11 );
    calc_out_dim( decoder->model_11, out_ne, out_ne );
    moshi_seanet_resnet_block_state( state_ctx, decoder->model_12, out_ne, states->model_12 );
    calc_out_dim( decoder->model_12, out_ne, out_ne );

    moshi_streaming_conv_1d_state( state_ctx, decoder->model_14, out_ne, states->model_14 );
    return states;
}



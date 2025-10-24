#pragma once

/*****************************************************************************\
 *   moshi.modules.seanet.SEANetResnetBlock
 * located in models:
 *   mimi.decoder.models.3,6,9,12
\*****************************************************************************/

struct moshi_seanet_resnet_block_t {
    own_ptr<moshi_streaming_conv_1d_t> block_1;
    own_ptr<moshi_stateless_conv_1d_t> block_3;
};

ggml_tensor * moshi_seanet_resnet_block(
        ScratchContext & ctx,
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
        moshi_seanet_resnet_block_t * resnet,
        ggml_tensor * &prev ) {
    moshi_streaming_conv_1d_state( state_ctx, resnet->block_1, prev );
}

/*****************************************************************************\
 *   moshi.modules.seanet.SEANetEncoder
 * located in models:
 *   mimi.encoder
\*****************************************************************************/

struct moshi_seanet_encoder_t {
    own_ptr<moshi_streaming_conv_1d_t> model_0;
    own_ptr<moshi_seanet_resnet_block_t> model_1;
    // elu
    own_ptr<moshi_streaming_conv_1d_t> model_3;
    own_ptr<moshi_seanet_resnet_block_t> model_4;
    // elu
    own_ptr<moshi_streaming_conv_1d_t> model_6;
    own_ptr<moshi_seanet_resnet_block_t> model_7;
    // elu
    own_ptr<moshi_streaming_conv_1d_t> model_9;
    own_ptr<moshi_seanet_resnet_block_t> model_10;
    // elu
    own_ptr<moshi_streaming_conv_1d_t> model_12;
    // elu
    own_ptr<moshi_streaming_conv_1d_t> model_14;
};

struct moshi_seanet_encoder_states_t {
    ggml_tensor * model_0;
    ggml_tensor * model_1;
    // elu
    ggml_tensor * model_3;
    ggml_tensor * model_4;
    // elu
    ggml_tensor * model_6;
    ggml_tensor * model_7;
    // elu
    ggml_tensor * model_9;
    ggml_tensor * model_10;
    // elu
    ggml_tensor * model_12;
    // elu
    ggml_tensor * model_14;
};

ggml_tensor * moshi_seanet_encoder(
        ScratchContext & ctx,
        moshi_seanet_encoder_states_t * states,
        moshi_seanet_encoder_t * encoder,
        ggml_tensor * x) {
    static moshi_seanet_encoder_states_t null_states = {NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL};
    if ( ! states ) states = &null_states;
    x = moshi_streaming_conv_1d( ctx, states->model_0, encoder->model_0, x );
    x = moshi_seanet_resnet_block( ctx, states->model_1, encoder->model_1, x );
    x = ggml_elu( ctx, x );
    x = moshi_streaming_conv_1d( ctx, states->model_3, encoder->model_3, x );
    x = moshi_seanet_resnet_block( ctx, states->model_4, encoder->model_4, x );
    x = ggml_elu( ctx, x );
    x = moshi_streaming_conv_1d( ctx, states->model_6, encoder->model_6, x );
    x = moshi_seanet_resnet_block( ctx, states->model_7, encoder->model_7, x );
    x = ggml_elu( ctx, x );
    x = moshi_streaming_conv_1d( ctx, states->model_9, encoder->model_9, x );
    x = moshi_seanet_resnet_block( ctx, states->model_10, encoder->model_10, x );
    x = ggml_elu( ctx, x );
    x = moshi_streaming_conv_1d( ctx, states->model_12, encoder->model_12, x );
    x = ggml_elu( ctx, x );
    x = moshi_streaming_conv_1d( ctx, states->model_14, encoder->model_14, x );
    return x;
}

void get_weights( WeightLoader * loader, std::string path, moshi_seanet_encoder_t * encoder ) {
    get_weights( loader, path + "model.0.", encoder->model_0 );
    get_weights( loader, path + "model.1.", encoder->model_1 );
    // elu
    get_weights( loader, path + "model.3.", encoder->model_3 );
    get_weights( loader, path + "model.4.", encoder->model_4 );
    // elu
    get_weights( loader, path + "model.6.", encoder->model_6 );
    get_weights( loader, path + "model.7.", encoder->model_7 );
    // elu
    get_weights( loader, path + "model.9.", encoder->model_9 );
    get_weights( loader, path + "model.10.", encoder->model_10 );
    // elu
    get_weights( loader, path + "model.12.", encoder->model_12 );
    // elu
    get_weights( loader, path + "model.14.", encoder->model_14 );
}

moshi_seanet_encoder_states_t * create_moshi_seanet_encoder_states(
    StateContext * state_ctx,
    moshi_seanet_encoder_t * encoder
) {
    auto states = new moshi_seanet_encoder_states_t;
    moshi_streaming_conv_1d_state( state_ctx, encoder->model_0, states->model_0 );
    moshi_seanet_resnet_block_state( state_ctx, encoder->model_1, states->model_1 );
    moshi_streaming_conv_1d_state( state_ctx, encoder->model_3, states->model_3 );
    moshi_seanet_resnet_block_state( state_ctx, encoder->model_4, states->model_4 );
    moshi_streaming_conv_1d_state( state_ctx, encoder->model_6, states->model_6 );
    moshi_seanet_resnet_block_state( state_ctx, encoder->model_7, states->model_7 );
    moshi_streaming_conv_1d_state( state_ctx, encoder->model_9, states->model_9 );
    moshi_seanet_resnet_block_state( state_ctx, encoder->model_10, states->model_10 );
    moshi_streaming_conv_1d_state( state_ctx, encoder->model_12, states->model_12 );
    moshi_streaming_conv_1d_state( state_ctx, encoder->model_14, states->model_14 );
    return states;
}

/*****************************************************************************\
 *   moshi.modules.seanet.SEANetDecoder
 * located in models:
 *   mimi.decoder
\*****************************************************************************/

struct moshi_seanet_decoder_t {
    own_ptr<moshi_streaming_conv_1d_t> model_0;
    own_ptr<moshi_streaming_conv_transpose_1d_t> model_2;
    own_ptr<moshi_seanet_resnet_block_t> model_3;
    own_ptr<moshi_streaming_conv_transpose_1d_t> model_5;
    own_ptr<moshi_seanet_resnet_block_t> model_6;
    own_ptr<moshi_streaming_conv_transpose_1d_t> model_8;
    own_ptr<moshi_seanet_resnet_block_t> model_9;
    own_ptr<moshi_streaming_conv_transpose_1d_t> model_11;
    own_ptr<moshi_seanet_resnet_block_t> model_12;
    own_ptr<moshi_streaming_conv_1d_t> model_14;
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
        ScratchContext & ctx,
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
    moshi_streaming_conv_1d_state( state_ctx, decoder->model_0, states->model_0 );
    calc_out_dim( decoder->model_0, x_ne, out_ne );

    moshi_streaming_conv_transpose_1d_state( state_ctx, decoder->model_2, out_ne, states->model_2 );
    calc_out_dim( decoder->model_2, out_ne, out_ne );
    moshi_seanet_resnet_block_state( state_ctx, decoder->model_3, states->model_3 );
    calc_out_dim( decoder->model_3, out_ne, out_ne );

    moshi_streaming_conv_transpose_1d_state( state_ctx, decoder->model_5, out_ne, states->model_5 );
    calc_out_dim( decoder->model_5, out_ne, out_ne );
    moshi_seanet_resnet_block_state( state_ctx, decoder->model_6, states->model_6 );
    calc_out_dim( decoder->model_6, out_ne, out_ne );

    moshi_streaming_conv_transpose_1d_state( state_ctx, decoder->model_8, out_ne, states->model_8 );
    calc_out_dim( decoder->model_8, out_ne, out_ne );
    moshi_seanet_resnet_block_state( state_ctx, decoder->model_9, states->model_9 );
    calc_out_dim( decoder->model_9, out_ne, out_ne );

    moshi_streaming_conv_transpose_1d_state( state_ctx, decoder->model_11, out_ne, states->model_11 );
    calc_out_dim( decoder->model_11, out_ne, out_ne );
    moshi_seanet_resnet_block_state( state_ctx, decoder->model_12, states->model_12 );
    calc_out_dim( decoder->model_12, out_ne, out_ne );

    moshi_streaming_conv_1d_state( state_ctx, decoder->model_14, states->model_14 );
    return states;
}



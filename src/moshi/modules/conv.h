#pragma once

/*****************************************************************************\
 *   moshi.modules.conv.StreamingConv1d
 * located in models:
 *   mimi.decoder.model.0,14
 *   mimi.decoder.model.3,6,9,12.block.1
\*****************************************************************************/

struct moshi_streaming_conv_1d_t {
    int in_channels;      // conv.conv.in_channels
    int out_channels;     // conv.conv.out_channels
    int kernel_size;      // conv.conv.kernel_size[0]
    // conv.conv.stride[0] == 1    anything else untested
    // conv.conv.groups == 1       anything else untested
    // conv.conv.padding[0] == 0   anything else untested
    // conv.conv.dilation[0] == 1  anything else untested
    ggml_tensor * weight; // conv.conv.weight
    ggml_tensor * bias;   // conv.conv.bias
};

ggml_tensor * moshi_streaming_conv_1d (
        ggml_context * ctx,
        ggml_tensor * prev,
        moshi_streaming_conv_1d_t * conv,
        ggml_tensor * x ) {
    const int kernel_size = conv->kernel_size;
    const int dilation = 1;
    const int stride = 1;
    int TP = (kernel_size - 1) * dilation + 1 - stride;

    auto prev_tp = ggml_view_3d( ctx, prev,
        TP, prev->ne[1], prev->ne[2],
        prev->nb[1], prev->nb[2],
        prev->nb[0] * (prev->ne[0] - TP) );
    x = ggml_concat( ctx, prev_tp, x, 0 );
    x = ggml_cpy( ctx, x, prev );

    // assert conv.conv.conv.stride[0] == 1
    // assert conv.conv.conv.padding[0] == 0
    // assert conv.conv.conv.dilation[0] == 1
    // assert conv.conv.conv.groups == 1
    auto y = ggml_conv_1d( ctx, conv->weight, x, 1, 0, 1 );
    if ( conv->bias ) {
        y = ggml_add( ctx, y, conv->bias );
    }

    return y;
}

void get_weights( WeightLoader * loader, std::string path,
        moshi_streaming_conv_1d_t * conv ) {
    // NOTE: ggml_conv_1d requires GGML_TYPE_F16 due to im2col requiring it
    assert( loader->fetch( &conv->weight, path + "conv.conv.weight", (void*)ggml_conv_1d ) );
    loader->fetch( &conv->bias, path + "conv.conv.bias", (void*)ggml_add, 1 );
}

void moshi_streaming_conv_1d_state(
        StateContext * state_ctx,
        moshi_streaming_conv_1d_t * conv,
        const NE x_ne,
        ggml_tensor * &prev ) {
    const int kernel_size = conv->kernel_size;
    const int stride = 1;
    //const int padding = 0;
    const int dilation = 1;
    //int lout = (lin + 2*padding - dilation*(kernel_size - 1) - 1) / stride + 1;
    //int lout = (kernel_size - 1) * dilation + 1 - stride;
    int TP = (kernel_size - 1) * dilation + 1 - stride;
    NE ne = { x_ne[0] + TP, conv->in_channels, x_ne[2], 1 };
    state_ctx->fill(ne, 0.f, &prev);
}

bool calc_out_dim( const moshi_streaming_conv_1d_t * conv,
        const NE x_ne, NE &y_ne ) {
    const int kernel_size = conv->kernel_size;
    const int stride = 1;
    const int padding = 0;
    const int dilation = 1;

    int TP = (kernel_size - 1) * dilation + 1 - stride;
    int lin = x_ne[0] + TP;
    //int cin = conv->in_channels;
    y_ne[0] = (lin + 2*padding - dilation*(kernel_size - 1) - 1) / stride + 1;
    y_ne[1] = conv->out_channels;
    y_ne[2] = x_ne[2];
    y_ne[3] = 1;
    return true;
}

/*****************************************************************************\
 *   moshi.modules.conv.StatelessConv1d
 * custom version of StreamingConv1d that did not use it's state variables.
 * located in models:
 *   mimi.decoder.model.3,6,9,12.block.3
\*****************************************************************************/

struct moshi_stateless_conv_1d_t {
    int in_channels;      // conv.conv.in_channels
    int out_channels;     // conv.conv.out_channels
    int kernel_size;      // conv.conv.kernel_size[0]
    // conv.conv.stride[0] == 1    anything else untested
    // conv.conv.groups == 1       anything else untested
    // conv.conv.padding[0] == 0   anything else untested
    // conv.conv.dilation[0] == 1  anything else untested
    ggml_tensor * weight; // conv.conv.weight
    ggml_tensor * bias;   // conv.conv.bias
};

ggml_tensor * moshi_stateless_conv_1d (
        ggml_context * ctx,
        moshi_stateless_conv_1d_t * conv,
        ggml_tensor * x ) {
    // assert conv.conv.conv.stride[0] == 1
    // assert conv.conv.conv.padding[0] == 0
    // assert conv.conv.conv.dilation[0] == 1
    // assert conv.conv.conv.groups == 1
    auto y = ggml_conv_1d( ctx, conv->weight, x, 1, 0, 1 );
    if ( conv->bias )
        y = ggml_add( ctx, y, conv->bias );
    return y;
}

bool calc_out_dim( const moshi_stateless_conv_1d_t * conv,
        const NE x_ne, NE &y_ne ) {
    const int kernel_size = conv->kernel_size;
    const int stride = 1;
    const int padding = 0;
    const int dilation = 1;

    int TP = (kernel_size - 1) * dilation + 1 - stride;
    int lin = x_ne[0] + TP;
    //int cin = conv->in_channels;
    y_ne[0] = (lin + 2*padding - dilation*(kernel_size - 1) - 1) / stride + 1;
    y_ne[1] = conv->out_channels;
    y_ne[2] = x_ne[2];
    return true;
}
void get_weights( WeightLoader * loader, std::string path, moshi_stateless_conv_1d_t * conv ) {
    // NOTE: ggml_conv_1d requires GGML_TYPE_F16 due to im2col requiring it
    assert( loader->fetch( &conv->weight, path + "conv.conv.weight", (void*)ggml_conv_1d ) );
    loader->fetch( &conv->bias, path + "conv.conv.bias", (void*)ggml_add, 1 );
}

/*****************************************************************************\
 *   moshi.modules.conv.StreamingConvTranspose1d
 * located in models:
 *   mimi.upsample.convtr
 *   mimi.decoder.model.2,5,8,11
\*****************************************************************************/

struct moshi_streaming_conv_transpose_1d_t {
    int in_channels;      // convtr.convtr.in_channels
    int out_channels;     // convtr.convtr.out_channels
    int kernel_size;      // convtr.convtr.kernel_size[0]
    int stride;           // convtr.convtr.stride[0]
    int groups;           // convtr.convtr.groups
    // convtr.convtr.padding[0] == 0   anything else untested
    // convtr.convtr.dilation[0] == 1  anything else untested
    ggml_tensor * weight; // convtr.convtr.weight
    ggml_tensor * bias;   // convtr.convtr.bias
};

void moshi_streaming_conv_transpose_1d_state(
        StateContext * state_ctx,
        moshi_streaming_conv_transpose_1d_t * convtr,
        const NE x_ne,
        ggml_tensor * &prev_y ) {
    int kernel_size = convtr->kernel_size;
    int stride = convtr->stride;
    int padding = 0;
    int output_padding = 0;
    int dilation = 1;
    int lout = (x_ne[0] - 1) * stride
            - 2 * padding
            + dilation * (kernel_size - 1)
            + output_padding + 1;
    NE ne = { lout, convtr->out_channels, x_ne[2], 1 };
    state_ctx->fill( ne, 0.f, &prev_y );
}

bool calc_out_dim( const moshi_streaming_conv_transpose_1d_t * conv,
        const NE x_ne, NE &y_ne ) {
    const int kernel_size = conv->kernel_size;
    const int stride = conv->stride;
    const int padding = 0;
    const int output_padding = 0;
    const int dilation = 1;
    //const int groups = conv->groups;

    // output from convtr
    int lin = x_ne[0];
    int lout = (lin - 1) * stride
            - 2*padding
            + dilation * (kernel_size - 1)
            + output_padding + 1;

    // after windowing
    int PT = kernel_size - stride;
    lout = lout - PT;

    y_ne[0] = lout;
    y_ne[1] = conv->out_channels;
    y_ne[2] = x_ne[2];
    y_ne[3] = 1;
    return true;
}

ggml_tensor * moshi_streaming_conv_transpose_1d(
        ggml_context * ctx,
        ggml_tensor * prev_y,
        moshi_streaming_conv_transpose_1d_t * convtr,
        ggml_tensor * x ) {
    int PT = convtr->kernel_size - convtr->stride;

    ggml_tensor * y = NULL;
    if ( convtr->groups == 1 ) {
        // this only works for groups of 1
        y = ggml_conv_transpose_1d( ctx, convtr->weight, x, convtr->stride, 0, 1 );
    } else {
        auto weight = convtr->weight;
        // HACK: trick to divide into 4 separate multiples
        //assert convtr.convtr.convtr.groups == weight.shape[0]
        // assert these as untested for now, 4 is ideal though
        //assert convtr.convtr.convtr.groups == 512, "untested"
        //assert weight.shape == (512, 1, 4), "untested"
        for ( int i = 0; i < weight->ne[0]; i++ ) {
            auto subweight = ggml_view_3d( ctx, weight,
                1, weight->ne[2], weight->ne[1],
                weight->nb[2], weight->nb[2], // TODO: should it be nb[1]?
                weight->nb[0] * i );
            auto y_piece = ggml_mul( ctx, x, subweight );
            if ( y )
                y = ggml_concat( ctx, y, y_piece, 0 );
            else
                y = y_piece;
        }
    }

    // y[..., :PT] += prev_y[..., -PT:]
    auto partial = ggml_view_3d( ctx, prev_y,
        PT, prev_y->ne[1], prev_y->ne[2],
        prev_y->nb[1], prev_y->nb[2],
        prev_y->nb[0] * (prev_y->ne[0] - PT) );
    auto lower = ggml_view_3d( ctx, y,
        PT, y->ne[1], y->ne[2],
        y->nb[1], y->nb[2],
        0 );
    lower = ggml_add_inplace( ctx, lower, partial );
    y = ggml_view_3d( ctx, lower,
        y->ne[0], y->ne[1], y->ne[2],
        y->nb[1], y->nb[2],
        0 );

    // prev_y = y
    y = ggml_cpy( ctx, y, prev_y );

    if ( convtr->bias ) {
        y = ggml_add( ctx, y, convtr->bias );
    }

    // y = y[..., :-PT]
    y = ggml_view_3d( ctx, y,
        y->ne[0] - PT, y->ne[1], y->ne[2],
        y->nb[1], y->nb[2],
        0 );
    
    return ggml_cont( ctx, y );
}

void get_weights( WeightLoader * loader, std::string path,
        moshi_streaming_conv_transpose_1d_t * convtr ) {
    assert( loader->fetch( &convtr->weight, path + "convtr.convtr.weight",
        (void*)ggml_conv_transpose_1d ) );
    loader->fetch( &convtr->bias, path + "convtr.convtr.bias", (void*)ggml_add, 1 );
}




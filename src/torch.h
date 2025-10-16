#pragma once

/*******************************************\
 *     Modules in PyTorch
 *
 * these are not complete implementations,
 * only enough to get moshi working.
\*******************************************/

/******************************\
 * torch.nn.Conv1d
\******************************/

struct torch_nn_conv1d_t {
    ggml_tensor * weight;
};

ggml_tensor * torch_nn_conv1d(
        ggml_context * ctx,
        torch_nn_conv1d_t * conv,
        ggml_tensor * x ) {
    // NOTE: these were not testable so not included
    //assert conv.stride[0] == 1
    //assert conv.padding[0] == 0
    //assert conv.dilation[0] == 1
    //assert conv.groups == 1
    //assert not conv.bias
    auto y = ggml_conv_1d( ctx, conv->weight, x, 1, 0, 1 );
    return y;
}

void get_weights( WeightLoader * loader, std::string path,
        torch_nn_conv1d_t * conv ) {
    // NOTE: ggml_conv_1d requires GGML_TYPE_F16 due to im2col requiring it
    assert( loader->fetch( &conv->weight, path + "weight", (void*)ggml_conv_1d ) );
}

/******************************\
 * torch.nn.LayerNorm
\******************************/

struct torch_nn_layer_norm_t {
    float eps;
    ggml_tensor * weight;
    ggml_tensor * bias;
};

ggml_tensor * torch_nn_layer_norm(
        ggml_context * ctx,
        torch_nn_layer_norm_t * norm,
        ggml_tensor * x ) {
    if ( x->type != GGML_TYPE_F32 )
        x = ggml_cast( ctx, x, GGML_TYPE_F32 );
    x = ggml_norm( ctx, x, norm->eps );
    x = ggml_mul( ctx, x, norm->weight );
    if ( norm->bias )
        x = ggml_add( ctx, x, norm->bias );
    return x;
}

void get_weights( WeightLoader * loader, std::string path,
        torch_nn_layer_norm_t * norm ) {
    assert( loader->fetch( &norm->weight, path + "weight", (void*)ggml_mul ) );
    loader->fetch( &norm->bias, path + "bias", (void*)ggml_add );
}

/******************************\
 * torch.nn.Linear
\******************************/

struct torch_nn_linear_t {
    ggml_tensor * weight;
    ggml_tensor * bias;
};

ggml_tensor * torch_nn_linear(
        ggml_context * ctx,
        torch_nn_linear_t * linear,
        ggml_tensor * x ) {
    ggml_tensor * y = ggml_mul_mat( ctx, linear->weight, x );
    if ( linear->bias )
        y = ggml_add( ctx, y, linear->bias );
    return y;
}

void get_weights( WeightLoader * loader, std::string path,
        torch_nn_linear_t * linear ) {
    assert( loader->fetch( &linear->weight, path + "weight", (void*)ggml_mul_mat ) );
    loader->fetch( &linear->bias, path + "bias", (void*)ggml_add );
}

// utility that only applies a subset of a linear module
ggml_tensor * torch_nn_linear_view(
        ggml_context * ctx,
        torch_nn_linear_t * linear,
        int offset,
        int width,
        ggml_tensor * x ) {
    auto weight = linear->weight;
    auto w_view = ggml_view_2d( ctx, weight,
        weight->ne[0], width,
        weight->nb[1],
        weight->nb[1] * offset );
    auto y = ggml_mul_mat( ctx, w_view, x );
    if ( linear->bias )
        y = ggml_add( ctx, y, linear->bias );
    return y;
}

// scaled_dot_product_attention used -infinity which does not multiply against 0
// so changed to use a very large negative number, would be nice to have a 
// mathematical way to generate the bias from a mask, as opposed to a boolean
// operations as it was before, since ggml does not currently support them
ggml_tensor * torch_nn_functional_scaled_dot_product_attention(
        ScratchContext & ctx,
        ggml_tensor * query,
        ggml_tensor * key,
        ggml_tensor * value,
        ggml_tensor * attn_mask ) {
    ggml_tensor * attn_bias = NULL;
    if (attn_mask) {
        // invert mask
        auto one = ctx.constant( 1.f );
        attn_bias = ggml_add( ctx, ggml_neg( ctx, attn_mask ), one );
        // max negative value
        // HACK: can't use infinity, so just use a very large number
        attn_bias = ggml_scale( ctx, attn_bias, -100000.0 );
    } else {
        attn_bias = NULL;
    }
    // if we need -inf, in theory we can just scale it by 2 or higher
    float scale_factor = 1.f / sqrtf( query->ne[0] );
    auto attn_weight = ggml_mul_mat( ctx, key, query );
    attn_weight = ggml_soft_max_ext( ctx, attn_weight, attn_bias, scale_factor, 0.0f );
    value = ggml_cont( ctx, ggml_transpose( ctx, value ) );
    auto x = ggml_mul_mat( ctx, value, attn_weight );
    return x;
}


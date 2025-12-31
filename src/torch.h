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
    auto n = loader->fetch( &conv->weight, path + "weight", (void*)ggml_conv_1d );
    assert( n );
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
    auto n = loader->fetch( &norm->weight, path + "weight", (void*)ggml_mul );
    assert( n );
    // bias not required
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
    if ( loader->quantize ) {
        auto n = loader->fetch( &linear->weight, path + "weight", loader->qtype );
        assert( n );
    } else {
        auto n = loader->fetch( &linear->weight, path + "weight", (void*)ggml_mul_mat );
        assert( n );
    }
    // bias not required
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

/*****************************************************************************\
 * torch.nn.functional.scaled_dot_product_attention
\*****************************************************************************/

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
    float scale_factor = 1.f / sqrtf( (float) query->ne[0] );
    auto attn_weight = ggml_mul_mat( ctx, key, query );
    attn_weight = ggml_soft_max_ext( ctx, attn_weight, attn_bias, scale_factor, 0.0f );
    value = ggml_cont( ctx, ggml_transpose( ctx, value ) );
    auto x = ggml_mul_mat( ctx, value, attn_weight );
    return x;
}

/*****************************************************************************\
 * custom scaled_dot_product_attention
 * 
 * 1) create a pattern
 * 2) index into the pattern to get the bias
 * 3) pass that bias to the custom scaled_dot_product_attention
\*****************************************************************************/

struct bias_pattern_t {
    int capacity;
    int t;
    int start; // = pattern->capacity * 2 - pattern->t
    own_ctx_tensor tensor;
};

int g_bias_pattern = 0;
void create_bias_pattern(
        ggml_backend * backend,
        bias_pattern_t & pattern,
        int capacity,
        int t,
        float hi = 1.f, float lo = 0.f
) {
    auto & tensor = pattern.tensor;
    int start = capacity * 2 - t;
    int width = start + capacity;
    tensor.new_tensor( GGML_NE( width, t ), GGML_TYPE_F32, backend );
    pattern.capacity = capacity;
    pattern.t = t;
    pattern.start = start;
    auto nelements = ggml_nelements( tensor );
    std::vector<float> values( nelements );
    for ( int j = 0; j < t; j++ ) {
        int toff = j * width;
        int right = start + 1 + j;
        for ( int i = 0; i < right; i++ ) {
            values[ toff + i ] = hi;
        }
        for ( int i = right; i < width; i++ ) {
            values[ toff + i ] = lo;
        }
        int b = t - j - 1;
        toff += capacity - 1;
        for ( int i = 0; i < b; i++ ) {
            values[ toff - i ] = lo;
        }
    }
    ggml_backend_tensor_set( tensor, values.data(), 0, ggml_nbytes( tensor ) );
    g_bias_pattern++;
}

ggml_tensor * bias_pattern_index(
        ggml_context * ctx,
        bias_pattern_t & pattern,
        int offset
) {
    auto & tensor = pattern.tensor;
    if ( offset <= pattern.capacity )
        offset = pattern.start - offset;
    else
        offset = pattern.capacity - ( offset % pattern.capacity );
    auto view = ggml_view_2d( ctx,
        tensor,
        pattern.capacity,
        pattern.t,
        tensor->nb[1],
        offset * tensor->nb[0] );
    auto cont = ggml_cont( ctx, view );
    return cont;
}

ggml_tensor * torch_nn_functional_scaled_dot_product_attention_custom(
        ScratchContext & ctx,
        ggml_tensor * query,
        ggml_tensor * key,
        ggml_tensor * value,
        ggml_tensor * attn_bias ) {
    float scale_factor = 1.f / sqrtf( (float) query->ne[0] );
    auto attn_weight = ggml_mul_mat( ctx, key, query );
    attn_weight = ggml_soft_max_ext( ctx, attn_weight, attn_bias, scale_factor, 0.0f );
    value = ggml_cont( ctx, ggml_transpose( ctx, value ) );
    auto x = ggml_mul_mat( ctx, value, attn_weight );
    return x;
}

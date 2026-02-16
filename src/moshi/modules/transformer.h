#pragma once

#include "gating.h"
#include "rope.h"

/*****************************************\
 * moshi.modules.transformer.RMSNorm
\*****************************************/

struct moshi_rms_norm_t {
    float eps;
    ggml_tensor * alpha;
};

ggml_tensor * moshi_rms_norm(
        ggml_context * ctx,
        moshi_rms_norm_t * norm,
        ggml_tensor * x ) {
    if ( x->type != GGML_TYPE_F32 )
        x = ggml_cast( ctx, x, GGML_TYPE_F32 );
    auto y = ggml_rms_norm( ctx, x, norm->eps );
    return ggml_mul( ctx, norm->alpha, y );
}

void get_weights( WeightLoader * loader, std::string path, moshi_rms_norm_t * norm ) {
    auto n = loader->fetch( &norm->alpha, path + "alpha", (void*)ggml_rms_norm );
    assert( n );
}

/*****************************************\
 * moshi.modules.transformer.LayerScale
\*****************************************/

struct moshi_layer_scale_t {
    ggml_tensor * scale;
};

ggml_tensor * moshi_layer_scale(
        ggml_context * ctx,
        moshi_layer_scale_t * m,
        ggml_tensor * x ) {
    return ggml_mul( ctx, x, m->scale );
}

void get_weights( WeightLoader * loader, std::string path,
        moshi_layer_scale_t * scale ) {
    auto n = loader->fetch( &scale->scale, path + "scale", (void*)ggml_mul );
    assert( n );
}

/*****************************************\
 * moshi.modules.transformer.LayerScale
\*****************************************/

ggml_tensor * moshi_apply_weights_per_step_linear(
        ggml_context * ctx,
        own_ptr_vector<torch_nn_linear_t> & modules,
        std::vector<int> & schedule,
        ggml_tensor * x,
        int offset ) {
    /* Utility to apply a multi linear layer to the given input. A multi linear layer
    applies a different set of weight for each time step.

    Args:
        modules (nn.ModuleList): apply weights per step.
        schedule (list[int] or None): schedule for weight sharing.
        x (torch.Tensor): Input tensor, with shape `[B, T, C]`.
        offset (int): offset for the current time step, in particular for decoding, with
            time steps provided one by one.
    */

    if ( modules.size() == 1 ) {
        auto module = modules[0];
        auto y = torch_nn_linear( ctx, module, x );
        return y;
    }

    int T = (int) x->ne[1];
    ggml_tensor * ys = NULL;
    for ( int t = 0; t < T; t++ ) {
        int module_index = t + offset;
        if ( schedule.size() )
            module_index = schedule[module_index];

        auto x_view = ggml_view_3d( ctx, x,
            x->ne[0], 1, x->ne[2],
            x->nb[1], x->nb[2],
            t * x->nb[1] );

        auto module = modules[module_index];
        auto y = torch_nn_linear( ctx, module, x_view );

        if ( ys )
            ys = ggml_concat(ctx, ys, y, 1);
        else
            ys = y;
    }
    return ys;
}

ggml_tensor * moshi_apply_weights_per_step_gating(
        ggml_context * ctx,
        own_ptr_vector<moshi_activation_gating_t> & modules,
        std::vector<int> & schedule,
        ggml_tensor * x,
        int offset ) {
    /* Utility to apply a multi linear layer to the given input. A multi linear layer
    applies a different set of weight for each time step.

    Args:
        modules (nn.ModuleList): apply weights per step.
        schedule (list[int] or None): schedule for weight sharing.
        x (torch.Tensor): Input tensor, with shape `[B, T, C]`.
        offset (int): offset for the current time step, in particular for decoding, with
            time steps provided one by one.
    */

    if ( modules.size() == 1 ) {
        int module_index = 0;
        auto module = modules[module_index];
        auto y = moshi_activation_gating( ctx, module, x );
        return y;
    }

    int T = (int) x->ne[1];
    ggml_tensor * ys = NULL;
    for ( int t = 0; t < T; t++ ) {
        int module_index = t + offset;
        if ( schedule.size() )
            module_index = schedule[module_index];

        auto x_view = ggml_view_3d( ctx, x,
            x->ne[0], 1, x->ne[2],
            x->nb[1], x->nb[2],
            t * x->nb[1] );

        auto module = modules[module_index];
        auto y = moshi_activation_gating( ctx, module, x_view );

        if ( ys )
            ys = ggml_concat( ctx, ys, y, 1 );
        else
            ys = y;
    }
    return ys;
}


struct moshi_kv_cache_state_t {
    ggml_tensor * keys;
    ggml_tensor * values;
};

#define CACHE_BF16

moshi_kv_cache_state_t * moshi_kv_cache_state(
        StateContext * state_ctx,
        int dim_per_head,
        int capacity,
        int num_heads,
        int batch_size ) {
    auto states = new moshi_kv_cache_state_t;
    NE ne = { dim_per_head, capacity, num_heads, batch_size };
#ifdef CACHE_BF16
    state_ctx->fill16( ne, GGML_TYPE_BF16, 0, &states->keys );
    state_ctx->fill16( ne, GGML_TYPE_BF16, 0, &states->values );
#else
    state_ctx->fill( ne, 0.f, &states->keys );
    state_ctx->fill( ne, 0.f, &states->values );
#endif
    return states;
}

std::tuple<ggml_tensor*,ggml_tensor*> moshi_kv_cache_insert_kv(
        ggml_context * ctx,
        ggml_tensor * keys,
        ggml_tensor * values,
        int index,
        ggml_tensor * k,
        ggml_tensor * v ) {
    int T = (int) k->ne[1];
    int capacity = (int) keys->ne[1];
    index = index % capacity;
    assert( (index % T) == 0 );
    // keys update cache
    auto cache_0_0 = ggml_view_4d( ctx, keys,
        keys->ne[0], // D
        T, // from context length to T
        keys->ne[2], // H
        keys->ne[3], // B
        keys->nb[1],
        keys->nb[2],
        keys->nb[3],
        keys->nb[1] * index
    );
#ifdef CACHE_BF16
    k = ggml_cast( ctx, k, GGML_TYPE_BF16 );
#endif
    cache_0_0 = ggml_cpy( ctx, k, cache_0_0 );
    keys = ggml_view_4d( ctx, cache_0_0,
        keys->ne[0], // D
        keys->ne[1], // context
        keys->ne[2], // H
        keys->ne[3], // B
        keys->nb[1],
        keys->nb[2],
        keys->nb[3],
        keys->nb[1] * -index
    );
    // values update cache
    auto cache_1_0 = ggml_view_4d( ctx, values,
        values->ne[0], // D
        T, // from context length to T
        values->ne[2], // H
        values->ne[3], // B
        values->nb[1],
        values->nb[2],
        values->nb[3],
        values->nb[1] * index
    );
#ifdef CACHE_BF16
    v = ggml_cast( ctx, v, GGML_TYPE_BF16 );
#endif
    cache_1_0 = ggml_cpy( ctx, v, cache_1_0 );
    values = ggml_view_4d( ctx, cache_1_0,
        values->ne[0], // D
        values->ne[1], // context
        values->ne[2], // H
        values->ne[3], // B
        values->nb[1],
        values->nb[2],
        values->nb[3],
        values->nb[1] * -index
    );
    return std::make_tuple( keys, values );
}

std::tuple<ggml_tensor*,ggml_tensor*> moshi_kv_cache_insert_kv(
        ggml_context * ctx,
        ggml_tensor * keys,
        ggml_tensor * values,
        ggml_tensor * indices,
        ggml_tensor * k,
        ggml_tensor * v ) {
    assert( indices->ne[0] == k->ne[1] ); // one index per T
    keys = ggml_set_rows( ctx, keys, k, indices );
    values = ggml_set_rows( ctx, values, v, indices );
    return std::make_tuple( keys, values );
}

ggml_tensor * moshi_kv_cache_get_positions(
        ScratchContext & ctx,
        int end_offset,
        int capacity ) {
    auto indexes = ctx.arange( 0, (float) capacity, 1 );

    auto last_offset = end_offset - 1;
    int end_index = last_offset % capacity;
    //delta = indexes - end_index
    auto const_end_index = ctx.constant( (float)end_index );
    auto delta = ggml_sub( ctx, indexes, const_end_index );

    // We know that if `index == end_index`, then we should output `end_offset`.
    // If `index = end_index - 1` we should output `end_offset - 1`
    // If `index = end_index - n` we should output `end_offset - n`
    // Now, for `index == end_index + 1` , we actually have the oldest entry in the cache,
    // so we should output `end_index + 1 - capacity`

    // so the clamp is an inplace op
    auto capacity_mask = ggml_clamp( ctx, delta, 0, 1 );
    auto const_last_offset = ctx.constant( (float)last_offset );
    auto positions = ggml_add( ctx, delta, const_last_offset );
    positions = ggml_sub( ctx, positions, ggml_scale( ctx, capacity_mask, (float) capacity ) );

    auto one = ctx.constant( 1.f );
    indexes = ggml_neg( ctx, indexes );

    auto const_end_offset = ctx.constant( (float)end_offset );
    indexes = ggml_add( ctx, indexes, const_end_offset );

    auto valid = ggml_clamp( ctx, indexes, 0, 1 );
    positions = ggml_add( ctx, positions, one );
    positions = ggml_mul( ctx, positions, valid );
    positions = ggml_sub( ctx, positions, one );

    return positions;
}

/*************************************************************\
 *  moshi.modules.transformer.StreamingMultiheadAttention
 *
 * location in models:
 * lm.transformer.layers.0.self_attn
 * lm.transformer.layers.0.cross_attention
 * lm.depformer.layers.0.self_attn
 * mimi.decoder_transformer.transformer.layers[0].self_attn
\*************************************************************/

struct moshi_smha_t {
    int embed_dim;
    int num_heads;
    bool cross_attention;
    bool cache_cross_attention;
    bool causal;
    int rope_max_period;
    int context;
    int weights_per_step;
    std::vector<int> weights_per_step_schedule;
    own_ptr_vector<torch_nn_linear_t> in_projs;
    own_ptr_vector<torch_nn_linear_t> out_projs;
};

struct moshi_smha_state_t {
    ggml_tensor * k_cross;
    ggml_tensor * v_cross;
    own_ptr<moshi_kv_cache_state_t> kv_cache;
};

moshi_smha_state_t * moshi_smha_state( StateContext * state_ctx,
        moshi_smha_t * attn, ggml_tensor * k_cross ) {
    auto state = new moshi_smha_state_t;
    int dim_per_head = attn->embed_dim / attn->num_heads;
    int num_heads = attn->num_heads;
    if ( ! attn->cross_attention ) {
        int capacity = attn->context? attn->context : attn->weights_per_step;
        int batch_size = 1;
        state->kv_cache = moshi_kv_cache_state( state_ctx, dim_per_head, capacity, num_heads,
            batch_size );
        state->k_cross = NULL;
        state->v_cross = NULL;
    } else {
        state->k_cross = NULL;
        state->v_cross = NULL;
        state->kv_cache = NULL;
        assert( k_cross );
        GGML_NE ne( dim_per_head, k_cross->ne[1], num_heads );
        state_ctx->fill( ne, 0.f, &state->k_cross );
        state_ctx->fill( ne, 0.f, &state->v_cross );
    }
    return state;
}

void init( ScratchContext * scratch_ctx, moshi_smha_state_t * state,
        moshi_smha_t * attn, ggml_tensor * condition_cross ) {
    if ( condition_cross && attn->cache_cross_attention ) {
        ScratchContext &ctx = *scratch_ctx;

        int H = attn->num_heads;
        auto in_proj = attn->in_projs[0];
        int dim = (int) in_proj->weight->ne[1] / 3;
        int dim2 = dim * 2;

        auto kv = torch_nn_linear_view( ctx, in_proj, dim, dim2, condition_cross );

        //k, v = rearrange(kv, "b t (p h d) -> p b h t d", p=2, h=attn->num_heads)
        auto k = ggml_view_3d( ctx, kv,
            kv->ne[0] / 2,
            kv->ne[1],
            kv->ne[2],
            kv->nb[1],
            kv->nb[2],
            0 );
        // b t (h d) -> b t h d
        k = ggml_cont( ctx, k );
        k = ggml_reshape_4d( ctx, k,
            k->ne[0] / H,
            H,
            k->ne[1],
            k->ne[2] );
        // b t h d -> b h t d
        k = ggml_permute( ctx, k, 0, 2, 1, 3 );

        auto v = ggml_view_3d( ctx, kv,
            kv->ne[0] / 2,
            kv->ne[1],
            kv->ne[2],
            kv->nb[1],
            kv->nb[2],
            kv->nb[1] / 2 );
        // b t (h d) -> b t h d
        v = ggml_cont( ctx, v );
        v = ggml_reshape_4d( ctx, v,
            v->ne[0] / H,
            H,
            v->ne[1],
            v->ne[2] );
        // b t h d -> b h t d
        v = ggml_permute( ctx, v, 0, 2, 1, 3 );

        k = ggml_cpy( ctx, k, state->k_cross );
        v = ggml_cpy( ctx, v, state->v_cross );
        ctx.build_forward_expand( k );
        ctx.build_forward_expand( v );
        ctx.compute();
    }
}

ggml_tensor * get_attn_bias( GraphContext & ctx, bias_pattern_t * pattern,
        int capacity, int64_t T, int64_t offset ) {
    if ( ! pattern->tensor ) {
        create_bias_pattern( ctx.backend, *pattern, capacity, (int) T, 0, -INFINITY );
    }
    return bias_pattern_index( ctx, *pattern, (int) offset );
}

// utility that can be done once and shared across layers
ggml_tensor * calculate_attn_bias( ScratchContext & ctx, moshi_smha_t * attn,
        int64_t T, int64_t noffset, bias_pattern_t * pattern = NULL ) {
    if ( ! attn->causal )
        return NULL;

    ggml_tensor * pos_k;
    assert( !attn->cross_attention );
    int capacity = attn->context? attn->context : attn->weights_per_step;

    if ( pattern ) {
        if ( ! pattern->tensor ) {
            //create_bias_pattern( ctx.backend, *pattern, capacity, T );
            create_bias_pattern( ctx.backend, *pattern, capacity, (int) T, 0, -INFINITY );
        }
        return bias_pattern_index( ctx, *pattern, (int) noffset );
    }

    pos_k = moshi_kv_cache_get_positions( ctx, (int)( noffset + T ), (int) capacity );

    auto offset = ctx.constant( (float)noffset );
    auto pos_q = ggml_add( ctx, ctx.arange( 0, (float) T, 1 ), offset );
    pos_q = ggml_view_2d( ctx, pos_q, 1, T, pos_q->nb[0], 0 );
    pos_q = ggml_repeat_4d( ctx, pos_q,
        pos_k->ne[0],
        pos_q->ne[1],
        pos_q->ne[2],
        pos_q->ne[3] );
    auto delta = ggml_sub( ctx, pos_q, pos_k );

    auto one = ctx.constant( 1.f );
    auto delta_mask = ggml_clamp( ctx, ggml_add( ctx, delta, one ), 0, 1 );
    auto pos_k_mask = ggml_clamp( ctx, ggml_add( ctx, pos_k, one ), 0, 1 );
    auto attn_bias = ggml_mul( ctx, delta_mask, pos_k_mask );
    if ( attn->context ) {
        auto context = ctx.constant( (float)attn->context );
        auto context_mask = ggml_clamp( ctx, ggml_add( ctx,
            ggml_neg( ctx, delta ), context ), 0, 1 );
        attn_bias = ggml_mul( ctx, attn_bias, context_mask );
    }
    return attn_bias;
}

ggml_tensor * moshi_streaming_multihead_attention(
        GraphContext & ctx,
        moshi_smha_t * attn,
        moshi_smha_state_t * state,
        ggml_tensor * indices,
        int offset,
        ggml_tensor * query,
        ggml_tensor * key,
        /*ggml_tensor * value,*/
        ggml_tensor * attn_bias = NULL,
        timestep_embedding_t * tsemb = NULL
    ) {
    CAPTURE_GROUP( "multihead_attention" );

    // the offset should only matter to depformers, which can be graphed at
    // a higher level because it only calls the graph a few times.
    assert( attn->weights_per_step_schedule.size() == 0 || offset < attn->weights_per_step_schedule.size() );
    assert( attn->weights_per_step_schedule.size() > 0 || attn->in_projs.size() == 1 || offset < attn->in_projs.size() );
    assert( attn->weights_per_step_schedule.size() > 0 || attn->out_projs.size() == 1 || offset < attn->out_projs.size() );

    int T = (int) query->ne[1];
    int H = attn->num_heads;

    ggml_tensor * q;
    ggml_tensor * k;
    ggml_tensor * v;

    if ( attn->cross_attention ) {
        //assert len(attn->in_projs) == 1
        auto in_proj = attn->in_projs[0];
        //assert in_proj.bias is None
        //assert isinstance(in_proj, nn.Linear)
        int dim = (int) in_proj->weight->ne[1] / 3;
        //int dim2 = dim * 2;
        //q = nn.functional.linear(query, in_proj.weight[:dim])
        //q = rearrange(q, "b t (h d) -> b h t d", h=attn->num_heads)
        q = torch_nn_linear_view( ctx, in_proj, 0, dim, query );

        k = state->k_cross;
        v = state->v_cross;
    } else {
        auto projected = moshi_apply_weights_per_step_linear( ctx,
            attn->in_projs, attn->weights_per_step_schedule,
            query, offset );

        //q, k, v = rearrange(
        //    projected, "b t (p h d) -> p b h t d", p=3, h=attn->num_heads
        //)

        q = ggml_view_3d( ctx, projected,
            projected->ne[0] / 3,
            projected->ne[1],
            projected->ne[2],
            projected->nb[1],
            projected->nb[2],
            0 );
        q = ggml_cont( ctx, q );

        k = ggml_view_3d( ctx, projected,
            projected->ne[0] / 3,
            projected->ne[1],
            projected->ne[2],
            projected->nb[1],
            projected->nb[2],
            projected->nb[1] / 3 );
        // b t (h d) -> b t h d
        k = ggml_cont( ctx, k );
        k = ggml_reshape_4d( ctx, k,
            k->ne[0] / H,
            H,
            k->ne[1],
            k->ne[2] );
        // b t h d -> b h t d
        k = ggml_permute( ctx, k, 0, 2, 1, 3 );

        v = ggml_view_3d( ctx, projected,
            projected->ne[0] / 3,
            projected->ne[1],
            projected->ne[2],
            projected->nb[1],
            projected->nb[2],
            projected->nb[1] * 2 / 3 );
        // b t (h d) -> b t h d
        v = ggml_cont( ctx, v );
        v = ggml_reshape_4d( ctx, v,
            v->ne[0] / H,
            H,
            v->ne[1],
            v->ne[2] );
        // b t h d -> b h t d
        v = ggml_permute( ctx, v, 0, 2, 1, 3 );
    }

    // b t (h d) -> b t h d
    q = ggml_reshape_4d( ctx, q,
        q->ne[0] / H,
        H,
        q->ne[1],
        q->ne[2] );
    // b t h d -> b h t d
    q = ggml_permute( ctx, q, 0, 2, 1, 3 );

    if ( attn->rope_max_period ) {
        std::tie(q, k) = moshi_apply_rope( ctx, q, k, tsemb, false );
    }

    if ( attn->causal && ! attn->cross_attention ) {
        assert( k->ne[1] == T );

        std::tie( k, v ) = moshi_kv_cache_insert_kv( ctx,
            state->kv_cache->keys, state->kv_cache->values,
            indices, k, v );
    }

    assert( attn_bias || ! attn->causal );

    //x = nn.functional.scaled_dot_product_attention(q, k, v, attn_bias, dropout_p=0.0)
    auto x = torch_nn_functional_scaled_dot_product_attention_custom( ctx,
        q, k, v, attn_bias );//, dropout_p=0.0);

    //x = rearrange(x, "b h t d -> b t (h d)")
    // b h t d -> b t h d
    auto x2 = ggml_cont( ctx, ggml_permute( ctx, x, 0, 2 ,1 ,3 ) );
    // b t h d -> b t (h d)
    x = ggml_reshape_3d( ctx, x2,
        x2->ne[0] * x2->ne[1],
        x2->ne[2],
        x2->ne[3] );

    x = moshi_apply_weights_per_step_linear( ctx,
        attn->out_projs, attn->weights_per_step_schedule,
        x, offset );

    return x;
}

// doesn't need or use an offset
ggml_tensor * moshi_streaming_multihead_attention(
        GraphContext & ctx,
        moshi_smha_t * attn,
        moshi_smha_state_t * state,
        ggml_tensor * indices,
        ggml_tensor * query,
        ggml_tensor * key,
        /*ggml_tensor * value,*/
        ggml_tensor * attn_bias = NULL,
        timestep_embedding_t * tsemb = NULL
    ) {
    CAPTURE_GROUP( "multihead_attention" );

    assert( attn->weights_per_step_schedule.size() == 0 );
    assert( attn->in_projs.size() == 1 );
    assert( attn->out_projs.size() == 1 );

    int T = (int) query->ne[1];
    int H = attn->num_heads;

    ggml_tensor * q;
    ggml_tensor * k;
    ggml_tensor * v;

    if ( attn->cross_attention ) {
        //assert len(attn->in_projs) == 1
        auto in_proj = attn->in_projs[0];
        //assert in_proj.bias is None
        //assert isinstance(in_proj, nn.Linear)
        int dim = (int) in_proj->weight->ne[1] / 3;
        //q = nn.functional.linear(query, in_proj.weight[:dim])
        //q = rearrange(q, "b t (h d) -> b h t d", h=attn->num_heads)
        q = torch_nn_linear_view( ctx, in_proj, 0, dim, query );

        k = state->k_cross;
        v = state->v_cross;
    } else {
        auto projected = torch_nn_linear( ctx, attn->in_projs[0], query );

        //q, k, v = rearrange(
        //    projected, "b t (p h d) -> p b h t d", p=3, h=attn->num_heads
        //)

        q = ggml_view_3d( ctx, projected,
            projected->ne[0] / 3,
            projected->ne[1],
            projected->ne[2],
            projected->nb[1],
            projected->nb[2],
            0 );
        q = ggml_cont( ctx, q );

        k = ggml_view_3d( ctx, projected,
            projected->ne[0] / 3,
            projected->ne[1],
            projected->ne[2],
            projected->nb[1],
            projected->nb[2],
            projected->nb[1] / 3 );
        // b t (h d) -> b t h d
        k = ggml_cont( ctx, k );
        k = ggml_reshape_4d( ctx, k,
            k->ne[0] / H,
            H,
            k->ne[1],
            k->ne[2] );
        // b t h d -> b h t d
        k = ggml_permute( ctx, k, 0, 2, 1, 3 );

        v = ggml_view_3d( ctx, projected,
            projected->ne[0] / 3,
            projected->ne[1],
            projected->ne[2],
            projected->nb[1],
            projected->nb[2],
            projected->nb[1] * 2 / 3 );
        // b t (h d) -> b t h d
        v = ggml_cont( ctx, v );
        v = ggml_reshape_4d( ctx, v,
            v->ne[0] / H,
            H,
            v->ne[1],
            v->ne[2] );
        // b t h d -> b h t d
        v = ggml_permute( ctx, v, 0, 2, 1, 3 );
    }

    // b t (h d) -> b t h d
    q = ggml_reshape_4d( ctx, q,
        q->ne[0] / H,
        H,
        q->ne[1],
        q->ne[2] );
    // b t h d -> b h t d
    q = ggml_permute( ctx, q, 0, 2, 1, 3 );

    if ( attn->rope_max_period ) {
        std::tie(q, k) = moshi_apply_rope( ctx, q, k, tsemb, false );
    }

    if ( attn->causal && ! attn->cross_attention ) {
        assert( k->ne[1] == T );

        std::tie( k, v ) = moshi_kv_cache_insert_kv( ctx,
            state->kv_cache->keys, state->kv_cache->values,
            indices, k, v );
    }

    assert( attn_bias || ! attn->causal );

    //x = nn.functional.scaled_dot_product_attention(q, k, v, attn_bias, dropout_p=0.0)
    auto x = torch_nn_functional_scaled_dot_product_attention_custom( ctx,
        q, k, v, attn_bias );//, dropout_p=0.0);

    //x = rearrange(x, "b h t d -> b t (h d)")
    // b h t d -> b t h d
    auto x2 = ggml_cont( ctx, ggml_permute( ctx, x, 0, 2 ,1 ,3 ) );
    // b t h d -> b t (h d)
    x = ggml_reshape_3d( ctx, x2,
        x2->ne[0] * x2->ne[1],
        x2->ne[2],
        x2->ne[3] );

    x = torch_nn_linear( ctx, attn->out_projs[0], x );

    return x;
}

ggml_tensor * moshi_streaming_multihead_cross_attention(
        GraphContext & ctx,
        moshi_smha_t * attn,
        moshi_smha_state_t * state,
        ggml_tensor * query
    ) {
    CAPTURE_GROUP( "multihead_cross_attention" );

    int T = (int) query->ne[1];
    int H = attn->num_heads;

    assert( attn->cross_attention );
    assert( ! attn->causal );
    assert( attn->in_projs.size() == 1 );
    assert( attn->out_projs.size() == 1 );

    auto in_proj = attn->in_projs[0];
    int dim = (int) in_proj->weight->ne[1] / 3;
    auto q = torch_nn_linear_view( ctx, in_proj, 0, dim, query );

    auto k = state->k_cross;
    auto v = state->v_cross;

    // b t (h d) -> b t h d
    q = ggml_reshape_4d( ctx, q,
        q->ne[0] / H,
        H,
        q->ne[1],
        q->ne[2] );
    // b t h d -> b h t d
    q = ggml_permute( ctx, q, 0, 2, 1, 3 );

    //x = nn.functional.scaled_dot_product_attention(q, k, v, attn_bias, dropout_p=0.0)
    auto x = torch_nn_functional_scaled_dot_product_attention_custom( ctx,
        q, k, v, NULL );//, dropout_p=0.0);

    //x = rearrange(x, "b h t d -> b t (h d)")
    // b h t d -> b t h d
    auto x2 = ggml_cont( ctx, ggml_permute( ctx, x, 0, 2 ,1 ,3 ) );
    // b t h d -> b t (h d)
    x = ggml_reshape_3d( ctx, x2,
        x2->ne[0] * x2->ne[1],
        x2->ne[2],
        x2->ne[3] );

    x = torch_nn_linear( ctx, attn->out_projs[0], x );

    return x;
}

void get_weights( WeightLoader * loader, std::string path, moshi_smha_t * attn ) {
    if ( loader->is_gguf ) {
        for ( size_t i = 0; i < attn->in_projs.size(); i++ ) {
            std::string name = path + "in_projs." + std::to_string(i) + ".weight";
            attn->in_projs[i]->weight = loader->get_tensor( name );
            assert( attn->in_projs[i]->weight );
            attn->in_projs[i]->bias = NULL;
        }
        for ( size_t i = 0; i < attn->out_projs.size(); i++ ) {
            std::string name = path + "out_projs." + std::to_string(i) + ".weight";
            attn->out_projs[i]->weight = loader->get_tensor( name );
            assert( attn->out_projs[i]->weight );
            attn->out_projs[i]->bias = NULL;
        }
        return;
    }
    {
        auto st = loader->find( path + "in_proj_weight" );
        assert( st );
        ggml_type dtype = safetensor_get_type( st->dtype );
        ggml_type dst_dtype = loader->quantize? loader->qtype : dtype;
        int n_dims = 2;
        GGML_NE ne( st->shape[1], st->shape[0] / attn->in_projs.size() );
        for ( size_t i = 0; i < attn->in_projs.size(); i++ ) {
            loader->add_alloc( &attn->in_projs[i]->weight, n_dims, ne, dst_dtype,
                path + "in_projs." + std::to_string(i) + ".weight" );
            attn->in_projs[i]->bias = NULL;
        }
        // queue initialization
        loader->add_init( [ attn, st, dtype, dst_dtype ]( WeightLoader * loader ) {
            auto & scratch_ctx = *loader->scratch;
            auto in_proj_weight = scratch_ctx.load( loader->stf, st );
            int64_t ne0 = in_proj_weight->ne[0];
            int64_t ne1 = in_proj_weight->ne[1] / (int64_t)attn->in_projs.size();
            int64_t nb1 = in_proj_weight->nb[1];
            for ( size_t i = 0; i < attn->in_projs.size(); i++ ) {
                auto weight = attn->in_projs[i]->weight;
                assert( weight );
                auto view = ggml_view_2d( scratch_ctx, in_proj_weight,
                    ne0, ne1, nb1, i * ne1 * nb1 );
                if ( dst_dtype == dtype ) {
                    scratch_ctx.build_forward_expand( view, weight );
                } else {
                    auto cast = ggml_cast( scratch_ctx, view, dst_dtype );
                    scratch_ctx.build_forward_expand( cast, weight );
                }
            }
            scratch_ctx.compute();
        } );
    }

    {
        auto st = loader->find( path + "out_proj.weight" );
        assert( st );
        ggml_type dtype = safetensor_get_type( st->dtype );
        ggml_type dst_dtype = loader->quantize? loader->qtype : dtype;
        int n_dims = 2;
        GGML_NE ne( st->shape[1], st->shape[0] / attn->out_projs.size() );
        for ( size_t i = 0; i < attn->out_projs.size(); i++ ) {
            loader->add_alloc( &attn->out_projs[i]->weight, n_dims, ne, dst_dtype,
                path + "out_projs." + std::to_string(i) + ".weight" );
            attn->out_projs[i]->bias = NULL;
        }
        // queue initialization
        loader->add_init( [ attn, st, dtype, dst_dtype ]( WeightLoader * loader ) {
            auto & scratch_ctx = *loader->scratch;
            auto out_proj_weight = scratch_ctx.load( loader->stf, st );
            int64_t ne0 = out_proj_weight->ne[0];
            int64_t ne1 = out_proj_weight->ne[1] / (int64_t)attn->out_projs.size();
            int64_t nb1 = out_proj_weight->nb[1];
            for ( size_t i = 0; i < attn->out_projs.size(); i++ ) {
                auto weight = attn->out_projs[i]->weight;
                assert( weight );
                auto view = ggml_view_2d( scratch_ctx, out_proj_weight,
                    ne0, ne1, nb1, i * ne1 * nb1 );
                if ( dst_dtype == dtype ) {
                    scratch_ctx.build_forward_expand( view, weight );
                } else {
                    auto cast = ggml_cast( scratch_ctx, view, dst_dtype );
                    scratch_ctx.build_forward_expand( cast, weight );
                }
            }
            scratch_ctx.compute();
        } );
    }
}

/*************************************************************\
 *  moshi.modules.transformer.StreamingTransformerLayer
 *
 * location in models:
 * lm.transformer.layers.*
 * lm.depformer.layers.*
 * mimi.decoder_transformer.transformer.layers.*
\*************************************************************/

struct moshi_streaming_transformer_layer_t {
    own_ptr<moshi_rms_norm_t> norm1_rms;
    own_ptr<torch_nn_layer_norm_t> norm1;

    own_ptr<moshi_smha_t> self_attn;

    own_ptr<moshi_layer_scale_t> layer_scale_1;

    own_ptr<torch_nn_layer_norm_t> norm_cross;
    own_ptr<moshi_smha_t> cross_attention;

    own_ptr<moshi_rms_norm_t> norm2_rms;
    own_ptr<torch_nn_layer_norm_t> norm2;

    //int weights_per_step; this is asserted the same as size of the schedule
    std::vector<int> weights_per_step_schedule;
    own_ptr_vector<moshi_activation_gating_t> gating;
    own_ptr<torch_nn_linear_t> linear1;
    own_ptr<torch_nn_linear_t> linear2;

    own_ptr<moshi_layer_scale_t> layer_scale_2;
};

struct moshi_streaming_transformer_layer_state_t {
    own_ptr<moshi_smha_state_t> self_attn;
    own_ptr<moshi_smha_state_t> cross_attention;
};

moshi_streaming_transformer_layer_state_t * moshi_streaming_transformer_layer_state(
        StateContext * state_ctx,
        moshi_streaming_transformer_layer_t * layer,
        ggml_tensor * k_cross ) {
    auto states = new moshi_streaming_transformer_layer_state_t;
    states->self_attn = moshi_smha_state( state_ctx, layer->self_attn, NULL );
    if ( layer->cross_attention )
        states->cross_attention = moshi_smha_state( state_ctx,
			layer->cross_attention, k_cross );
    else
        states->cross_attention = NULL;
    return states;
}

void init( ScratchContext * ctx, moshi_streaming_transformer_layer_state_t * states,
        moshi_streaming_transformer_layer_t * layer,
        ggml_tensor * condition_cross ) {
    init( ctx, states->self_attn, layer->self_attn, NULL );
    if ( states->cross_attention )
        init( ctx, states->cross_attention, layer->cross_attention, condition_cross );
}

ggml_tensor * moshi_streaming_transformer_layer(
        GraphContext & ctx,
        moshi_streaming_transformer_layer_t * layer,
        moshi_streaming_transformer_layer_state_t * states,
        ggml_tensor * indices,
        ggml_tensor * x,
        ggml_tensor * attn_bias,
        timestep_embedding_t * tsemb ) {

    //////////// x = layer._sa_block(x)

    ggml_tensor * nx;
    if ( layer->norm1_rms )
        nx = moshi_rms_norm(ctx, layer->norm1_rms, x);
    else
        nx = torch_nn_layer_norm(ctx, layer->norm1, x);

    auto update = moshi_streaming_multihead_attention( ctx,
        layer->self_attn, states->self_attn, indices,
        nx, nx, /*nx,*/ attn_bias, tsemb );

    if ( layer->layer_scale_1 ) {
        update = moshi_layer_scale( ctx, layer->layer_scale_1, update );
    }
    x = ggml_add( ctx, x, update );

    if ( layer->cross_attention ) {
        nx = torch_nn_layer_norm( ctx, layer->norm_cross, x );

        update = moshi_streaming_multihead_cross_attention( ctx,
            layer->cross_attention, states->cross_attention, nx );

        x = ggml_add( ctx, x, update );
    }

    //////////// x = layer._ff_block(x)

    if ( layer->norm2_rms )
        nx = moshi_rms_norm( ctx, layer->norm2_rms, x );
    else
        nx = torch_nn_layer_norm( ctx, layer->norm2, x );

    assert( layer->gating.size() <= 1 && layer->weights_per_step_schedule.size() == 0 );
    if ( ! layer->gating.size() ) {
        //linear1_r = layer.linear1(nx)
        auto linear1_r = torch_nn_linear( ctx, layer->linear1, nx );

        auto activated = ggml_gelu( ctx, linear1_r );

        //update = layer.linear2(activated)
        update = torch_nn_linear( ctx, layer->linear2, activated );
    } else {
        update = moshi_activation_gating( ctx, layer->gating[0], nx );
    }

    if ( layer->layer_scale_2 ) {
        update = moshi_layer_scale( ctx, layer->layer_scale_2, update );
    }
    x = ggml_add( ctx, x, update );

    return x;
}

ggml_tensor * moshi_streaming_transformer_layer(
        GraphContext & ctx,
        moshi_streaming_transformer_layer_t * layer,
        moshi_streaming_transformer_layer_state_t * states,
        ggml_tensor * indices,
        int offset,
        ggml_tensor * x,
        ggml_tensor * attn_bias,
        timestep_embedding_t * tsemb ) {

    //////////// x = layer._sa_block(x)

    ggml_tensor * nx;
    if ( layer->norm1_rms )
        nx = moshi_rms_norm(ctx, layer->norm1_rms, x);
    else
        nx = torch_nn_layer_norm(ctx, layer->norm1, x);

    auto update = moshi_streaming_multihead_attention( ctx,
        layer->self_attn, states->self_attn, indices, offset,
        nx, nx, /*nx,*/ attn_bias, tsemb );

    if ( layer->layer_scale_1 ) {
        update = moshi_layer_scale( ctx, layer->layer_scale_1, update );
    }
    x = ggml_add( ctx, x, update );

    if ( layer->cross_attention ) {
        nx = torch_nn_layer_norm( ctx, layer->norm_cross, x );

        update = moshi_streaming_multihead_cross_attention( ctx,
            layer->cross_attention, states->cross_attention,
            nx );

        x = ggml_add( ctx, x, update );
    }

    //////////// x = layer._ff_block(x)

    if ( layer->norm2_rms )
        nx = moshi_rms_norm( ctx, layer->norm2_rms, x );
    else
        nx = torch_nn_layer_norm( ctx, layer->norm2, x );

    if ( ! layer->gating.size() ) {
        //linear1_r = layer.linear1(nx)
        auto linear1_r = torch_nn_linear( ctx, layer->linear1, nx );

        auto activated = ggml_gelu( ctx, linear1_r );

        //update = layer.linear2(activated)
        update = torch_nn_linear( ctx, layer->linear2, activated );
    } else if ( layer->gating.size() > 1 || layer->weights_per_step_schedule.size() ) {
        update = moshi_apply_weights_per_step_gating( ctx,
            layer->gating, layer->weights_per_step_schedule,
            nx, offset );
    } else {
        update = moshi_activation_gating( ctx, layer->gating[0], nx );
    }

    if ( layer->layer_scale_2 ) {
        update = moshi_layer_scale( ctx, layer->layer_scale_2, update );
    }
    x = ggml_add( ctx, x, update );

    return x;
}


void get_weights( WeightLoader * loader, std::string path,
        moshi_streaming_transformer_layer_t * layer ) {
    if ( layer->norm1_rms )
        get_weights( loader, path + "norm1.", layer->norm1_rms );
    else
        get_weights( loader, path + "norm1.", layer->norm1 );

    get_weights( loader, path + "self_attn.", layer->self_attn );

    if ( layer->layer_scale_1 )
        get_weights( loader, path + "layer_scale_1.", layer->layer_scale_1 );

    if ( layer->cross_attention ) {
        get_weights( loader, path + "norm_cross.", layer->norm_cross );
        get_weights( loader, path + "cross_attention.", layer->cross_attention );
    }

    if ( layer->norm2_rms )
        get_weights( loader, path + "norm2.", layer->norm2_rms );
    else
        get_weights( loader, path + "norm2.", layer->norm2 );

    if ( layer->gating.size() ) {
        if ( layer->weights_per_step_schedule.size() || layer->gating.size() > 1 ) {
            for ( size_t i = 0; i < layer->gating.size(); i++ ) {
                get_weights( loader, path + "gating." + std::to_string(i) + ".",
                    layer->gating[i] );
            }
        } else {
            get_weights( loader, path + "gating.", layer->gating[0] );
        }
    } else {
        get_weights( loader, path + "linear1.", layer->linear1 );
        get_weights( loader, path + "linear2.", layer->linear2 );
    }

    if ( layer->layer_scale_2 )
        get_weights( loader, path + "layer_scale_2.", layer->layer_scale_2 );
}

/*************************************************************\
 *  moshi.modules.transformer.StreamingTransformerLayer
 *
 * location in models:
 * lm.transformer
 * lm.depformer
 * mimi.decoder_transformer.transformer
\*************************************************************/

struct moshi_streaming_transformer_t {
    int context;
    int weights_per_step;
    int capacity;
    int rope_max_period;
    int dim_per_head;
    own_ptr_vector<moshi_streaming_transformer_layer_t> layers;
    bias_pattern_t pattern;
};

struct moshi_streaming_transformer_graph_t {
    // TODO: decide if we keep this or free it
    GraphContext * ctx = NULL;
    ggml_tensor * attn_bias;
    ggml_tensor * offset;
    ggml_tensor * indices;
    ggml_tensor * x;
    ggml_tensor * result;
};

struct moshi_streaming_transformer_state_t {
    int offset;
    own_ptr_vector<moshi_streaming_transformer_layer_state_t> layers;
    moshi_streaming_transformer_graph_t graph;
};

moshi_streaming_transformer_state_t * moshi_streaming_transformer_state(
        StateContext * state_ctx,
        moshi_streaming_transformer_t * transformer,
        ggml_tensor * k_cross ) {
    auto states = new moshi_streaming_transformer_state_t;
    for (auto layer : transformer->layers) {
        states->layers.push_back(
            moshi_streaming_transformer_layer_state( state_ctx, layer, k_cross )
        );
    }
    return states;
}

void init( ScratchContext * ctx, moshi_streaming_transformer_state_t *states,
        moshi_streaming_transformer_t * m,
        ggml_tensor * condition_cross ) {
    states->offset = 0;
    for ( size_t idx = 0; idx < m->layers.size(); idx++ ) {
        init( ctx, states->layers[idx], m->layers[idx], condition_cross );
    }
}

ggml_tensor * moshi_streaming_transformer(
        GraphContext & ctx,
        moshi_streaming_transformer_t * m,
        moshi_streaming_transformer_state_t * states,
        int offset,
        ggml_tensor * attn_bias,
        timestep_embedding_t * tsemb,
        ggml_tensor * indices,
        ggml_tensor * x ) {

    for ( size_t idx = 0; idx < m->layers.size(); idx++ ) {
        CAPTURE_GROUP( "layer." + std::to_string(idx) );
        auto layer = m->layers[idx];
        auto layer_states = states->layers[idx];
        x = moshi_streaming_transformer_layer( ctx, layer, layer_states,
            indices, offset, x, attn_bias, tsemb
        );
    }

    return x;
}

ggml_tensor * moshi_streaming_transformer(
        GraphContext & ctx,
        moshi_streaming_transformer_t * m,
        moshi_streaming_transformer_state_t * states,
        ggml_tensor * attn_bias,
        timestep_embedding_t * tsemb,
        ggml_tensor * indices,
        ggml_tensor * x ) {

    for ( size_t idx = 0; idx < m->layers.size(); idx++ ) {
        CAPTURE_GROUP( "layer." + std::to_string(idx) );
        auto layer = m->layers[idx];
        auto layer_states = states->layers[idx];
        x = moshi_streaming_transformer_layer( ctx, layer, layer_states,
            indices, x, attn_bias, tsemb
        );
    }

    return x;
}

ggml_tensor * moshi_streaming_transformer(
        GraphContext & ctx,
        moshi_streaming_transformer_t * m,
        moshi_streaming_transformer_state_t * states,
        ggml_tensor * x ) {

    auto rope_max_period = m->rope_max_period;
    int64_t D = m->dim_per_head;
    int capacity = m->capacity;

    int offset = states->offset;

    int64_t T = x->ne[1];

    auto attn_bias = get_attn_bias( ctx, &m->pattern, capacity, T, offset );

    timestep_embedding_t tsemb = { NULL, NULL };
    if ( rope_max_period ) {
        auto toffset = ctx.constant( (float)offset );
        moshi_get_timestep_embedding( ctx, (int)T, (int)D, toffset, rope_max_period, tsemb );
    }

    std::vector<int> offsets(T);
    for (int i = 0; i < offsets.size(); i++)
        offsets[i] = (offset + i) % capacity;
    NE ne = { T, 1, 1, 1 };
    auto indices = ctx.input( ne, offsets );

    x = moshi_streaming_transformer( ctx, m, states, offset, attn_bias, &tsemb, indices, x );

    states->offset += (int)T;

    return x;
}

ggml_tensor * moshi_streaming_transformer_graph_build(
        GraphContext & gctx,
        moshi_streaming_transformer_t * m,
        moshi_streaming_transformer_state_t * states,
        ggml_tensor * x ) {

    states->graph.ctx = &gctx;

    auto rope_max_period = m->rope_max_period;
    int64_t D = m->dim_per_head;
    int capacity = m->capacity;

    int64_t T = x->ne[1];

    // attn_bias
    create_bias_pattern( gctx.backend, m->pattern, capacity, (int) T, 0, -INFINITY );
    states->graph.attn_bias = gctx.new_tensor(
        GGML_TYPE_F32, GGML_NE( capacity, T ) );

    // offset for timestep_embedding
    timestep_embedding_t tsemb = { NULL, NULL };
    if ( rope_max_period ) {
        states->graph.offset = gctx.new_tensor(
            GGML_TYPE_F32, GGML_NE( 1 ) );
        moshi_get_timestep_embedding( gctx, (int)T, (int)D,
            states->graph.offset, rope_max_period, tsemb );
    } else {
        states->graph.offset = NULL;
    }

    // indices
    states->graph.indices = gctx.new_tensor(
        GGML_TYPE_I32, GGML_NE( T ) );

    return moshi_streaming_transformer( gctx,
        m, states,
        states->graph.attn_bias,
        &tsemb,
        states->graph.indices,
        x );
}

void moshi_streaming_transformer_graph_step(
        ScratchContext & ctx,
        moshi_streaming_transformer_t * m,
        moshi_streaming_transformer_state_t * states,
        int T ) {

    auto rope_max_period = m->rope_max_period;
    int64_t D = m->dim_per_head;
    int capacity = m->capacity;

    int offset = states->offset;
    states->offset += T;

    // update attn_bias
    if ( states->graph.attn_bias ) {
        auto attn_bias = bias_pattern_index( ctx, m->pattern, offset );
        attn_bias = ggml_cpy( ctx, attn_bias, states->graph.attn_bias );
        ctx.build_forward_expand( attn_bias );
    }

    // update offset
    if ( states->graph.offset ) {
        states->graph.ctx->tensor_set( states->graph.offset, (float)offset );
    }

    // update indices
    std::vector<int32_t> offsets( ggml_nelements( states->graph.indices ) );
    for (int i = 0; i < offsets.size(); i++)
        offsets[i] = (offset + i) % capacity;
    states->graph.ctx->tensor_set( states->graph.indices, offsets );
}

ggml_tensor * moshi_streaming_transformer_graph(
        ScratchContext & ctx,
        moshi_streaming_transformer_t * m,
        moshi_streaming_transformer_state_t * states,
        ggml_tensor * x ) {

    int64_t T = x->ne[1];

    if ( ! states->graph.ctx ) {
        // create graph
        states->graph.ctx = new GraphContext( 256, ctx.backend );

        states->graph.x = ggml_dup_tensor( *states->graph.ctx, x );

        auto result = moshi_streaming_transformer_graph_build(
            *states->graph.ctx, m, states, states->graph.x );

        states->graph.result = ggml_dup_tensor( *states->graph.ctx, result );
        auto result_cpy = ggml_cpy( *states->graph.ctx, result, states->graph.result );
        states->graph.ctx->build_forward_expand( result_cpy );

        states->graph.ctx->alloc();
    }

    moshi_streaming_transformer_graph_step( ctx, m, states, (int)T );

    // cpy x
    auto x_cpy = ggml_cpy( ctx, x, states->graph.x );
    ctx.build_forward_expand( x_cpy );

    // copy inputs to transformer graph
    ctx.compute();

    // compute transformer
    states->graph.ctx->compute();

    // result is static and can be used by scratch ctx
    return states->graph.result;
}

void get_weights( WeightLoader * loader, std::string path,
        moshi_streaming_transformer_t * transformer ) {
    for ( size_t i = 0; i < transformer->layers.size(); i++ ) {
        get_weights( loader, path + "layers." + std::to_string(i) + ".", transformer->layers[i] );
    }
}

/*************************************************************\
 *  moshi.modules.transformer.ProjectedTransformer
 *
 * location in models:
 * mimi.decoder_transformer
\*************************************************************/

ggml_tensor * moshi_projected_transformer(
        ScratchContext & ctx,
        moshi_streaming_transformer_state_t * states,
        moshi_streaming_transformer_t * transformer,
        ggml_tensor * x ) {
    x = ggml_cont( ctx, ggml_transpose( ctx, x ) );
    auto z = moshi_streaming_transformer( ctx, transformer, states, x );
    auto y = ggml_transpose( ctx, z );
    return y;
}

ggml_tensor * moshi_projected_transformer_graph_build(
        GraphContext & ctx,
        moshi_streaming_transformer_state_t * states,
        moshi_streaming_transformer_t * transformer,
        ggml_tensor * x ) {
    x = ggml_cont( ctx, ggml_transpose( ctx, x ) );
    auto z = moshi_streaming_transformer_graph_build( ctx, transformer, states, x );
    auto y = ggml_transpose( ctx, z );
    return y;
}

void moshi_projected_transformer_graph_step(
        ScratchContext & ctx,
        moshi_streaming_transformer_state_t * states,
        moshi_streaming_transformer_t * transformer,
        int T ) {
    moshi_streaming_transformer_graph_step( ctx, transformer, states, T );
}

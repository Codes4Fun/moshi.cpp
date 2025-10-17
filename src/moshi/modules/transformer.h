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
    assert( loader->fetch( &norm->alpha, path + "alpha", (void*)ggml_rms_norm ) );
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
    assert( loader->fetch( &scale->scale, path + "scale", (void*)ggml_mul ) );
}

/*****************************************\
 * moshi.modules.transformer.LayerScale
\*****************************************/

ggml_tensor * moshi_apply_weights_per_step_linear(
        ggml_context * ctx,
        std::vector<torch_nn_linear_t*> & modules,
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

    int T = x->ne[1];
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
        std::vector<moshi_activation_gating_t*> & modules,
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

    int T = x->ne[1];
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
    int end_offset;
};

void moshi_kv_cache_state(
        StateContext * state_ctx,
        int dim_per_head,
        int capacity,
        int num_heads,
        int batch_size,
        moshi_kv_cache_state_t * &states ) {
    states = new moshi_kv_cache_state_t;
    NE ne = { dim_per_head, capacity, num_heads, batch_size };
    state_ctx->fill( ne, 0.f, &states->keys );
    state_ctx->fill( ne, 0.f, &states->values );
}

void init( moshi_kv_cache_state_t * state ) {
    state->end_offset = 0;
}

std::tuple<ggml_tensor*,ggml_tensor*> moshi_kv_cache_insert_kv(
        ggml_context * ctx,
        ggml_tensor * keys,
        ggml_tensor * values,
        int index,
        ggml_tensor * k,
        ggml_tensor * v ) {
    int T = k->ne[1];
    int capacity = keys->ne[1];
    index = index % capacity;
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
    cache_0_0 = ggml_cpy( ctx, k, cache_0_0 );
    keys = ggml_view_4d( ctx, cache_0_0,
        keys->ne[0], // D
        keys->ne[1], // context
        keys->ne[2], // H
        keys->ne[3], // B
        keys->nb[1],
        keys->nb[2],
        keys->nb[3],
        -keys->nb[1] * index
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
    cache_1_0 = ggml_cpy( ctx, v, cache_1_0 );
    values = ggml_view_4d( ctx, cache_1_0,
        values->ne[0], // D
        values->ne[1], // context
        values->ne[2], // H
        values->ne[3], // B
        values->nb[1],
        values->nb[2],
        values->nb[3],
        -values->nb[1] * index
    );
    return std::make_tuple( keys, values );
}

ggml_tensor * moshi_kv_cache_get_positions(
        ScratchContext & ctx,
        int end_offset,
        int capacity ) {
    auto indexes = ctx.arange( 0, capacity, 1 );

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
    positions = ggml_sub( ctx, positions, ggml_scale( ctx, capacity_mask, capacity ) );

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
    std::vector<torch_nn_linear_t*> in_projs;
    std::vector<torch_nn_linear_t*> out_projs;
};

struct moshi_smha_state_t {
    int offset;
    bool cache_ready = false;
    ggml_tensor * k_cross;
    ggml_tensor * v_cross;
    moshi_kv_cache_state_t * kv_cache;
};

moshi_smha_state_t * moshi_smha_state( StateContext * state_ctx,
        moshi_smha_t * attn, ggml_tensor * k_cross ) {
    auto state = new moshi_smha_state_t;
    int dim_per_head = attn->embed_dim / attn->num_heads;
    int num_heads = attn->num_heads;
    if ( ! attn->cross_attention ) {
        int capacity = attn->context? attn->context : attn->weights_per_step;
        int batch_size = 1;
        moshi_kv_cache_state( state_ctx, dim_per_head, capacity, num_heads, batch_size,
            state->kv_cache );
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

void init( moshi_smha_state_t * state ) {
    state->offset = 0;
    state->cache_ready = false;
    if ( state->kv_cache )
        init( state->kv_cache );
}

void cache_kv( ggml_context * ctx, moshi_smha_state_t * state,
        ggml_tensor * &k_cross, ggml_tensor * &v_cross ) {
    assert( state->k_cross );
    k_cross = ggml_cpy( ctx, k_cross, state->k_cross );
    v_cross = ggml_cpy( ctx, v_cross, state->v_cross );
    state->cache_ready = true;
}

// utility that can be done once and shared across layers
ggml_tensor * calculate_attn_bias( ScratchContext & ctx, moshi_smha_t * attn,
        int64_t T, int64_t noffset ) {
    if ( ! attn->causal )
        return NULL;
    ggml_tensor * pos_k;
    assert( !attn->cross_attention );
    int capacity = attn->context? attn->context : attn->weights_per_step;
    pos_k = moshi_kv_cache_get_positions( ctx, noffset + T, capacity );

    auto offset = ctx.constant( (float)noffset );
    auto pos_q = ggml_add( ctx, ctx.arange( 0, T, 1 ), offset );
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
        ScratchContext & ctx,
        moshi_smha_t * attn,
        moshi_smha_state_t * state,
        ggml_tensor * query,
        ggml_tensor * key,
        ggml_tensor * value,
        ggml_tensor * attn_bias = NULL ) {
    CAPTURE_GROUP( "multihead_attention" );

    int T = query->ne[1];
    int H = attn->num_heads;

    ggml_tensor * offset = NULL;
    if ( attn->rope_max_period || attn->causal )
        offset = ctx.constant( (float)state->offset );

    ggml_tensor * q;
    ggml_tensor * k;
    ggml_tensor * v;

    if ( attn->cross_attention ) {
        //assert len(attn->in_projs) == 1
        auto in_proj = attn->in_projs[0];
        //assert in_proj.bias is None
        //assert isinstance(in_proj, nn.Linear)
        int dim = in_proj->weight->ne[1] / 3;
        int dim2 = dim * 2;
        //q = nn.functional.linear(query, in_proj.weight[:dim])
        //q = rearrange(q, "b t (h d) -> b h t d", h=attn->num_heads)
        q = torch_nn_linear_view( ctx, in_proj, 0, dim, query );

        if ( state->cache_ready ) {
            k = state->k_cross;
            v = state->v_cross;
        } else {
            //kv = nn.functional.linear(key, in_proj.weight[dim:])
            auto kv = torch_nn_linear_view( ctx, in_proj, dim, dim2, key );

            //k, v = rearrange(kv, "b t (p h d) -> p b h t d", p=2, h=attn->num_heads)
            k = ggml_view_3d( ctx, kv,
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

            v = ggml_view_3d( ctx, kv,
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

            if ( attn->cache_cross_attention ) {
                cache_kv( ctx, state, k, v );
            }
        }
    } else {
        auto projected = moshi_apply_weights_per_step_linear( ctx,
            attn->in_projs, attn->weights_per_step_schedule,
            query, state->offset );

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
        std::tie(q, k) = moshi_apply_rope( ctx, q, k, offset,
            attn->rope_max_period, false );
    }

    if ( attn->causal && ! attn->cross_attention ) {
        assert( k->ne[1] == T );

        std::tie( k, v ) = moshi_kv_cache_insert_kv( ctx,
            state->kv_cache->keys, state->kv_cache->values,
            state->offset,
            k, v );
    }

    if ( ! attn_bias )
        attn_bias = calculate_attn_bias( ctx, attn, T, state->offset );

    //x = nn.functional.scaled_dot_product_attention(q, k, v, attn_bias, dropout_p=0.0)
    auto x = torch_nn_functional_scaled_dot_product_attention( ctx,
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
        x, state->offset );

    state->offset += T;
    return x;
}

void get_weights( WeightLoader * loader, std::string path, moshi_smha_t * attn ) {
    WeightLoader::bindings_t in_projs_bindings;
    for ( size_t i = 0; i < attn->in_projs.size(); i++ ) {
        in_projs_bindings.push_back({
            &attn->in_projs[i]->weight,
            path + "in_projs." + std::to_string(i) + ".weight"
        });
    }
    assert( loader->fetch(in_projs_bindings, [path, attn]( WeightLoader * loader ) {
        auto st = loader->find( path + "in_proj_weight" );
        if ( ! st )
            return false;
        ggml_type dtype = safetensor_get_type( st->dtype );
        int n_dims = 2;
        GGML_NE ne( st->shape[1], st->shape[0] / attn->in_projs.size() );
        for ( size_t i = 0; i < attn->in_projs.size(); i++ ) {
            loader->add_alloc( &attn->in_projs[i]->weight, n_dims, ne, dtype,
                path + "in_projs." + std::to_string(i) + ".weight" );
            attn->in_projs[i]->bias = NULL;
        }
        // queue initialization
        loader->add_init( [ attn, st ]( WeightLoader * loader ) {
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
                scratch_ctx.build_forward_expand( view, weight );
            }
            scratch_ctx.compute();
        } );
        return true;
    } ) );

    WeightLoader::bindings_t out_projs_bindings;
    for ( size_t i = 0; i < attn->out_projs.size(); i++ ) {
        out_projs_bindings.push_back({
            &attn->out_projs[i]->weight,
            path + "out_projs." + std::to_string(i) + ".weight"
        });
    }
    assert( loader->fetch(out_projs_bindings, [path, attn]( WeightLoader * loader ) {
        auto st = loader->find( path + "out_proj.weight" );
        if ( ! st )
            return false;
        ggml_type dtype = safetensor_get_type( st->dtype );
        int n_dims = 2;
        GGML_NE ne( st->shape[1], st->shape[0] / attn->out_projs.size() );
        for ( size_t i = 0; i < attn->out_projs.size(); i++ ) {
            loader->add_alloc( &attn->out_projs[i]->weight, n_dims, ne, dtype,
                path + "out_projs." + std::to_string(i) + ".weight" );
            attn->out_projs[i]->bias = NULL;
        }
        // queue initialization
        loader->add_init( [ attn, st ]( WeightLoader * loader ) {
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
                scratch_ctx.build_forward_expand( view, weight );
            }
            scratch_ctx.compute();
        } );
        return true;
    } ) );
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
    moshi_rms_norm_t * norm1_rms;
    torch_nn_layer_norm_t * norm1;

    moshi_smha_t * self_attn;

    moshi_layer_scale_t * layer_scale_1;

    torch_nn_layer_norm_t * norm_cross;
    moshi_smha_t * cross_attention;

    moshi_rms_norm_t * norm2_rms;
    torch_nn_layer_norm_t * norm2;

    //int weights_per_step; this is asserted the same as size of the schedule
    std::vector<int> weights_per_step_schedule;
    std::vector<moshi_activation_gating_t*> gating;
    torch_nn_linear_t * linear1;
    torch_nn_linear_t * linear2;

    moshi_layer_scale_t * layer_scale_2;
};

struct moshi_streaming_transformer_layer_state_t {
    int offset;
    moshi_smha_state_t * self_attn;
    moshi_smha_state_t * cross_attention;
};

moshi_streaming_transformer_layer_state_t * moshi_streaming_transformer_layer_state(
        StateContext * state_ctx,
        moshi_streaming_transformer_layer_t * layer,
        ggml_tensor * k_cross ) {
    auto states = new moshi_streaming_transformer_layer_state_t;
    states->offset = 0;
    states->self_attn = moshi_smha_state( state_ctx, layer->self_attn, NULL );
    if ( layer->cross_attention )
        states->cross_attention = moshi_smha_state( state_ctx,
			layer->cross_attention, k_cross );
    else
        states->cross_attention = NULL;
    return states;
}

void init( moshi_streaming_transformer_layer_state_t * states ) {
    states->offset = 0;
    init( states->self_attn );
    if ( states->cross_attention )
        init( states->cross_attention );
}

ggml_tensor * moshi_streaming_transformer_layer(
        ScratchContext & ctx,
        moshi_streaming_transformer_layer_t * layer,
        moshi_streaming_transformer_layer_state_t * states,
        ggml_tensor * x,
        ggml_tensor * attn_bias,
        ggml_tensor * cross_attention_src = NULL) {

    //////////// x = layer._sa_block(x)

    ggml_tensor * nx;
    if ( layer->norm1_rms )
        nx = moshi_rms_norm(ctx, layer->norm1_rms, x);
    else
        nx = torch_nn_layer_norm(ctx, layer->norm1, x);

    auto update = moshi_streaming_multihead_attention( ctx,
        layer->self_attn, states->self_attn,
        nx, nx, nx, attn_bias );

    if ( layer->layer_scale_1 ) {
        update = moshi_layer_scale( ctx, layer->layer_scale_1, update );
    }
    x = ggml_add( ctx, x, update );

    if ( layer->cross_attention ) {
        nx = torch_nn_layer_norm( ctx, layer->norm_cross, x );

        // queries are from src, keys and values from cross_attention_src.
        //update = layer.cross_attention(nx, cross_attention_src, cross_attention_src)
        update = moshi_streaming_multihead_attention( ctx,
            layer->cross_attention, states->cross_attention,
            nx, cross_attention_src, cross_attention_src, NULL );

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
    } else if ( layer->weights_per_step_schedule.size() ) {
        update = moshi_apply_weights_per_step_gating( ctx,
            layer->gating, layer->weights_per_step_schedule,
            nx, states->offset );
    } else {
        update = moshi_activation_gating( ctx, layer->gating[0], nx );
    }

    if ( layer->layer_scale_2 ) {
        update = moshi_layer_scale( ctx, layer->layer_scale_2, update );
    }
    x = ggml_add( ctx, x, update );

    states->offset += x->ne[1];
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
        if ( layer->weights_per_step_schedule.size() ) {
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
    std::vector<moshi_streaming_transformer_layer_t*> layers;
};

struct moshi_streaming_transformer_state_t {
    std::vector<moshi_streaming_transformer_layer_state_t*> layers;
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

void init( moshi_streaming_transformer_state_t *states ) {
    for ( auto layer : states->layers )
        init( layer );
}

ggml_tensor * moshi_streaming_transformer(
        ScratchContext & ctx,
        moshi_streaming_transformer_t * m,
        moshi_streaming_transformer_state_t * states,
        ggml_tensor * x,
        ggml_tensor * cross_attention_src = NULL ) {
    //ProfileScope profile(transformer_us);

    // self_attn kv_cache capacity, to build k_pos once
    auto attn = m->layers[0]->self_attn;
    auto state = states->layers[0]->self_attn;
    int64_t T = x->ne[1];
    auto attn_bias = calculate_attn_bias( ctx, attn, T, state->offset );

    for ( size_t idx = 0; idx < m->layers.size(); idx++ ) {
        CAPTURE_GROUP( "layer." + std::to_string(idx) );
        auto layer = m->layers[idx];
        auto layer_states = states->layers[idx];
        x = moshi_streaming_transformer_layer( ctx, layer, layer_states, x,
                attn_bias, cross_attention_src );
    }

    return x;
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


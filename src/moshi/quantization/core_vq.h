#pragma once

/*****************************************************************************\
 *   moshi.quantization.core_vq.EuclideanCodebook
 * 
 * model locations:
 *   mimi.quantizer.rvq_first.vq.layers.*._codebook
 *   mimi.quantizer.rvq_rest.vq.layers.*._codebook
\*****************************************************************************/

struct moshi_EuclideanCodebook_t {
    ggml_tensor * embedding;
};

ggml_tensor * moshi_EuclideanCodebook_decode(
        ggml_context * ctx,
        moshi_EuclideanCodebook_t * codebook,
        ggml_tensor * codes ) {
    /*
    Given a tensor of codes of shape `[*]`, returns a tensor of shape `[*, D]`,
    corresponding to the centroids associated to each code index.
    */
    assert( (codes->type == GGML_TYPE_I64 || codes->type == GGML_TYPE_I32) );
    return ggml_get_rows( ctx, codebook->embedding, ggml_cont(ctx, codes) );
}

ggml_tensor * moshi_EuclideanCodebook_encode(
        ScratchContext & ctx,
        moshi_EuclideanCodebook_t * codebook,
        ggml_tensor * x ) {
    /*
    Given a tensor `x` of shape `[*, D]`, returns a tensor of integer codes of shape `[*]`.
    The codes are defined as the indexes of the centroids nearest to each vector in `x`.
    */

    auto a = ggml_cont( ctx, x );
    auto b = codebook->embedding;
    auto ane1 = a->ne[1];
    auto bne1 = b->ne[1];
    a = ggml_reshape_3d( ctx, a, a->ne[0], 1, a->ne[1] );
    a = ggml_repeat_4d( ctx, a, a->ne[0], bne1, a->ne[2], 1 );
    a = ggml_reshape_3d( ctx, a, a->ne[0], a->ne[1] * a->ne[2], a->ne[3] );

    b = ggml_repeat_4d( ctx, b, b->ne[0], b->ne[1] * ane1, b->ne[2], b->ne[3] );

    auto c = ggml_sub( ctx, b, a );
    c = ggml_mul( ctx, c, c );
    c = ggml_sum_rows( ctx, c );
    c = ggml_reshape_3d( ctx, c, bne1, ane1, 1 );

    c = ggml_add( ctx, c, ctx.constant(1.f));
    c = ggml_div( ctx, ctx.fill(c->ne, 1.f), c );
    c = ggml_argmax( ctx, c );

    return c;
}

void get_weights( WeightLoader * loader, std::string path,
        moshi_EuclideanCodebook_t * codebook ) {
    WeightLoader::bindings_t bindings;
    bindings.push_back({ &codebook->embedding, path + "embedding" });
    assert( loader->fetch(bindings, [path, codebook] (WeightLoader * loader) {
        auto sum_st = loader->find( path + "embedding_sum" );
        if ( ! sum_st )
            return false;
        auto usage_st = loader->find( path + "cluster_usage" );
        if ( ! usage_st )
            return false;

        NE ne;
        int n_dims = safetensor_get_shape( sum_st, ne );
        loader->add_alloc( &codebook->embedding, n_dims, ne, GGML_TYPE_F32, path + "embedding" );

        loader->add_init( [ sum_st, usage_st, codebook ] ( WeightLoader * loader ) {
            auto & scratch_ctx = *loader->scratch;
            auto embedding_sum = scratch_ctx.load( loader->stf, sum_st );
            auto cluster_usage = scratch_ctx.load( loader->stf, usage_st );
            auto clamp = ggml_clamp( scratch_ctx, cluster_usage, 1e-5f, INFINITY );
            auto cont = ggml_cont( scratch_ctx, ggml_transpose(scratch_ctx, clamp) );
            auto embedding = ggml_div( scratch_ctx, embedding_sum, cont );
            scratch_ctx.build_forward_expand( embedding, codebook->embedding );
            scratch_ctx.compute();
        });

        return true;
    } ) );
}


/*****************************************************************************\
 *   moshi.quantization.core_vq.VectorQuantization
 * 
 * model locations:
 *   mimi.quantizer.rvq_first.vq.layers.*
 *   mimi.quantizer.rvq_rest.vq.layers.*
\*****************************************************************************/

struct moshi_vq_t {
    own_ptr<moshi_EuclideanCodebook_t> _codebook;
};

ggml_tensor * moshi_vq_decode(
        ggml_context * ctx,
        moshi_vq_t * vq,
        ggml_tensor * codes ) {
    /* Converts integer codes into quantized vectors. */
    auto quantized = moshi_EuclideanCodebook_decode( ctx, vq->_codebook, codes );
    quantized = ggml_permute( ctx, quantized, 1, 0, 2, 3 );
    quantized = ggml_cont( ctx, quantized );
    return quantized;
}

ggml_tensor * moshi_vq_encode(
        ScratchContext & ctx,
        moshi_vq_t * vq,
        ggml_tensor * x ) {
    /* Encodes `x` into discrete integer codes. */
    //x = self._rearrange_input(x)
    x = ggml_permute( ctx, x, 1, 0, 2, 3 );
    // x = self.project_in(x) was identity
    auto codes = moshi_EuclideanCodebook_encode( ctx, vq->_codebook, x);
    return codes;
}

void get_weights( WeightLoader * loader, std::string path, moshi_vq_t * vq ) {
    get_weights( loader, path + "_codebook.", vq->_codebook );
}

/*****************************************************************************\
 *   moshi.quantization.core_vq.ResidualVectorQuantization
 * 
 * model locations:
 *   mimi.quantizer.rvq_first.vq
 *   mimi.quantizer.rvq_rest.vq
\*****************************************************************************/

struct moshi_residual_vq_t {
    own_ptr_vector<moshi_vq_t> layers;
};

ggml_tensor * moshi_residual_vq_decode(
        ggml_context * ctx,
        moshi_residual_vq_t * rvq,
        ggml_tensor * codes ) {
    /* Converts the integer codes into quantized vectors. */
    auto T =       codes->ne[0];
    auto B =       codes->ne[1];
    auto K =       codes->ne[2];
    auto Bstride = codes->nb[1];
    auto Kstride = codes->nb[2];

    ggml_tensor * quantized = NULL;
    for ( size_t idx = 0; idx < rvq->layers.size() && idx < (size_t)K; idx++ ) {
        auto layer = rvq->layers[idx];

        // select one K
        auto layer_codes = ggml_view_3d( ctx, codes,
            T, B, 1,
            Bstride, Kstride,
            Kstride * idx );

        auto decoded = moshi_vq_decode( ctx, layer, layer_codes );

        if (quantized)
            quantized = ggml_add( ctx, quantized, decoded );
        else
            quantized = decoded;
    }

    return quantized;
}

ggml_tensor * moshi_residual_vq_encode(
        ScratchContext & ctx,
        moshi_residual_vq_t * rvq,
        ggml_tensor * x,
        int n_q ) {
    auto residual = x;
    ggml_tensor * out_indices = NULL;
    if ( ! n_q )
        n_q = (int) rvq->layers.size();
    for ( int i = 0; i < n_q; i++ ) {
        auto layer = rvq->layers[i];
        auto indices = moshi_vq_encode( ctx, layer, residual );
        auto quantized = moshi_vq_decode( ctx, layer, indices );
        indices = ggml_cast( ctx, indices, GGML_TYPE_F32 );
        residual = ggml_sub( ctx, residual, quantized );
        // we don't have torch.stack, but concat on 3rd dimension is the same
        // dimensions being [T, B, K] with B batch as 1, K is our codes
        if ( ! out_indices )
            out_indices = indices;
        else
            out_indices = ggml_concat( ctx, out_indices, indices, 2 );
    }
    return out_indices;
}

void get_weights( WeightLoader * loader, std::string path, moshi_residual_vq_t * rvq ) {
    for ( size_t i = 0; i < rvq->layers.size(); i++ ) {
        get_weights( loader, path + "layers." + std::to_string(i) + ".", rvq->layers[i] );
    }
}

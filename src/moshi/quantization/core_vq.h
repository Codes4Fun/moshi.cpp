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
    return ggml_get_rows( ctx, codebook->embedding, codes );
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
            auto clamp = ggml_clamp( scratch_ctx, cluster_usage, 1e-5, INFINITY );
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
    moshi_EuclideanCodebook_t * _codebook;
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
    std::vector<moshi_vq_t*> layers;
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

void get_weights( WeightLoader * loader, std::string path, moshi_residual_vq_t * rvq ) {
    for ( size_t i = 0; i < rvq->layers.size(); i++ ) {
        get_weights( loader, path + "layers." + std::to_string(i) + ".", rvq->layers[i] );
    }
}

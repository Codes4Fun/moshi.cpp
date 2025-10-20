#pragma once

/*****************************************************************************\
 *   moshi.quantization.vq.ResidualVectorQuantizer
 * 
 * model locations:
 *   mimi.quantizer.rvq_first
 *   mimi.quantizer.rvq_rest
\*****************************************************************************/

struct moshi_rvq_t {
    moshi_residual_vq_t * vq;
    torch_nn_conv1d_t * output_proj;
};

ggml_tensor * moshi_rvq_decode(
        ggml_context * ctx,
        moshi_rvq_t * rvq,
        ggml_tensor * codes ) {
    /* Decode the given codes to the quantized representation. */
    // codes is [B, K, T], with T frames, K nb of codebooks, vq.decode expects [K, B, T].
    codes = ggml_permute( ctx, codes, 0, 2, 1, 3 );

    auto quantized = moshi_residual_vq_decode( ctx, rvq->vq, codes );
    if ( rvq->output_proj )
        quantized = torch_nn_conv1d( ctx, rvq->output_proj, quantized );
    return quantized;
}

void get_weights( WeightLoader * loader, std::string path, moshi_rvq_t * rvq ) {
    get_weights( loader, path + "vq.", rvq->vq );
    get_weights( loader, path + "output_proj.", rvq->output_proj );
}

/*****************************************************************************\
 *   moshi.quantization.vq.Splitrvq
 * 
 * model locations:
 *   mimi.quantizer
\*****************************************************************************/

struct moshi_split_rvq_t {
    int n_q_semantic;
    moshi_rvq_t * rvq_first;
    moshi_rvq_t * rvq_rest;
};

ggml_tensor * moshi_split_rvq_decode(
        ggml_context * ctx,
        moshi_split_rvq_t * srvq,
        ggml_tensor * codes ) {
    /* Decode the given codes to the quantized representation. */
    // codes is [B, K, T], with T frames, K nb of codebooks.
    auto B =       codes->ne[2];
    auto K =       codes->ne[1];
    auto T =       codes->ne[0];
    auto Bstride = codes->nb[2];
    auto Kstride = codes->nb[1];

    auto rvq_codes = ggml_view_3d( ctx, codes,
        T, srvq->n_q_semantic, B,
        Kstride, Bstride, 0
    );
    auto quantized = moshi_rvq_decode( ctx, srvq->rvq_first, rvq_codes );

    if ( K > srvq->n_q_semantic ) {
        rvq_codes = ggml_view_3d( ctx, codes,
            T, K - srvq->n_q_semantic, B,
            Kstride, Bstride,
            Kstride * srvq->n_q_semantic
        );
        auto quantized2 = moshi_rvq_decode(
            ctx, srvq->rvq_rest, rvq_codes );
        quantized = ggml_add( ctx, quantized, quantized2 );
    }
    return quantized;
}

void get_weights( WeightLoader * loader, std::string path, moshi_split_rvq_t * srvq ) {
    get_weights( loader, path + "rvq_first.", srvq->rvq_first );
    get_weights( loader, path + "rvq_rest.", srvq->rvq_rest );
}


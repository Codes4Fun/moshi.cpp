#pragma once

/*****************************************************************************\
 *   moshi.quantization.vq.ResidualVectorQuantizer
 * 
 * model locations:
 *   mimi.quantizer.rvq_first
 *   mimi.quantizer.rvq_rest
\*****************************************************************************/

struct moshi_rvq_t {
    int n_q;
    moshi_residual_vq_t * vq;
    torch_nn_conv1d_t * output_proj;
    torch_nn_conv1d_t * input_proj;
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

ggml_tensor * moshi_rvq_encode(
        ScratchContext & ctx,
        moshi_rvq_t * rvq,
        ggml_tensor * x ) {
    // x = self.input_proj(x)
    x = torch_nn_conv1d( ctx, rvq->input_proj, x );
    //codes = self.vq.encode(x, n_q=n_q)
    auto codes = moshi_residual_vq_encode( ctx, rvq->vq, x, rvq->n_q ); // don't have n_q!
    // codes = codes.transpose(0, 1)
    // the results from vq encode are [T, B, K]
    codes = ggml_permute( ctx, codes, 0, 2, 1, 3 );
    // codes is [T, K, B], with T frames, K nb of codebooks.
    return codes;
}

void get_weights( WeightLoader * loader, std::string path, moshi_rvq_t * rvq ) {
    get_weights( loader, path + "vq.", rvq->vq );
    get_weights( loader, path + "output_proj.", rvq->output_proj );
    get_weights( loader, path + "input_proj.", rvq->input_proj );
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

ggml_tensor * moshi_split_rvq_encode(
        ScratchContext & ctx,
        moshi_split_rvq_t * srvq,
        ggml_tensor * x
) {
    // codes = self.rvq_first.encode(x)
    auto codes = moshi_rvq_encode( ctx, srvq->rvq_first, x );

    // we don't store n_q, maybe we should?
    // if self.n_q > self.n_q_semantic:
        // acoustic_codes = self.rvq_rest.encode(x)
        auto acoustic_codes = moshi_rvq_encode( ctx,
            srvq->rvq_rest, x );
        //codes = torch.cat([codes, acoustic_codes], dim=1)
        codes = ggml_concat( ctx, codes, acoustic_codes, 1 );
    // codes is [B, K, T], with T frames, K nb of codebooks.
    return codes;
}
            
void get_weights( WeightLoader * loader, std::string path, moshi_split_rvq_t * srvq ) {
    get_weights( loader, path + "rvq_first.", srvq->rvq_first );
    get_weights( loader, path + "rvq_rest.", srvq->rvq_rest );
}


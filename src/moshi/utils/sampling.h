#pragma once

// NOTE: the call to exponential is the main source of randomness
ggml_tensor * moshi_multinomial(
        ScratchContext &ctx,
        ggml_tensor * input,
        int num_samples
    ) {
    auto input_ = ggml_reshape_2d( ctx, input,
        input->ne[0], input->ne[1] * input->ne[2] * input->ne[3]
    );
    auto q = ctx.exponential( GGML_NE( input_->ne[0], input_->ne[1] ), 1.f);
    q = ggml_div( ctx, input_, q );

    auto output_ = ggml_argmax( ctx, q );
    auto output = ggml_reshape_4d( ctx, output_,
        output_->ne[0],
        input->ne[1],
        input->ne[2],
        input->ne[3]
    );
    return output;
}

int moshi_sample_top_k_int( ScratchContext &ctx, ggml_tensor * probs, int k) {
    /* Sample next token from top K values along the last dimension of the input probs tensor.

    Args:
        probs (tensor): Input probabilities with token candidates on the last dimension.
        k (int): The k in “top-k”.
    Returns:
        torch.Tensor: Sampled tokens.
    */
    k = probs->ne[0] < k? probs->ne[0] : k;
    auto indices = ggml_top_k( ctx, probs, k );
    auto probs_rows = ggml_permute( ctx, probs, 1, 0, 2, 3 );
    probs_rows = ggml_get_rows( ctx, ggml_cont( ctx, probs_rows ), indices );
    probs = ggml_permute( ctx, probs_rows, 1, 0, 2, 3 );

    auto next_token = moshi_multinomial( ctx, probs, 1 );

    auto indices_rows = ggml_permute( ctx, indices, 1, 0, 2, 3 );
    next_token = ggml_get_rows( ctx, ggml_cont( ctx, indices_rows ), next_token );
    int next_token_int;
    ctx.build_forward_expand( next_token, &next_token_int );
    ctx.compute();
    return next_token_int;
}

int moshi_sample_token_int(
        ScratchContext & ctx,
        ggml_tensor * logits,
        bool use_sampling = false,
        float temp = 1.0,
        int top_k = 0,
        float top_p = 0.0
    ) {
    //assert use_sampling, use_sampling
    //assert temp == 0.6, temp
    //assert top_k > 0, top_k
    //assert top_p == 0.0, top_p
    /* Given logits of shape [*, Card], returns a LongTensor of shape [*]. */
    // Apply softmax for sampling if temp > 0. Else, do greedy sampling to avoid zero division error.
    if ( use_sampling && temp > 0.f ) {
        auto logits_temp = ggml_scale( ctx, logits, 1.0 / temp);
        auto probs = ggml_soft_max( ctx, logits_temp );

        return moshi_sample_top_k_int( ctx, probs, top_k );
    }
    auto next_token = ggml_argmax( ctx, logits );
    int next_token_int;
    ctx.build_forward_expand( next_token, &next_token_int );
    ctx.compute();
    return next_token_int;
}

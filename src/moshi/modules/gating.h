#pragma once

/*****************************************\
 * moshi.modules.gating.ActivationGating
\*****************************************/

struct moshi_activation_gating_t {
    own_ptr<torch_nn_linear_t> linear_in;
    own_ptr<torch_nn_linear_t> linear_out;
};

ggml_tensor * moshi_activation_gating(
        ggml_context * ctx,
        moshi_activation_gating_t * gating,
        ggml_tensor * x ) {
    x = torch_nn_linear( ctx, gating->linear_in, x );

    auto x_left = ggml_view_4d( ctx, x,
        x->ne[0] / 2, 1, x->ne[1], x->ne[2],
        // byte strides
        x->nb[1] / 2, x->nb[1], x->nb[2],
        // byte offset
        0
    );
    auto x_right = ggml_view_4d( ctx, x,
        x->ne[0] / 2, 1, x->ne[1], x->ne[2],
        // byte strides
        x->nb[1] / 2, x->nb[1], x->nb[2],
        // byte offset
        x->nb[1] / 2
    );
    x_left = ggml_silu( ctx, x_left );
    x = ggml_mul( ctx, x_left, x_right );

    x = torch_nn_linear( ctx, gating->linear_out, x );
    return x;
}

void get_weights( WeightLoader * loader, std::string path,
        moshi_activation_gating_t * gating ) {
    get_weights( loader, path + "linear_in.", gating->linear_in );
    get_weights( loader, path + "linear_out.", gating->linear_out );
}


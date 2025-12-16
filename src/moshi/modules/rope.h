#pragma once

// this seems cacheable, repeats for many layers
std::tuple<ggml_tensor*,ggml_tensor*> moshi_get_timestep_embedding(
        ScratchContext & ctx,
        int T, int D,
        ggml_tensor * offset,
        int max_period ) {
    int D_half = D / 2;
    auto ts = ctx.arange( 0, (float) T, 1 );
    ts = ggml_add( ctx, ts, offset );
    auto rot = ggml_timestep_embedding( ctx, ts, D, max_period );
    auto rotr = ggml_view_2d( ctx, rot, D_half, T, rot->nb[1], 0 );
    auto roti = ggml_view_2d( ctx, rot, D_half, T, rot->nb[1], rot->nb[0] * D_half );
    return std::make_tuple( rotr, roti);
}

std::tuple<ggml_tensor*,ggml_tensor*> moshi_apply_rope(
        ScratchContext & ctx,
        ggml_tensor * q,
        ggml_tensor * k,
        ggml_tensor * offset,
        int max_period = 10000,
        bool time_before_heads = false ) {
    /*
    Args:
        q (torch.Tensor): queries, shape `[B, T, H, D]`.
        k (torch.Tensor): keys, shape `[B, T, H, D]`.
        offset (int): current offset, e.g. when streaming.
        max_period (float): maximum period for the cos and sin.
        time_before_heads (bool):  if True, expected [B, T, H, D], else [B, H, T ,D]
    */
    int B = (int) q->ne[3];
    int T, H;
    if ( time_before_heads ) {
        T = (int) q->ne[2];
        H = (int) q->ne[1];
    } else {
        H = (int) q->ne[2];
        T = (int) q->ne[1];
    }
    int D = (int) q->ne[0];
    int D_half = D / 2;
    int B_x_H = B * H;

    // [B, H, T, D]
    q = ggml_cont( ctx, q );
    q = ggml_reshape_4d( ctx, q, 2, D_half, T, B_x_H );
    // [B*H, T, D/2, 2]
    q = ggml_cont( ctx, ggml_permute( ctx, q, 3, 0, 1, 2 ) );
    // [2, B*H, T, D/2]
    auto qr = ggml_view_3d( ctx, q,
        D_half, T, B_x_H,
        q->nb[1], q->nb[2],
        0 );
    // [B*H, T, D/2]
    auto qi = ggml_view_3d( ctx, q,
        D_half, T, B_x_H,
        q->nb[1], q->nb[2],
        q->nb[2] * B_x_H );
    // [B*H, T, D/2]
    qr = ggml_reshape_4d( ctx, qr, D_half, T, H, B );
    // [B, H, T, D/2]
    qi = ggml_reshape_4d( ctx, qi, D_half, T, H, B );
    // [B, H, T, D/2]

    // [B, H, T, D]
    k = ggml_cont( ctx, k );
    k = ggml_reshape_4d( ctx, k, 2, D_half, T, B_x_H );
    // [B*H, T, D/2, 2]
    k = ggml_cont( ctx, ggml_permute( ctx, k, 3, 0, 1, 2 ) );
    // [2, B*H, T, D/2]
    auto kr = ggml_view_3d( ctx, k,
        D_half, T, B_x_H,
        k->nb[1], k->nb[2],
        0 );
    // [B*H, T, D/2]
    auto ki = ggml_view_3d( ctx, k,
        D_half, T, B_x_H,
        k->nb[1], k->nb[2],
        k->nb[2] * B_x_H );
    // [B*H, T, D/2]
    kr = ggml_reshape_4d( ctx, kr, D_half, T, H, B );
    // [B, H, T, D/2]
    ki = ggml_reshape_4d( ctx, ki, D_half, T, H, B );
    // [B, H, T, D/2]

    auto [rotr, roti] = moshi_get_timestep_embedding(
        ctx, T, D, offset, max_period );

    // qor = qr * rotr - qi * roti
    // qoi = qr * roti + qi * rotr
    auto qor = ggml_sub( ctx,
        ggml_mul( ctx, qr, rotr ),
        ggml_mul( ctx, qi, roti ) );
    auto qoi = ggml_add( ctx,
        ggml_mul( ctx, qr, roti ),
        ggml_mul( ctx, qi, rotr ) );

    // kor = kr * rotr - ki * roti
    // koi = kr * roti + ki * rotr
    auto kor = ggml_sub( ctx,
        ggml_mul( ctx, kr, rotr ),
        ggml_mul( ctx, ki, roti ) );
    auto koi = ggml_add( ctx,
        ggml_mul( ctx, kr, roti ),
        ggml_mul( ctx, ki, rotr ) );
    
    auto qo = ggml_concat( ctx, qor, qoi, 0 );
    auto ko = ggml_concat( ctx, kor, koi, 0 );

    // [B, H, T, D]
    return std::make_tuple( qo, ko );
}


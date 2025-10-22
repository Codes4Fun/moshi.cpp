#pragma once

bool visited( ggml_tensor * tensor ){
    if (tensor->name[GGML_MAX_NAME-1] == 1)
        return true;
    assert( tensor->name[GGML_MAX_NAME-1] == 0 );
    ((ggml_tensor*)tensor)->name[GGML_MAX_NAME-2] = 0;
    ((ggml_tensor*)tensor)->name[GGML_MAX_NAME-1] = 1;
    return false;
}

void check( const ggml_tensor * tensor ) {
    if (!tensor)
        return;
    assert( tensor->name[0] == '@' );
    if (visited( (ggml_tensor*)tensor ))
        return;
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (tensor->src[i] && tensor->src[i] != tensor)
            check( tensor->src[i] );
    }
}

struct ggml_tensor * _ggml_new_tensor(
        struct ggml_context * ctx,
        enum   ggml_type type,
        int    n_dims,
        const int64_t *ne,
        const char * file, int line ) {
    auto tensor = ggml_new_tensor( ctx, type, n_dims, ne );
    snprintf( tensor->name, GGML_MAX_NAME, "@%s %d", file, line);
    return tensor;
}
#define ggml_new_tensor(...) _ggml_new_tensor(__VA_ARGS__, __func__, __LINE__)

struct ggml_tensor * _ggml_new_tensor_1d(
        struct ggml_context * ctx,
        enum   ggml_type type,
        int64_t ne0,
        const char * file, int line ) {
    auto tensor = ggml_new_tensor_1d( ctx, type, ne0 );
    snprintf( tensor->name, GGML_MAX_NAME, "@%s %d", file, line);
    return tensor;
}
#define ggml_new_tensor_1d(...) _ggml_new_tensor_1d(__VA_ARGS__, __func__, __LINE__)

struct ggml_tensor * _ggml_new_tensor_2d(
        struct ggml_context * ctx,
        enum   ggml_type type,
        int64_t ne0,
        int64_t ne1,
        const char * file, int line ) {
    auto tensor = ggml_new_tensor_2d( ctx, type, ne0, ne1 );
    snprintf( tensor->name, GGML_MAX_NAME, "@%s %d", file, line);
    return tensor;
}
#define ggml_new_tensor_2d(...) _ggml_new_tensor_2d(__VA_ARGS__, __func__, __LINE__)

struct ggml_tensor * _ggml_new_tensor_3d(
        struct ggml_context * ctx,
        enum   ggml_type type,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2,
        const char * file, int line ) {
    auto tensor = ggml_new_tensor_3d( ctx, type, ne0, ne1, ne2 );
    snprintf( tensor->name, GGML_MAX_NAME, "@%s %d", file, line);
    return tensor;
}
#define ggml_new_tensor_3d(...) _ggml_new_tensor_3d(__VA_ARGS__, __func__, __LINE__)

struct ggml_tensor * _ggml_new_tensor_4d(
        struct ggml_context * ctx,
        enum   ggml_type type,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2,
        int64_t ne3,
        const char * file, int line ) {
    auto tensor = ggml_new_tensor_4d( ctx, type, ne0, ne1, ne2, ne3 );
    snprintf( tensor->name, GGML_MAX_NAME, "@%s %d", file, line);
    return tensor;
}
#define ggml_new_tensor_4d(...) _ggml_new_tensor_4d(__VA_ARGS__, __func__, __LINE__)

struct ggml_tensor * _ggml_dup_tensor (
        struct ggml_context * ctx,
        const struct ggml_tensor * src,
        const char * file, int line ) {
    check( src );
    auto tensor = ggml_dup_tensor( ctx, src );
    snprintf( tensor->name, GGML_MAX_NAME, "@%s %d", file, line);
    return tensor;
}
#define ggml_dup_tensor(...)  _ggml_dup_tensor (__VA_ARGS__, __func__, __LINE__)

struct ggml_tensor * _ggml_set_name   (
        struct ggml_tensor * tensor,
        const char * name,
        const char * file, int line ) {
    snprintf( tensor->name, GGML_MAX_NAME, "@%s %d %s", file, line, name);
    return tensor;
}
#define ggml_set_name(...)    _ggml_set_name   (__VA_ARGS__, __func__, __LINE__)

struct ggml_tensor * _ggml_add(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b,
        const char * file, int line ) {
    check( a );
    check( b );
    auto tensor = ggml_add( ctx, a, b );
    snprintf( tensor->name, GGML_MAX_NAME, "@%s %d", file, line);
    return tensor;
}
#define ggml_add(...) _ggml_add(__VA_ARGS__, __func__, __LINE__)

struct ggml_tensor * _ggml_add_inplace(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b,
        const char * file, int line ) {
    check( a );
    check( b );
    auto tensor = ggml_add_inplace( ctx, a, b );
    snprintf( tensor->name, GGML_MAX_NAME, "@%s %d", file, line);
    return tensor;
}
#define ggml_add_inplace(...) _ggml_add_inplace(__VA_ARGS__, __func__, __LINE__)

struct ggml_tensor * _ggml_sub(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b,
        const char * file, int line ) {
    check( a );
    check( b );
    auto tensor = ggml_sub( ctx, a, b );
    snprintf( tensor->name, GGML_MAX_NAME, "@%s %d", file, line);
    return tensor;
}
#define ggml_sub(...) _ggml_sub(__VA_ARGS__, __func__, __LINE__)

struct ggml_tensor * _ggml_mul(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b,
        const char * file, int line ) {
    check( a );
    check( b );
    auto tensor = ggml_mul( ctx, a, b );
    snprintf( tensor->name, GGML_MAX_NAME, "@%s %d", file, line);
    return tensor;
}
#define ggml_mul(...) _ggml_mul(__VA_ARGS__, __func__, __LINE__)

struct ggml_tensor * _ggml_div(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b,
        const char * file, int line ) {
    check( a );
    check( b );
    auto tensor = ggml_div( ctx, a, b );
    snprintf( tensor->name, GGML_MAX_NAME, "@%s %d", file, line);
    return tensor;
}
#define ggml_div(...) _ggml_div(__VA_ARGS__, __func__, __LINE__)

struct ggml_tensor * _ggml_neg(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        const char * file, int line ) {
    check( a );
    auto tensor = ggml_neg( ctx, a );
    snprintf( tensor->name, GGML_MAX_NAME, "@%s %d", file, line);
    return tensor;
}
#define ggml_neg(...) _ggml_neg(__VA_ARGS__, __func__, __LINE__)

struct ggml_tensor * _ggml_sum(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        const char * file, int line ) {
    check( a );
    auto tensor = ggml_sum( ctx, a );
    snprintf( tensor->name, GGML_MAX_NAME, "@%s %d", file, line);
    return tensor;
}
#define ggml_sum(...) _ggml_sum(__VA_ARGS__, __func__, __LINE__)

struct ggml_tensor * _ggml_repeat_4d(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        int64_t    ne0,
        int64_t    ne1,
        int64_t    ne2,
        int64_t    ne3,
        const char * file, int line ) {
    check( a );
    auto tensor = ggml_repeat_4d( ctx, a, ne0, ne1, ne2, ne3 );
    snprintf( tensor->name, GGML_MAX_NAME, "@%s %d", file, line);
    return tensor;
}
#define ggml_repeat_4d(...) _ggml_repeat_4d(__VA_ARGS__, __func__, __LINE__)

struct ggml_tensor * _ggml_concat(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b,
        int                   dim,
        const char * file, int line ) {
    check( a );
    check( b );
    auto tensor = ggml_concat( ctx, a, b, dim );
    snprintf( tensor->name, GGML_MAX_NAME, "@%s %d", file, line);
    return tensor;
}
#define ggml_concat(...) _ggml_concat(__VA_ARGS__, __func__, __LINE__)

struct ggml_tensor * _ggml_elu(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        const char * file, int line ) {
    check( a );
    auto tensor = ggml_elu( ctx, a );
    snprintf( tensor->name, GGML_MAX_NAME, "@%s %d", file, line);
    return tensor;
}
#define ggml_elu(...) _ggml_elu(__VA_ARGS__, __func__, __LINE__)

struct ggml_tensor * _ggml_gelu(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        const char * file, int line ) {
    check( a );
    auto tensor = ggml_gelu( ctx, a );
    snprintf( tensor->name, GGML_MAX_NAME, "@%s %d", file, line);
    return tensor;
}
#define ggml_gelu(...) _ggml_gelu(__VA_ARGS__, __func__, __LINE__)

struct ggml_tensor * _ggml_silu(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        const char * file, int line ) {
    check( a );
    auto tensor = ggml_silu( ctx, a );
    snprintf( tensor->name, GGML_MAX_NAME, "@%s %d", file, line);
    return tensor;
}
#define ggml_silu(...) _ggml_silu(__VA_ARGS__, __func__, __LINE__)

struct ggml_tensor * _ggml_norm(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        float                 eps,
        const char * file, int line ) {
    check( a );
    auto tensor = ggml_norm( ctx, a, eps );
    snprintf( tensor->name, GGML_MAX_NAME, "@%s %d", file, line);
    return tensor;
}
#define ggml_norm(...) _ggml_norm(__VA_ARGS__, __func__, __LINE__)

struct ggml_tensor * _ggml_rms_norm(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        float                 eps,
        const char * file, int line ) {
    check( a );
    auto tensor = ggml_rms_norm( ctx, a, eps );
    snprintf( tensor->name, GGML_MAX_NAME, "@%s %d", file, line);
    return tensor;
}
#define ggml_rms_norm(...) _ggml_rms_norm(__VA_ARGS__, __func__, __LINE__)

struct ggml_tensor * _ggml_mul_mat(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b,
        const char * file, int line ) {
    check( a );
    check( b );
    auto tensor = ggml_mul_mat( ctx, a, b );
    snprintf( tensor->name, GGML_MAX_NAME, "@%s %d", file, line);
    return tensor;
}
#define ggml_mul_mat(...) _ggml_mul_mat(__VA_ARGS__, __func__, __LINE__)

struct ggml_tensor * _ggml_argmax(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        const char * file, int line ) {
    check( a );
    auto tensor = ggml_argmax( ctx, a );
    snprintf( tensor->name, GGML_MAX_NAME, "@%s %d", file, line);
    return tensor;
}
#define ggml_argmax(...) _ggml_argmax(__VA_ARGS__, __func__, __LINE__)

struct ggml_tensor * _ggml_scale(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        float                 s,
        const char * file, int line ) {
    check( a );
    auto tensor = ggml_scale( ctx, a, s );
    snprintf( tensor->name, GGML_MAX_NAME, "@%s %d", file, line);
    return tensor;
}
#define ggml_scale(...) _ggml_scale(__VA_ARGS__, __func__, __LINE__)

struct ggml_tensor * _ggml_cpy(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b,
        const char * file, int line ) {
    check( a );
    check( b );
    auto tensor = ggml_cpy( ctx, a, b );
    snprintf( tensor->name, GGML_MAX_NAME, "@%s %d", file, line);
    return tensor;
}
#define ggml_cpy(...) _ggml_cpy(__VA_ARGS__, __func__, __LINE__)

struct ggml_tensor * _ggml_cast(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        enum   ggml_type      type,
        const char * file, int line ) {
    check( a );
    auto tensor = ggml_cast( ctx, a, type );
    snprintf( tensor->name, GGML_MAX_NAME, "@%s %d", file, line);
    return tensor;
}
#define ggml_cast(...) _ggml_cast(__VA_ARGS__, __func__, __LINE__)

struct ggml_tensor * _ggml_cont(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        const char * file, int line ) {
    check( a );
    auto tensor = ggml_cont( ctx, a );
    snprintf( tensor->name, GGML_MAX_NAME, "@%s %d", file, line);
    return tensor;
}
#define ggml_cont(...) _ggml_cont(__VA_ARGS__, __func__, __LINE__)

struct ggml_tensor * _ggml_reshape_2d(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        const char * file, int line ) {
    check( a );
    auto tensor = ggml_reshape_2d( ctx, a, ne0, ne1 );
    snprintf( tensor->name, GGML_MAX_NAME, "@%s %d", file, line);
    return tensor;
}
#define ggml_reshape_2d(...) _ggml_reshape_2d(__VA_ARGS__, __func__, __LINE__)

struct ggml_tensor * _ggml_reshape_3d(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        const char * file, int line ) {
    check( a );
    auto tensor = ggml_reshape_3d( ctx, a, ne0, ne1, ne2 );
    snprintf( tensor->name, GGML_MAX_NAME, "@%s %d", file, line);
    return tensor;
}
#define ggml_reshape_3d(...) _ggml_reshape_3d(__VA_ARGS__, __func__, __LINE__)

struct ggml_tensor * _ggml_reshape_4d(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3,
        const char * file, int line ) {
    check( a );
    auto tensor = ggml_reshape_4d( ctx, a, ne0, ne1, ne2, ne3 );
    snprintf( tensor->name, GGML_MAX_NAME, "@%s %d", file, line);
    return tensor;
}
#define ggml_reshape_4d(...) _ggml_reshape_4d(__VA_ARGS__, __func__, __LINE__)

struct ggml_tensor * _ggml_view_1d(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        int64_t               ne0,
        size_t                offset,
        const char * file, int line ) {
    check( a );
    auto tensor = ggml_view_1d( ctx, a, ne0, offset );
    snprintf( tensor->name, GGML_MAX_NAME, "@%s %d", file, line);
    return tensor;
}
#define ggml_view_1d(...) _ggml_view_1d(__VA_ARGS__, __func__, __LINE__)

struct ggml_tensor * _ggml_view_2d(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        size_t                nb1, // row stride in bytes
        size_t                offset,
        const char * file, int line ) {
    check( a );
    auto tensor = ggml_view_2d( ctx, a, ne0, ne1, nb1, offset );
    snprintf( tensor->name, GGML_MAX_NAME, "@%s %d", file, line);
    return tensor;
}
#define ggml_view_2d(...) _ggml_view_2d(__VA_ARGS__, __func__, __LINE__)

struct ggml_tensor * _ggml_view_3d(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        size_t                nb1, // row   stride in bytes
        size_t                nb2, // slice stride in bytes
        size_t                offset,
        const char * file, int line ) {
    check( a );
    auto tensor = ggml_view_3d( ctx, a, ne0, ne1, ne2, nb1, nb2, offset );
    snprintf( tensor->name, GGML_MAX_NAME, "@%s %d", file, line);
    return tensor;
}
#define ggml_view_3d(...) _ggml_view_3d(__VA_ARGS__, __func__, __LINE__)

struct ggml_tensor * _ggml_view_4d(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3,
        size_t                nb1, // row   stride in bytes
        size_t                nb2, // slice stride in bytes
        size_t                nb3,
        size_t                offset,
        const char * file, int line ) {
    check( a );
    auto tensor = ggml_view_4d( ctx, a, ne0, ne1, ne2, ne3, nb1, nb2, nb3, offset );
    snprintf( tensor->name, GGML_MAX_NAME, "@%s %d", file, line);
    return tensor;
}
#define ggml_view_4d(...) _ggml_view_4d(__VA_ARGS__, __func__, __LINE__)

struct ggml_tensor * _ggml_permute(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        int                   axis0,
        int                   axis1,
        int                   axis2,
        int                   axis3,
        const char * file, int line ) {
    check( a );
    auto tensor = ggml_permute( ctx, a, axis0, axis1, axis2, axis3 );
    snprintf( tensor->name, GGML_MAX_NAME, "@%s %d", file, line);
    return tensor;
}
#define ggml_permute(...) _ggml_permute(__VA_ARGS__, __func__, __LINE__)

struct ggml_tensor * _ggml_transpose(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        const char * file, int line ) {
    check( a );
    auto tensor = ggml_transpose( ctx, a );
    snprintf( tensor->name, GGML_MAX_NAME, "@%s %d", file, line);
    return tensor;
}
#define ggml_transpose(...) _ggml_transpose(__VA_ARGS__, __func__, __LINE__)

struct ggml_tensor * _ggml_soft_max(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        const char * file, int line ) {
    check( a );
    auto tensor = ggml_soft_max( ctx, a );
    snprintf( tensor->name, GGML_MAX_NAME, "@%s %d", file, line);
    return tensor;
}
#define ggml_soft_max(...) _ggml_soft_max(__VA_ARGS__, __func__, __LINE__)

struct ggml_tensor * _ggml_soft_max_ext(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * mask,
        float                 scale,
        float                 max_bias,
        const char * file, int line ) {
    check( a );
    check( mask );
    auto tensor = ggml_soft_max_ext( ctx, a, mask, scale, max_bias );
    snprintf( tensor->name, GGML_MAX_NAME, "@%s %d", file, line);
    return tensor;
}
#define ggml_soft_max_ext(...) _ggml_soft_max_ext(__VA_ARGS__, __func__, __LINE__)

struct ggml_tensor * _ggml_get_rows(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,  // data
        struct ggml_tensor  * b,
        const char * file, int line ) {
    check( a );
    check( b );
    auto tensor = ggml_get_rows( ctx, a, b );
    snprintf( tensor->name, GGML_MAX_NAME, "@%s %d", file, line);
    return tensor;
}
#define ggml_get_rows(...) _ggml_get_rows(__VA_ARGS__, __func__, __LINE__)

struct ggml_tensor * _ggml_clamp(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        float                 min,
        float                 max,
        const char * file, int line ) {
    check( a );
    auto tensor = ggml_clamp( ctx, a, min, max );
    snprintf( tensor->name, GGML_MAX_NAME, "@%s %d", file, line);
    return tensor;
}
#define ggml_clamp(...) _ggml_clamp(__VA_ARGS__, __func__, __LINE__)

struct ggml_tensor * _ggml_conv_1d(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,   // convolution kernel
        struct ggml_tensor  * b,   // data
        int                   s0,  // stride
        int                   p0,  // padding
        int                   d0,
        const char * file, int line ) {
    check( a );
    check( b );
    auto tensor = ggml_conv_1d( ctx, a, b, s0, p0, d0 );
    snprintf( tensor->name, GGML_MAX_NAME, "@%s %d", file, line);
    visited( tensor ); // produces other tensors
    return tensor;
}
#define ggml_conv_1d(...) _ggml_conv_1d(__VA_ARGS__, __func__, __LINE__)

struct ggml_tensor * _ggml_conv_transpose_1d(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,   // convolution kernel
        struct ggml_tensor  * b,   // data
        int                   s0,  // stride
        int                   p0,  // padding
        int                   d0,
        const char * file, int line ) {
    check( a );
    check( b );
    auto tensor = ggml_conv_transpose_1d( ctx, a, b, s0, p0, d0 );
    snprintf( tensor->name, GGML_MAX_NAME, "@%s %d", file, line);
    return tensor;
}
#define ggml_conv_transpose_1d(...) _ggml_conv_transpose_1d(__VA_ARGS__, __func__, __LINE__)

struct ggml_tensor * _ggml_arange(
        struct ggml_context * ctx,
        float                 start,
        float                 stop,
        float                 step,
        const char * file, int line ) {
    auto tensor = ggml_arange( ctx, start, stop, step );
    snprintf( tensor->name, GGML_MAX_NAME, "@%s %d", file, line);
    return tensor;
}
#define ggml_arange(...) _ggml_arange(__VA_ARGS__, __func__, __LINE__)

struct ggml_tensor * _ggml_top_k(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        int                   k,
        const char * file, int line ) {
    check( a );
    auto tensor = ggml_top_k( ctx, a, k );
    snprintf( tensor->name, GGML_MAX_NAME, "@%s %d", file, line);
    visited( tensor ); // produces other tensors
    return tensor;
}
#define ggml_top_k(...) _ggml_top_k(__VA_ARGS__, __func__, __LINE__)

struct ggml_tensor * _ggml_timestep_embedding(
        struct ggml_context * ctx,
        struct ggml_tensor  * timesteps,
        int                   dim,
        int                   max_period,
        const char * file, int line ) {
    check( timesteps );
    auto tensor = ggml_timestep_embedding( ctx, timesteps, dim, max_period );
    snprintf( tensor->name, GGML_MAX_NAME, "@%s %d", file, line);
    return tensor;
}
#define ggml_timestep_embedding(...) _ggml_timestep_embedding(__VA_ARGS__, __func__, __LINE__)


struct ggml_tensor * _ggml_sum_rows( ggml_context * ctx, ggml_tensor * a,
        const char * file, int line ) {
    check( a );
    auto tensor = ggml_sum_rows( ctx, a );
    snprintf( tensor->name, GGML_MAX_NAME, "@%s %d", file, line);
    return tensor;
}
#define ggml_sum_rows(...) _ggml_sum_rows(__VA_ARGS__, __func__, __LINE__)


void _ggml_build_forward_expand(
        struct ggml_cgraph * cgraph,
        struct ggml_tensor * tensor ) {
    check( tensor );
    ggml_build_forward_expand( cgraph, tensor );
}
#define ggml_build_forward_expand(...) _ggml_build_forward_expand(__VA_ARGS__)






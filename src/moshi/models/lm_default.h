#pragma once

moshi_lmmodel_t * moshi_lmmodel_alloc_default() {
    auto lm_transformer = new moshi_streaming_transformer_t{ /*.layers=*/ {
        new moshi_streaming_transformer_layer_t{
            /*.norm1_rms=*/ new moshi_rms_norm_t{ /*.eps=*/ 0.000000 },
            /*.norm1=*/ NULL,
            /*self_attn=*/ new moshi_smha_t{
                /*.embed_dim=*/ 2048,
                /*.num_heads=*/ 16,
                /*.cross_attention=*/ false,
                /*.cache_cross_attention=*/ true,
                /*.causal=*/ true,
                /*.rope_max_period=*/ 10000,
                /*.context=*/ 500,
                /*.weights_per_step=*/ 0,
                /*.weights_per_step_schedule=*/ {},
                /*.in_projs=*/ {new torch_nn_linear_t},
                /*.out_projs=*/ {new torch_nn_linear_t}
            },
            /*.layer_scale_1=*/ NULL,
            /*.norm_cross=*/ new torch_nn_layer_norm_t{ /*.eps=*/ 0.000000 },
            /*.cross_attention=*/ new moshi_smha_t{
                /*.embed_dim=*/ 2048,
                /*.num_heads=*/ 16,
                /*.cross_attention=*/ true,
                /*.cache_cross_attention=*/ true,
                /*.causal=*/ false,
                /*.rope_max_period=*/ 0,
                /*.context=*/ 0,
                /*.weights_per_step=*/ 0,
                /*.weights_per_step_schedule=*/ {},
                /*.in_projs=*/ {new torch_nn_linear_t},
                /*.out_projs=*/ {new torch_nn_linear_t}
            },
            /*.norm2_rms=*/ new moshi_rms_norm_t{ /*.eps=*/ 0.000000 },
            /*.norm2=*/ NULL,
            /*.weights_per_step_schedule=*/ {},
            /*.gating=*/ {
                new moshi_activation_gating_t{
                    /*.linear_in=*/new torch_nn_linear_t,
                    /*.linear_out=*/new torch_nn_linear_t
                },
            },
            /*.linear1=*/ NULL,
            /*.linear2=*/ NULL,
            /*.layer_scale_2=*/ NULL
        }, new moshi_streaming_transformer_layer_t{
            /*.norm1_rms=*/ new moshi_rms_norm_t{ /*.eps=*/ 0.000000 },
            /*.norm1=*/ NULL,
            /*self_attn=*/ new moshi_smha_t{
                /*.embed_dim=*/ 2048,
                /*.num_heads=*/ 16,
                /*.cross_attention=*/ false,
                /*.cache_cross_attention=*/ true,
                /*.causal=*/ true,
                /*.rope_max_period=*/ 10000,
                /*.context=*/ 500,
                /*.weights_per_step=*/ 0,
                /*.weights_per_step_schedule=*/ {},
                /*.in_projs=*/ { new torch_nn_linear_t },
                /*.out_projs=*/ { new torch_nn_linear_t }
            }, /*.layer_scale_1=*/ NULL,
            /*.norm_cross=*/ new torch_nn_layer_norm_t{ /*.eps=*/ 0.000000 },
            /*.cross_attention=*/ new moshi_smha_t{
                /*.embed_dim=*/ 2048,
                /*.num_heads=*/ 16,
                /*.cross_attention=*/ true,
                /*.cache_cross_attention=*/ true,
                /*.causal=*/ false,
                /*.rope_max_period=*/ 0,
                /*.context=*/ 0,
                /*.weights_per_step=*/ 0,
                /*.weights_per_step_schedule=*/ {},
                /*.in_projs=*/ { new torch_nn_linear_t },
                /*.out_projs=*/ { new torch_nn_linear_t }
            }, /*.norm2_rms=*/ new moshi_rms_norm_t{ /*.eps=*/ 0.000000 },
            /*.norm2=*/ NULL,
            /*.weights_per_step_schedule=*/ {},
            /*.gating=*/ {
                new moshi_activation_gating_t{
                    /*.linear_in=*/new torch_nn_linear_t,
                    /*.linear_out=*/new torch_nn_linear_t
                },
            },
            /*.linear1=*/ NULL,
            /*.linear2=*/ NULL,
            /*.layer_scale_2=*/ NULL
        }, new moshi_streaming_transformer_layer_t{
            /*.norm1_rms=*/ new moshi_rms_norm_t{ /*.eps=*/ 0.000000 },
            /*.norm1=*/ NULL,
            /*self_attn=*/ new moshi_smha_t{
                /*.embed_dim=*/ 2048,
                /*.num_heads=*/ 16,
                /*.cross_attention=*/ false,
                /*.cache_cross_attention=*/ true,
                /*.causal=*/ true,
                /*.rope_max_period=*/ 10000,
                /*.context=*/ 500,
                /*.weights_per_step=*/ 0,
                /*.weights_per_step_schedule=*/ {},
                /*.in_projs=*/ { new torch_nn_linear_t },
                /*.out_projs=*/ { new torch_nn_linear_t }
            }, /*.layer_scale_1=*/ NULL,
            /*.norm_cross=*/ new torch_nn_layer_norm_t{ /*.eps=*/ 0.000000 },
            /*.cross_attention=*/ new moshi_smha_t{
                /*.embed_dim=*/ 2048,
                /*.num_heads=*/ 16,
                /*.cross_attention=*/ true,
                /*.cache_cross_attention=*/ true,
                /*.causal=*/ false,
                /*.rope_max_period=*/ 0,
                /*.context=*/ 0,
                /*.weights_per_step=*/ 0,
                /*.weights_per_step_schedule=*/ {},
                /*.in_projs=*/ { new torch_nn_linear_t },
                /*.out_projs=*/ { new torch_nn_linear_t }
            }, /*.norm2_rms=*/ new moshi_rms_norm_t{ /*.eps=*/ 0.000000 },
            /*.norm2=*/ NULL,
            /*.weights_per_step_schedule=*/ {},
            /*.gating=*/ {
                new moshi_activation_gating_t{
                    /*.linear_in=*/new torch_nn_linear_t,
                    /*.linear_out=*/new torch_nn_linear_t
                },
            },
            /*.linear1=*/ NULL,
            /*.linear2=*/ NULL,
            /*.layer_scale_2=*/ NULL
        }, new moshi_streaming_transformer_layer_t{
            /*.norm1_rms=*/ new moshi_rms_norm_t{ /*.eps=*/ 0.000000 },
            /*.norm1=*/ NULL,
            /*self_attn=*/ new moshi_smha_t{
                /*.embed_dim=*/ 2048,
                /*.num_heads=*/ 16,
                /*.cross_attention=*/ false,
                /*.cache_cross_attention=*/ true,
                /*.causal=*/ true,
                /*.rope_max_period=*/ 10000,
                /*.context=*/ 500,
                /*.weights_per_step=*/ 0,
                /*.weights_per_step_schedule=*/ {},
                /*.in_projs=*/ { new torch_nn_linear_t },
                /*.out_projs=*/ { new torch_nn_linear_t }
            }, /*.layer_scale_1=*/ NULL,
            /*.norm_cross=*/ new torch_nn_layer_norm_t{ /*.eps=*/ 0.000000 },
            /*.cross_attention=*/ new moshi_smha_t{
                /*.embed_dim=*/ 2048,
                /*.num_heads=*/ 16,
                /*.cross_attention=*/ true,
                /*.cache_cross_attention=*/ true,
                /*.causal=*/ false,
                /*.rope_max_period=*/ 0,
                /*.context=*/ 0,
                /*.weights_per_step=*/ 0,
                /*.weights_per_step_schedule=*/ {},
                /*.in_projs=*/ { new torch_nn_linear_t },
                /*.out_projs=*/ { new torch_nn_linear_t }
            }, /*.norm2_rms=*/ new moshi_rms_norm_t{ /*.eps=*/ 0.000000 },
            /*.norm2=*/ NULL,
            /*.weights_per_step_schedule=*/ {},
            /*.gating=*/ {
                new moshi_activation_gating_t{
                    /*.linear_in=*/new torch_nn_linear_t,
                    /*.linear_out=*/new torch_nn_linear_t
                },
            },
            /*.linear1=*/ NULL,
            /*.linear2=*/ NULL,
            /*.layer_scale_2=*/ NULL
        }, new moshi_streaming_transformer_layer_t{
            /*.norm1_rms=*/ new moshi_rms_norm_t{ /*.eps=*/ 0.000000 },
            /*.norm1=*/ NULL,
            /*self_attn=*/ new moshi_smha_t{
                /*.embed_dim=*/ 2048,
                /*.num_heads=*/ 16,
                /*.cross_attention=*/ false,
                /*.cache_cross_attention=*/ true,
                /*.causal=*/ true,
                /*.rope_max_period=*/ 10000,
                /*.context=*/ 500,
                /*.weights_per_step=*/ 0,
                /*.weights_per_step_schedule=*/ {},
                /*.in_projs=*/ { new torch_nn_linear_t },
                /*.out_projs=*/ { new torch_nn_linear_t }
            }, /*.layer_scale_1=*/ NULL,
            /*.norm_cross=*/ new torch_nn_layer_norm_t{ /*.eps=*/ 0.000000 },
            /*.cross_attention=*/ new moshi_smha_t{
                /*.embed_dim=*/ 2048,
                /*.num_heads=*/ 16,
                /*.cross_attention=*/ true,
                /*.cache_cross_attention=*/ true,
                /*.causal=*/ false,
                /*.rope_max_period=*/ 0,
                /*.context=*/ 0,
                /*.weights_per_step=*/ 0,
                /*.weights_per_step_schedule=*/ {},
                /*.in_projs=*/ { new torch_nn_linear_t },
                /*.out_projs=*/ { new torch_nn_linear_t }
            }, /*.norm2_rms=*/ new moshi_rms_norm_t{ /*.eps=*/ 0.000000 },
            /*.norm2=*/ NULL,
            /*.weights_per_step_schedule=*/ {},
            /*.gating=*/ {
                new moshi_activation_gating_t{
                    /*.linear_in=*/new torch_nn_linear_t,
                    /*.linear_out=*/new torch_nn_linear_t
                },
            },
            /*.linear1=*/ NULL,
            /*.linear2=*/ NULL,
            /*.layer_scale_2=*/ NULL
        }, new moshi_streaming_transformer_layer_t{
            /*.norm1_rms=*/ new moshi_rms_norm_t{ /*.eps=*/ 0.000000 },
            /*.norm1=*/ NULL,
            /*self_attn=*/ new moshi_smha_t{
                /*.embed_dim=*/ 2048,
                /*.num_heads=*/ 16,
                /*.cross_attention=*/ false,
                /*.cache_cross_attention=*/ true,
                /*.causal=*/ true,
                /*.rope_max_period=*/ 10000,
                /*.context=*/ 500,
                /*.weights_per_step=*/ 0,
                /*.weights_per_step_schedule=*/ {},
                /*.in_projs=*/ { new torch_nn_linear_t },
                /*.out_projs=*/ { new torch_nn_linear_t }
            }, /*.layer_scale_1=*/ NULL,
            /*.norm_cross=*/ new torch_nn_layer_norm_t{ /*.eps=*/ 0.000000 },
            /*.cross_attention=*/ new moshi_smha_t{
                /*.embed_dim=*/ 2048,
                /*.num_heads=*/ 16,
                /*.cross_attention=*/ true,
                /*.cache_cross_attention=*/ true,
                /*.causal=*/ false,
                /*.rope_max_period=*/ 0,
                /*.context=*/ 0,
                /*.weights_per_step=*/ 0,
                /*.weights_per_step_schedule=*/ {},
                /*.in_projs=*/ { new torch_nn_linear_t },
                /*.out_projs=*/ { new torch_nn_linear_t }
            }, /*.norm2_rms=*/ new moshi_rms_norm_t{ /*.eps=*/ 0.000000 },
            /*.norm2=*/ NULL,
            /*.weights_per_step_schedule=*/ {},
            /*.gating=*/ {
                new moshi_activation_gating_t{
                    /*.linear_in=*/new torch_nn_linear_t,
                    /*.linear_out=*/new torch_nn_linear_t
                },
            },
            /*.linear1=*/ NULL,
            /*.linear2=*/ NULL,
            /*.layer_scale_2=*/ NULL
        }, new moshi_streaming_transformer_layer_t{
            /*.norm1_rms=*/ new moshi_rms_norm_t{ /*.eps=*/ 0.000000 },
            /*.norm1=*/ NULL,
            /*self_attn=*/ new moshi_smha_t{
                /*.embed_dim=*/ 2048,
                /*.num_heads=*/ 16,
                /*.cross_attention=*/ false,
                /*.cache_cross_attention=*/ true,
                /*.causal=*/ true,
                /*.rope_max_period=*/ 10000,
                /*.context=*/ 500,
                /*.weights_per_step=*/ 0,
                /*.weights_per_step_schedule=*/ {},
                /*.in_projs=*/ { new torch_nn_linear_t },
                /*.out_projs=*/ { new torch_nn_linear_t }
            }, /*.layer_scale_1=*/ NULL,
            /*.norm_cross=*/ new torch_nn_layer_norm_t{ /*.eps=*/ 0.000000 },
            /*.cross_attention=*/ new moshi_smha_t{
                /*.embed_dim=*/ 2048,
                /*.num_heads=*/ 16,
                /*.cross_attention=*/ true,
                /*.cache_cross_attention=*/ true,
                /*.causal=*/ false,
                /*.rope_max_period=*/ 0,
                /*.context=*/ 0,
                /*.weights_per_step=*/ 0,
                /*.weights_per_step_schedule=*/ {},
                /*.in_projs=*/ { new torch_nn_linear_t },
                /*.out_projs=*/ { new torch_nn_linear_t }
            }, /*.norm2_rms=*/ new moshi_rms_norm_t{ /*.eps=*/ 0.000000 },
            /*.norm2=*/ NULL,
            /*.weights_per_step_schedule=*/ {},
            /*.gating=*/ {
                new moshi_activation_gating_t{
                    /*.linear_in=*/new torch_nn_linear_t,
                    /*.linear_out=*/new torch_nn_linear_t
                },
            },
            /*.linear1=*/ NULL,
            /*.linear2=*/ NULL,
            /*.layer_scale_2=*/ NULL
        }, new moshi_streaming_transformer_layer_t{
            /*.norm1_rms=*/ new moshi_rms_norm_t{ /*.eps=*/ 0.000000 },
            /*.norm1=*/ NULL,
            /*self_attn=*/ new moshi_smha_t{
                /*.embed_dim=*/ 2048,
                /*.num_heads=*/ 16,
                /*.cross_attention=*/ false,
                /*.cache_cross_attention=*/ true,
                /*.causal=*/ true,
                /*.rope_max_period=*/ 10000,
                /*.context=*/ 500,
                /*.weights_per_step=*/ 0,
                /*.weights_per_step_schedule=*/ {},
                /*.in_projs=*/ { new torch_nn_linear_t },
                /*.out_projs=*/ { new torch_nn_linear_t }
            }, /*.layer_scale_1=*/ NULL,
            /*.norm_cross=*/ new torch_nn_layer_norm_t{ /*.eps=*/ 0.000000 },
            /*.cross_attention=*/ new moshi_smha_t{
                /*.embed_dim=*/ 2048,
                /*.num_heads=*/ 16,
                /*.cross_attention=*/ true,
                /*.cache_cross_attention=*/ true,
                /*.causal=*/ false,
                /*.rope_max_period=*/ 0,
                /*.context=*/ 0,
                /*.weights_per_step=*/ 0,
                /*.weights_per_step_schedule=*/ {},
                /*.in_projs=*/ { new torch_nn_linear_t },
                /*.out_projs=*/ { new torch_nn_linear_t }
            }, /*.norm2_rms=*/ new moshi_rms_norm_t{ /*.eps=*/ 0.000000 },
            /*.norm2=*/ NULL,
            /*.weights_per_step_schedule=*/ {},
            /*.gating=*/ {
                new moshi_activation_gating_t{
                    /*.linear_in=*/new torch_nn_linear_t,
                    /*.linear_out=*/new torch_nn_linear_t
                },
            },
            /*.linear1=*/ NULL,
            /*.linear2=*/ NULL,
            /*.layer_scale_2=*/ NULL
        }, new moshi_streaming_transformer_layer_t{
            /*.norm1_rms=*/ new moshi_rms_norm_t{ /*.eps=*/ 0.000000 },
            /*.norm1=*/ NULL,
            /*self_attn=*/ new moshi_smha_t{
                /*.embed_dim=*/ 2048,
                /*.num_heads=*/ 16,
                /*.cross_attention=*/ false,
                /*.cache_cross_attention=*/ true,
                /*.causal=*/ true,
                /*.rope_max_period=*/ 10000,
                /*.context=*/ 500,
                /*.weights_per_step=*/ 0,
                /*.weights_per_step_schedule=*/ {},
                /*.in_projs=*/ { new torch_nn_linear_t },
                /*.out_projs=*/ { new torch_nn_linear_t }
            }, /*.layer_scale_1=*/ NULL,
            /*.norm_cross=*/ new torch_nn_layer_norm_t{ /*.eps=*/ 0.000000 },
            /*.cross_attention=*/ new moshi_smha_t{
                /*.embed_dim=*/ 2048,
                /*.num_heads=*/ 16,
                /*.cross_attention=*/ true,
                /*.cache_cross_attention=*/ true,
                /*.causal=*/ false,
                /*.rope_max_period=*/ 0,
                /*.context=*/ 0,
                /*.weights_per_step=*/ 0,
                /*.weights_per_step_schedule=*/ {},
                /*.in_projs=*/ { new torch_nn_linear_t },
                /*.out_projs=*/ { new torch_nn_linear_t }
            }, /*.norm2_rms=*/ new moshi_rms_norm_t{ /*.eps=*/ 0.000000 },
            /*.norm2=*/ NULL,
            /*.weights_per_step_schedule=*/ {},
            /*.gating=*/ {
                new moshi_activation_gating_t{
                    /*.linear_in=*/new torch_nn_linear_t,
                    /*.linear_out=*/new torch_nn_linear_t
                },
            },
            /*.linear1=*/ NULL,
            /*.linear2=*/ NULL,
            /*.layer_scale_2=*/ NULL
        }, new moshi_streaming_transformer_layer_t{
            /*.norm1_rms=*/ new moshi_rms_norm_t{ /*.eps=*/ 0.000000 },
            /*.norm1=*/ NULL,
            /*self_attn=*/ new moshi_smha_t{
                /*.embed_dim=*/ 2048,
                /*.num_heads=*/ 16,
                /*.cross_attention=*/ false,
                /*.cache_cross_attention=*/ true,
                /*.causal=*/ true,
                /*.rope_max_period=*/ 10000,
                /*.context=*/ 500,
                /*.weights_per_step=*/ 0,
                /*.weights_per_step_schedule=*/ {},
                /*.in_projs=*/ { new torch_nn_linear_t },
                /*.out_projs=*/ { new torch_nn_linear_t }
            }, /*.layer_scale_1=*/ NULL,
            /*.norm_cross=*/ new torch_nn_layer_norm_t{ /*.eps=*/ 0.000000 },
            /*.cross_attention=*/ new moshi_smha_t{
                /*.embed_dim=*/ 2048,
                /*.num_heads=*/ 16,
                /*.cross_attention=*/ true,
                /*.cache_cross_attention=*/ true,
                /*.causal=*/ false,
                /*.rope_max_period=*/ 0,
                /*.context=*/ 0,
                /*.weights_per_step=*/ 0,
                /*.weights_per_step_schedule=*/ {},
                /*.in_projs=*/ { new torch_nn_linear_t },
                /*.out_projs=*/ { new torch_nn_linear_t }
            }, /*.norm2_rms=*/ new moshi_rms_norm_t{ /*.eps=*/ 0.000000 },
            /*.norm2=*/ NULL,
            /*.weights_per_step_schedule=*/ {},
            /*.gating=*/ {
                new moshi_activation_gating_t{
                    /*.linear_in=*/new torch_nn_linear_t,
                    /*.linear_out=*/new torch_nn_linear_t
                },
            },
            /*.linear1=*/ NULL,
            /*.linear2=*/ NULL,
            /*.layer_scale_2=*/ NULL
        }, new moshi_streaming_transformer_layer_t{
            /*.norm1_rms=*/ new moshi_rms_norm_t{ /*.eps=*/ 0.000000 },
            /*.norm1=*/ NULL,
            /*self_attn=*/ new moshi_smha_t{
                /*.embed_dim=*/ 2048,
                /*.num_heads=*/ 16,
                /*.cross_attention=*/ false,
                /*.cache_cross_attention=*/ true,
                /*.causal=*/ true,
                /*.rope_max_period=*/ 10000,
                /*.context=*/ 500,
                /*.weights_per_step=*/ 0,
                /*.weights_per_step_schedule=*/ {},
                /*.in_projs=*/ { new torch_nn_linear_t },
                /*.out_projs=*/ { new torch_nn_linear_t }
            }, /*.layer_scale_1=*/ NULL,
            /*.norm_cross=*/ new torch_nn_layer_norm_t{ /*.eps=*/ 0.000000 },
            /*.cross_attention=*/ new moshi_smha_t{
                /*.embed_dim=*/ 2048,
                /*.num_heads=*/ 16,
                /*.cross_attention=*/ true,
                /*.cache_cross_attention=*/ true,
                /*.causal=*/ false,
                /*.rope_max_period=*/ 0,
                /*.context=*/ 0,
                /*.weights_per_step=*/ 0,
                /*.weights_per_step_schedule=*/ {},
                /*.in_projs=*/ { new torch_nn_linear_t },
                /*.out_projs=*/ { new torch_nn_linear_t }
            }, /*.norm2_rms=*/ new moshi_rms_norm_t{ /*.eps=*/ 0.000000 },
            /*.norm2=*/ NULL,
            /*.weights_per_step_schedule=*/ {},
            /*.gating=*/ {
                new moshi_activation_gating_t{
                    /*.linear_in=*/new torch_nn_linear_t,
                    /*.linear_out=*/new torch_nn_linear_t
                },
            },
            /*.linear1=*/ NULL,
            /*.linear2=*/ NULL,
            /*.layer_scale_2=*/ NULL
        }, new moshi_streaming_transformer_layer_t{
            /*.norm1_rms=*/ new moshi_rms_norm_t{ /*.eps=*/ 0.000000 },
            /*.norm1=*/ NULL,
            /*self_attn=*/ new moshi_smha_t{
                /*.embed_dim=*/ 2048,
                /*.num_heads=*/ 16,
                /*.cross_attention=*/ false,
                /*.cache_cross_attention=*/ true,
                /*.causal=*/ true,
                /*.rope_max_period=*/ 10000,
                /*.context=*/ 500,
                /*.weights_per_step=*/ 0,
                /*.weights_per_step_schedule=*/ {},
                /*.in_projs=*/ { new torch_nn_linear_t },
                /*.out_projs=*/ { new torch_nn_linear_t }
            }, /*.layer_scale_1=*/ NULL,
            /*.norm_cross=*/ new torch_nn_layer_norm_t{ /*.eps=*/ 0.000000 },
            /*.cross_attention=*/ new moshi_smha_t{
                /*.embed_dim=*/ 2048,
                /*.num_heads=*/ 16,
                /*.cross_attention=*/ true,
                /*.cache_cross_attention=*/ true,
                /*.causal=*/ false,
                /*.rope_max_period=*/ 0,
                /*.context=*/ 0,
                /*.weights_per_step=*/ 0,
                /*.weights_per_step_schedule=*/ {},
                /*.in_projs=*/ { new torch_nn_linear_t },
                /*.out_projs=*/ { new torch_nn_linear_t }
            }, /*.norm2_rms=*/ new moshi_rms_norm_t{ /*.eps=*/ 0.000000 },
            /*.norm2=*/ NULL,
            /*.weights_per_step_schedule=*/ {},
            /*.gating=*/ {
                new moshi_activation_gating_t{
                    /*.linear_in=*/new torch_nn_linear_t,
                    /*.linear_out=*/new torch_nn_linear_t
                },
            },
            /*.linear1=*/ NULL,
            /*.linear2=*/ NULL,
            /*.layer_scale_2=*/ NULL
        }, new moshi_streaming_transformer_layer_t{
            /*.norm1_rms=*/ new moshi_rms_norm_t{ /*.eps=*/ 0.000000 },
            /*.norm1=*/ NULL,
            /*self_attn=*/ new moshi_smha_t{
                /*.embed_dim=*/ 2048,
                /*.num_heads=*/ 16,
                /*.cross_attention=*/ false,
                /*.cache_cross_attention=*/ true,
                /*.causal=*/ true,
                /*.rope_max_period=*/ 10000,
                /*.context=*/ 500,
                /*.weights_per_step=*/ 0,
                /*.weights_per_step_schedule=*/ {},
                /*.in_projs=*/ { new torch_nn_linear_t },
                /*.out_projs=*/ { new torch_nn_linear_t }
            }, /*.layer_scale_1=*/ NULL,
            /*.norm_cross=*/ new torch_nn_layer_norm_t{ /*.eps=*/ 0.000000 },
            /*.cross_attention=*/ new moshi_smha_t{
                /*.embed_dim=*/ 2048,
                /*.num_heads=*/ 16,
                /*.cross_attention=*/ true,
                /*.cache_cross_attention=*/ true,
                /*.causal=*/ false,
                /*.rope_max_period=*/ 0,
                /*.context=*/ 0,
                /*.weights_per_step=*/ 0,
                /*.weights_per_step_schedule=*/ {},
                /*.in_projs=*/ { new torch_nn_linear_t },
                /*.out_projs=*/ { new torch_nn_linear_t }
            }, /*.norm2_rms=*/ new moshi_rms_norm_t{ /*.eps=*/ 0.000000 },
            /*.norm2=*/ NULL,
            /*.weights_per_step_schedule=*/ {},
            /*.gating=*/ {
                new moshi_activation_gating_t{
                    /*.linear_in=*/new torch_nn_linear_t,
                    /*.linear_out=*/new torch_nn_linear_t
                },
            },
            /*.linear1=*/ NULL,
            /*.linear2=*/ NULL,
            /*.layer_scale_2=*/ NULL
        }, new moshi_streaming_transformer_layer_t{
            /*.norm1_rms=*/ new moshi_rms_norm_t{ /*.eps=*/ 0.000000 },
            /*.norm1=*/ NULL,
            /*self_attn=*/ new moshi_smha_t{
                /*.embed_dim=*/ 2048,
                /*.num_heads=*/ 16,
                /*.cross_attention=*/ false,
                /*.cache_cross_attention=*/ true,
                /*.causal=*/ true,
                /*.rope_max_period=*/ 10000,
                /*.context=*/ 500,
                /*.weights_per_step=*/ 0,
                /*.weights_per_step_schedule=*/ {},
                /*.in_projs=*/ { new torch_nn_linear_t },
                /*.out_projs=*/ { new torch_nn_linear_t }
            }, /*.layer_scale_1=*/ NULL,
            /*.norm_cross=*/ new torch_nn_layer_norm_t{ /*.eps=*/ 0.000000 },
            /*.cross_attention=*/ new moshi_smha_t{
                /*.embed_dim=*/ 2048,
                /*.num_heads=*/ 16,
                /*.cross_attention=*/ true,
                /*.cache_cross_attention=*/ true,
                /*.causal=*/ false,
                /*.rope_max_period=*/ 0,
                /*.context=*/ 0,
                /*.weights_per_step=*/ 0,
                /*.weights_per_step_schedule=*/ {},
                /*.in_projs=*/ { new torch_nn_linear_t },
                /*.out_projs=*/ { new torch_nn_linear_t }
            }, /*.norm2_rms=*/ new moshi_rms_norm_t{ /*.eps=*/ 0.000000 },
            /*.norm2=*/ NULL,
            /*.weights_per_step_schedule=*/ {},
            /*.gating=*/ {
                new moshi_activation_gating_t{
                    /*.linear_in=*/new torch_nn_linear_t,
                    /*.linear_out=*/new torch_nn_linear_t
                },
            },
            /*.linear1=*/ NULL,
            /*.linear2=*/ NULL,
            /*.layer_scale_2=*/ NULL
        }, new moshi_streaming_transformer_layer_t{
            /*.norm1_rms=*/ new moshi_rms_norm_t{ /*.eps=*/ 0.000000 },
            /*.norm1=*/ NULL,
            /*self_attn=*/ new moshi_smha_t{
                /*.embed_dim=*/ 2048,
                /*.num_heads=*/ 16,
                /*.cross_attention=*/ false,
                /*.cache_cross_attention=*/ true,
                /*.causal=*/ true,
                /*.rope_max_period=*/ 10000,
                /*.context=*/ 500,
                /*.weights_per_step=*/ 0,
                /*.weights_per_step_schedule=*/ {},
                /*.in_projs=*/ { new torch_nn_linear_t },
                /*.out_projs=*/ { new torch_nn_linear_t }
            }, /*.layer_scale_1=*/ NULL,
            /*.norm_cross=*/ new torch_nn_layer_norm_t{ /*.eps=*/ 0.000000 },
            /*.cross_attention=*/ new moshi_smha_t{
                /*.embed_dim=*/ 2048,
                /*.num_heads=*/ 16,
                /*.cross_attention=*/ true,
                /*.cache_cross_attention=*/ true,
                /*.causal=*/ false,
                /*.rope_max_period=*/ 0,
                /*.context=*/ 0,
                /*.weights_per_step=*/ 0,
                /*.weights_per_step_schedule=*/ {},
                /*.in_projs=*/ { new torch_nn_linear_t },
                /*.out_projs=*/ { new torch_nn_linear_t }
            }, /*.norm2_rms=*/ new moshi_rms_norm_t{ /*.eps=*/ 0.000000 },
            /*.norm2=*/ NULL,
            /*.weights_per_step_schedule=*/ {},
            /*.gating=*/ {
                new moshi_activation_gating_t{
                    /*.linear_in=*/new torch_nn_linear_t,
                    /*.linear_out=*/new torch_nn_linear_t
                },
            },
            /*.linear1=*/ NULL,
            /*.linear2=*/ NULL,
            /*.layer_scale_2=*/ NULL
        }, new moshi_streaming_transformer_layer_t{
            /*.norm1_rms=*/ new moshi_rms_norm_t{ /*.eps=*/ 0.000000 },
            /*.norm1=*/ NULL,
            /*self_attn=*/ new moshi_smha_t{
                /*.embed_dim=*/ 2048,
                /*.num_heads=*/ 16,
                /*.cross_attention=*/ false,
                /*.cache_cross_attention=*/ true,
                /*.causal=*/ true,
                /*.rope_max_period=*/ 10000,
                /*.context=*/ 500,
                /*.weights_per_step=*/ 0,
                /*.weights_per_step_schedule=*/ {},
                /*.in_projs=*/ { new torch_nn_linear_t },
                /*.out_projs=*/ { new torch_nn_linear_t }
            }, /*.layer_scale_1=*/ NULL,
            /*.norm_cross=*/ new torch_nn_layer_norm_t{ /*.eps=*/ 0.000000 },
            /*.cross_attention=*/ new moshi_smha_t{
                /*.embed_dim=*/ 2048,
                /*.num_heads=*/ 16,
                /*.cross_attention=*/ true,
                /*.cache_cross_attention=*/ true,
                /*.causal=*/ false,
                /*.rope_max_period=*/ 0,
                /*.context=*/ 0,
                /*.weights_per_step=*/ 0,
                /*.weights_per_step_schedule=*/ {},
                /*.in_projs=*/ { new torch_nn_linear_t },
                /*.out_projs=*/ { new torch_nn_linear_t }
            }, /*.norm2_rms=*/ new moshi_rms_norm_t{ /*.eps=*/ 0.000000 },
            /*.norm2=*/ NULL,
            /*.weights_per_step_schedule=*/ {},
            /*.gating=*/ {
                new moshi_activation_gating_t{
                    /*.linear_in=*/new torch_nn_linear_t,
                    /*.linear_out=*/new torch_nn_linear_t
                },
            },
            /*.linear1=*/ NULL,
            /*.linear2=*/ NULL,
            /*.layer_scale_2=*/ NULL
        },
    }
    };

    auto lm_depformer =
    new moshi_streaming_transformer_t{ /*.layers=*/ {
        new moshi_streaming_transformer_layer_t{
            /*.norm1_rms=*/ new moshi_rms_norm_t{ /*.eps=*/ 0.000000 },
            /*.norm1=*/ NULL,
            /*self_attn=*/ new moshi_smha_t{
                /*.embed_dim=*/ 1024,
                /*.num_heads=*/ 16,
                /*.cross_attention=*/ false,
                /*.cache_cross_attention=*/ true,
                /*.causal=*/ true,
                /*.rope_max_period=*/ 0,
                /*.context=*/ 0,
                /*.weights_per_step=*/ 32,
                /*.weights_per_step_schedule=*/ {0,1,2,3,4,5,6,7,8,8,8,8,8,8,8,8,9,9,9,9,9,9,9,9,10,10,10,10,10,10,10,10,},
                /*.in_projs=*/ {
                    new torch_nn_linear_t,
                    new torch_nn_linear_t,
                    new torch_nn_linear_t,
                    new torch_nn_linear_t,
                    new torch_nn_linear_t,
                    new torch_nn_linear_t,
                    new torch_nn_linear_t,
                    new torch_nn_linear_t,
                    new torch_nn_linear_t,
                    new torch_nn_linear_t,
                    new torch_nn_linear_t,
                },
                /*.out_projs=*/ {
                    new torch_nn_linear_t,
                    new torch_nn_linear_t,
                    new torch_nn_linear_t,
                    new torch_nn_linear_t,
                    new torch_nn_linear_t,
                    new torch_nn_linear_t,
                    new torch_nn_linear_t,
                    new torch_nn_linear_t,
                    new torch_nn_linear_t,
                    new torch_nn_linear_t,
                    new torch_nn_linear_t,
                }
            }, /*.layer_scale_1=*/ NULL,
            /*.norm_cross=*/ NULL,
            /*.cross_attention=*/ NULL,
            /*.norm2_rms=*/ new moshi_rms_norm_t{ /*.eps=*/ 0.000000 },
            /*.norm2=*/ NULL,
            /*.weights_per_step_schedule=*/ {0,1,2,3,4,5,6,7,8,8,8,8,8,8,8,8,9,9,9,9,9,9,9,9,10,10,10,10,10,10,10,10,},
            /*.gating=*/ {
                new moshi_activation_gating_t{
                    /*.linear_in=*/new torch_nn_linear_t,
                    /*.linear_out=*/new torch_nn_linear_t
                },
                new moshi_activation_gating_t{
                    /*.linear_in=*/new torch_nn_linear_t,
                    /*.linear_out=*/new torch_nn_linear_t
                },
                new moshi_activation_gating_t{
                    /*.linear_in=*/new torch_nn_linear_t,
                    /*.linear_out=*/new torch_nn_linear_t
                },
                new moshi_activation_gating_t{
                    /*.linear_in=*/new torch_nn_linear_t,
                    /*.linear_out=*/new torch_nn_linear_t
                },
                new moshi_activation_gating_t{
                    /*.linear_in=*/new torch_nn_linear_t,
                    /*.linear_out=*/new torch_nn_linear_t
                },
                new moshi_activation_gating_t{
                    /*.linear_in=*/new torch_nn_linear_t,
                    /*.linear_out=*/new torch_nn_linear_t
                },
                new moshi_activation_gating_t{
                    /*.linear_in=*/new torch_nn_linear_t,
                    /*.linear_out=*/new torch_nn_linear_t
                },
                new moshi_activation_gating_t{
                    /*.linear_in=*/new torch_nn_linear_t,
                    /*.linear_out=*/new torch_nn_linear_t
                },
                new moshi_activation_gating_t{
                    /*.linear_in=*/new torch_nn_linear_t,
                    /*.linear_out=*/new torch_nn_linear_t
                },
                new moshi_activation_gating_t{
                    /*.linear_in=*/new torch_nn_linear_t,
                    /*.linear_out=*/new torch_nn_linear_t
                },
                new moshi_activation_gating_t{
                    /*.linear_in=*/new torch_nn_linear_t,
                    /*.linear_out=*/new torch_nn_linear_t
                },
            },
            /*.linear1=*/ NULL,
            /*.linear2=*/ NULL,
            /*.layer_scale_2=*/ NULL
        }, new moshi_streaming_transformer_layer_t{
            /*.norm1_rms=*/ new moshi_rms_norm_t{ /*.eps=*/ 0.000000 },
            /*.norm1=*/ NULL,
            /*self_attn=*/ new moshi_smha_t{
                /*.embed_dim=*/ 1024,
                /*.num_heads=*/ 16,
                /*.cross_attention=*/ false,
                /*.cache_cross_attention=*/ true,
                /*.causal=*/ true,
                /*.rope_max_period=*/ 0,
                /*.context=*/ 0,
                /*.weights_per_step=*/ 32,
                /*.weights_per_step_schedule=*/ {0,1,2,3,4,5,6,7,8,8,8,8,8,8,8,8,9,9,9,9,9,9,9,9,10,10,10,10,10,10,10,10,},
                /*.in_projs=*/ {
                    new torch_nn_linear_t,
                    new torch_nn_linear_t,
                    new torch_nn_linear_t,
                    new torch_nn_linear_t,
                    new torch_nn_linear_t,
                    new torch_nn_linear_t,
                    new torch_nn_linear_t,
                    new torch_nn_linear_t,
                    new torch_nn_linear_t,
                    new torch_nn_linear_t,
                    new torch_nn_linear_t,
                },
                /*.out_projs=*/ {
                    new torch_nn_linear_t,
                    new torch_nn_linear_t,
                    new torch_nn_linear_t,
                    new torch_nn_linear_t,
                    new torch_nn_linear_t,
                    new torch_nn_linear_t,
                    new torch_nn_linear_t,
                    new torch_nn_linear_t,
                    new torch_nn_linear_t,
                    new torch_nn_linear_t,
                    new torch_nn_linear_t,
                }
            }, /*.layer_scale_1=*/ NULL,
            /*.norm_cross=*/ NULL,
            /*.cross_attention=*/ NULL,
            /*.norm2_rms=*/ new moshi_rms_norm_t{ /*.eps=*/ 0.000000 },
            /*.norm2=*/ NULL,
            /*.weights_per_step_schedule=*/ {0,1,2,3,4,5,6,7,8,8,8,8,8,8,8,8,9,9,9,9,9,9,9,9,10,10,10,10,10,10,10,10,},
            /*.gating=*/ {
                new moshi_activation_gating_t{
                    /*.linear_in=*/new torch_nn_linear_t,
                    /*.linear_out=*/new torch_nn_linear_t
                },
                new moshi_activation_gating_t{
                    /*.linear_in=*/new torch_nn_linear_t,
                    /*.linear_out=*/new torch_nn_linear_t
                },
                new moshi_activation_gating_t{
                    /*.linear_in=*/new torch_nn_linear_t,
                    /*.linear_out=*/new torch_nn_linear_t
                },
                new moshi_activation_gating_t{
                    /*.linear_in=*/new torch_nn_linear_t,
                    /*.linear_out=*/new torch_nn_linear_t
                },
                new moshi_activation_gating_t{
                    /*.linear_in=*/new torch_nn_linear_t,
                    /*.linear_out=*/new torch_nn_linear_t
                },
                new moshi_activation_gating_t{
                    /*.linear_in=*/new torch_nn_linear_t,
                    /*.linear_out=*/new torch_nn_linear_t
                },
                new moshi_activation_gating_t{
                    /*.linear_in=*/new torch_nn_linear_t,
                    /*.linear_out=*/new torch_nn_linear_t
                },
                new moshi_activation_gating_t{
                    /*.linear_in=*/new torch_nn_linear_t,
                    /*.linear_out=*/new torch_nn_linear_t
                },
                new moshi_activation_gating_t{
                    /*.linear_in=*/new torch_nn_linear_t,
                    /*.linear_out=*/new torch_nn_linear_t
                },
                new moshi_activation_gating_t{
                    /*.linear_in=*/new torch_nn_linear_t,
                    /*.linear_out=*/new torch_nn_linear_t
                },
                new moshi_activation_gating_t{
                    /*.linear_in=*/new torch_nn_linear_t,
                    /*.linear_out=*/new torch_nn_linear_t
                },
            },
            /*.linear1=*/ NULL,
            /*.linear2=*/ NULL,
            /*.layer_scale_2=*/ NULL
        }, new moshi_streaming_transformer_layer_t{
            /*.norm1_rms=*/ new moshi_rms_norm_t{ /*.eps=*/ 0.000000 },
            /*.norm1=*/ NULL,
            /*self_attn=*/ new moshi_smha_t{
                /*.embed_dim=*/ 1024,
                /*.num_heads=*/ 16,
                /*.cross_attention=*/ false,
                /*.cache_cross_attention=*/ true,
                /*.causal=*/ true,
                /*.rope_max_period=*/ 0,
                /*.context=*/ 0,
                /*.weights_per_step=*/ 32,
                /*.weights_per_step_schedule=*/ {0,1,2,3,4,5,6,7,8,8,8,8,8,8,8,8,9,9,9,9,9,9,9,9,10,10,10,10,10,10,10,10,},
                /*.in_projs=*/ {
                    new torch_nn_linear_t,
                    new torch_nn_linear_t,
                    new torch_nn_linear_t,
                    new torch_nn_linear_t,
                    new torch_nn_linear_t,
                    new torch_nn_linear_t,
                    new torch_nn_linear_t,
                    new torch_nn_linear_t,
                    new torch_nn_linear_t,
                    new torch_nn_linear_t,
                    new torch_nn_linear_t,
                },
                /*.out_projs=*/ {
                    new torch_nn_linear_t,
                    new torch_nn_linear_t,
                    new torch_nn_linear_t,
                    new torch_nn_linear_t,
                    new torch_nn_linear_t,
                    new torch_nn_linear_t,
                    new torch_nn_linear_t,
                    new torch_nn_linear_t,
                    new torch_nn_linear_t,
                    new torch_nn_linear_t,
                    new torch_nn_linear_t,
                }
            }, /*.layer_scale_1=*/ NULL,
            /*.norm_cross=*/ NULL,
            /*.cross_attention=*/ NULL,
            /*.norm2_rms=*/ new moshi_rms_norm_t{ /*.eps=*/ 0.000000 },
            /*.norm2=*/ NULL,
            /*.weights_per_step_schedule=*/ {0,1,2,3,4,5,6,7,8,8,8,8,8,8,8,8,9,9,9,9,9,9,9,9,10,10,10,10,10,10,10,10,},
            /*.gating=*/ {
                new moshi_activation_gating_t{
                    /*.linear_in=*/new torch_nn_linear_t,
                    /*.linear_out=*/new torch_nn_linear_t
                },
                new moshi_activation_gating_t{
                    /*.linear_in=*/new torch_nn_linear_t,
                    /*.linear_out=*/new torch_nn_linear_t
                },
                new moshi_activation_gating_t{
                    /*.linear_in=*/new torch_nn_linear_t,
                    /*.linear_out=*/new torch_nn_linear_t
                },
                new moshi_activation_gating_t{
                    /*.linear_in=*/new torch_nn_linear_t,
                    /*.linear_out=*/new torch_nn_linear_t
                },
                new moshi_activation_gating_t{
                    /*.linear_in=*/new torch_nn_linear_t,
                    /*.linear_out=*/new torch_nn_linear_t
                },
                new moshi_activation_gating_t{
                    /*.linear_in=*/new torch_nn_linear_t,
                    /*.linear_out=*/new torch_nn_linear_t
                },
                new moshi_activation_gating_t{
                    /*.linear_in=*/new torch_nn_linear_t,
                    /*.linear_out=*/new torch_nn_linear_t
                },
                new moshi_activation_gating_t{
                    /*.linear_in=*/new torch_nn_linear_t,
                    /*.linear_out=*/new torch_nn_linear_t
                },
                new moshi_activation_gating_t{
                    /*.linear_in=*/new torch_nn_linear_t,
                    /*.linear_out=*/new torch_nn_linear_t
                },
                new moshi_activation_gating_t{
                    /*.linear_in=*/new torch_nn_linear_t,
                    /*.linear_out=*/new torch_nn_linear_t
                },
                new moshi_activation_gating_t{
                    /*.linear_in=*/new torch_nn_linear_t,
                    /*.linear_out=*/new torch_nn_linear_t
                },
            },
            /*.linear1=*/ NULL,
            /*.linear2=*/ NULL,
            /*.layer_scale_2=*/ NULL
        }, new moshi_streaming_transformer_layer_t{
            /*.norm1_rms=*/ new moshi_rms_norm_t{ /*.eps=*/ 0.000000 },
            /*.norm1=*/ NULL,
            /*self_attn=*/ new moshi_smha_t{
                /*.embed_dim=*/ 1024,
                /*.num_heads=*/ 16,
                /*.cross_attention=*/ false,
                /*.cache_cross_attention=*/ true,
                /*.causal=*/ true,
                /*.rope_max_period=*/ 0,
                /*.context=*/ 0,
                /*.weights_per_step=*/ 32,
                /*.weights_per_step_schedule=*/ {0,1,2,3,4,5,6,7,8,8,8,8,8,8,8,8,9,9,9,9,9,9,9,9,10,10,10,10,10,10,10,10,},
                /*.in_projs=*/ {
                    new torch_nn_linear_t,
                    new torch_nn_linear_t,
                    new torch_nn_linear_t,
                    new torch_nn_linear_t,
                    new torch_nn_linear_t,
                    new torch_nn_linear_t,
                    new torch_nn_linear_t,
                    new torch_nn_linear_t,
                    new torch_nn_linear_t,
                    new torch_nn_linear_t,
                    new torch_nn_linear_t,
                },
                /*.out_projs=*/ {
                    new torch_nn_linear_t,
                    new torch_nn_linear_t,
                    new torch_nn_linear_t,
                    new torch_nn_linear_t,
                    new torch_nn_linear_t,
                    new torch_nn_linear_t,
                    new torch_nn_linear_t,
                    new torch_nn_linear_t,
                    new torch_nn_linear_t,
                    new torch_nn_linear_t,
                    new torch_nn_linear_t,
                }
            }, /*.layer_scale_1=*/ NULL,
            /*.norm_cross=*/ NULL,
            /*.cross_attention=*/ NULL,
            /*.norm2_rms=*/ new moshi_rms_norm_t{ /*.eps=*/ 0.000000 },
            /*.norm2=*/ NULL,
            /*.weights_per_step_schedule=*/ {0,1,2,3,4,5,6,7,8,8,8,8,8,8,8,8,9,9,9,9,9,9,9,9,10,10,10,10,10,10,10,10,},
            /*.gating=*/ {
                new moshi_activation_gating_t{
                    /*.linear_in=*/new torch_nn_linear_t,
                    /*.linear_out=*/new torch_nn_linear_t
                },
                new moshi_activation_gating_t{
                    /*.linear_in=*/new torch_nn_linear_t,
                    /*.linear_out=*/new torch_nn_linear_t
                },
                new moshi_activation_gating_t{
                    /*.linear_in=*/new torch_nn_linear_t,
                    /*.linear_out=*/new torch_nn_linear_t
                },
                new moshi_activation_gating_t{
                    /*.linear_in=*/new torch_nn_linear_t,
                    /*.linear_out=*/new torch_nn_linear_t
                },
                new moshi_activation_gating_t{
                    /*.linear_in=*/new torch_nn_linear_t,
                    /*.linear_out=*/new torch_nn_linear_t
                },
                new moshi_activation_gating_t{
                    /*.linear_in=*/new torch_nn_linear_t,
                    /*.linear_out=*/new torch_nn_linear_t
                },
                new moshi_activation_gating_t{
                    /*.linear_in=*/new torch_nn_linear_t,
                    /*.linear_out=*/new torch_nn_linear_t
                },
                new moshi_activation_gating_t{
                    /*.linear_in=*/new torch_nn_linear_t,
                    /*.linear_out=*/new torch_nn_linear_t
                },
                new moshi_activation_gating_t{
                    /*.linear_in=*/new torch_nn_linear_t,
                    /*.linear_out=*/new torch_nn_linear_t
                },
                new moshi_activation_gating_t{
                    /*.linear_in=*/new torch_nn_linear_t,
                    /*.linear_out=*/new torch_nn_linear_t
                },
                new moshi_activation_gating_t{
                    /*.linear_in=*/new torch_nn_linear_t,
                    /*.linear_out=*/new torch_nn_linear_t
                },
            },
            /*.linear1=*/ NULL,
            /*.linear2=*/ NULL,
            /*.layer_scale_2=*/ NULL
        },
    }};

    return new moshi_lmmodel_t{
        /*.n_q=*/ 32,
        /*.dep_q=*/ 32,
        /*.card=*/ 2048,
        /*.text_card=*/ 8000,
        /*.delays=*/ {0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2},
        /*.dim=*/ 2048,
        /*.depformer_weights_per_step_schedule=*/ {0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10},
        /*.emb=*/ {
            /*0*/ new moshi_scaled_embedding_t{NULL},
            /*1*/ new moshi_scaled_embedding_t{NULL},
            /*2*/ new moshi_scaled_embedding_t{NULL},
            /*3*/ new moshi_scaled_embedding_t{NULL},
            /*4*/ new moshi_scaled_embedding_t{NULL},
            /*5*/ new moshi_scaled_embedding_t{NULL},
            /*6*/ new moshi_scaled_embedding_t{NULL},
            /*7*/ new moshi_scaled_embedding_t{NULL},
            /*8*/ new moshi_scaled_embedding_t{NULL},
            /*9*/ new moshi_scaled_embedding_t{NULL},
            /*10*/ new moshi_scaled_embedding_t{NULL},
            /*11*/ new moshi_scaled_embedding_t{NULL},
            /*12*/ new moshi_scaled_embedding_t{NULL},
            /*13*/ new moshi_scaled_embedding_t{NULL},
            /*14*/ new moshi_scaled_embedding_t{NULL},
            /*15*/ new moshi_scaled_embedding_t{NULL},
            /*16*/ new moshi_scaled_embedding_t{NULL},
            /*17*/ new moshi_scaled_embedding_t{NULL},
            /*18*/ new moshi_scaled_embedding_t{NULL},
            /*19*/ new moshi_scaled_embedding_t{NULL},
            /*20*/ new moshi_scaled_embedding_t{NULL},
            /*21*/ new moshi_scaled_embedding_t{NULL},
            /*22*/ new moshi_scaled_embedding_t{NULL},
            /*23*/ new moshi_scaled_embedding_t{NULL},
            /*24*/ new moshi_scaled_embedding_t{NULL},
            /*25*/ new moshi_scaled_embedding_t{NULL},
            /*26*/ new moshi_scaled_embedding_t{NULL},
            /*27*/ new moshi_scaled_embedding_t{NULL},
            /*28*/ new moshi_scaled_embedding_t{NULL},
            /*29*/ new moshi_scaled_embedding_t{NULL},
            /*30*/ new moshi_scaled_embedding_t{NULL},
            /*31*/ new moshi_scaled_embedding_t{NULL},
        },
        /*.text_emb=*/ new moshi_scaled_embedding_demux_t{
            /*.num_embeddings=*/ 8001,
            /*.out1=*/ new torch_nn_linear_t,
            /*.out2=*/ new torch_nn_linear_t
        },

        /*.text_linear=*/ new torch_nn_linear_t,
        /*.transformer=*/ lm_transformer,
        /*.out_norm=*/ new moshi_rms_norm_t{1e-08},
        /*.depformer_multi_linear=*/ true,

        /*.depformer_in=*/ {
            /*0*/ new torch_nn_linear_t,
            /*1*/ new torch_nn_linear_t,
            /*2*/ new torch_nn_linear_t,
            /*3*/ new torch_nn_linear_t,
            /*4*/ new torch_nn_linear_t,
            /*5*/ new torch_nn_linear_t,
            /*6*/ new torch_nn_linear_t,
            /*7*/ new torch_nn_linear_t,
            /*8*/ new torch_nn_linear_t,
            /*9*/ new torch_nn_linear_t,
            /*10*/ new torch_nn_linear_t,
        },
        /*.depformer_emb=*/ {
            /*0*/ new moshi_scaled_embedding_t{new torch_nn_linear_t},
            /*1*/ new moshi_scaled_embedding_t{new torch_nn_linear_t},
            /*2*/ new moshi_scaled_embedding_t{new torch_nn_linear_t},
            /*3*/ new moshi_scaled_embedding_t{new torch_nn_linear_t},
            /*4*/ new moshi_scaled_embedding_t{new torch_nn_linear_t},
            /*5*/ new moshi_scaled_embedding_t{new torch_nn_linear_t},
            /*6*/ new moshi_scaled_embedding_t{new torch_nn_linear_t},
            /*7*/ new moshi_scaled_embedding_t{new torch_nn_linear_t},
            /*8*/ new moshi_scaled_embedding_t{new torch_nn_linear_t},
            /*9*/ new moshi_scaled_embedding_t{new torch_nn_linear_t},
            /*10*/ new moshi_scaled_embedding_t{new torch_nn_linear_t},
            /*11*/ new moshi_scaled_embedding_t{new torch_nn_linear_t},
            /*12*/ new moshi_scaled_embedding_t{new torch_nn_linear_t},
            /*13*/ new moshi_scaled_embedding_t{new torch_nn_linear_t},
            /*14*/ new moshi_scaled_embedding_t{new torch_nn_linear_t},
            /*15*/ new moshi_scaled_embedding_t{new torch_nn_linear_t},
            /*16*/ new moshi_scaled_embedding_t{new torch_nn_linear_t},
            /*17*/ new moshi_scaled_embedding_t{new torch_nn_linear_t},
            /*18*/ new moshi_scaled_embedding_t{new torch_nn_linear_t},
            /*19*/ new moshi_scaled_embedding_t{new torch_nn_linear_t},
            /*20*/ new moshi_scaled_embedding_t{new torch_nn_linear_t},
            /*21*/ new moshi_scaled_embedding_t{new torch_nn_linear_t},
            /*22*/ new moshi_scaled_embedding_t{new torch_nn_linear_t},
            /*23*/ new moshi_scaled_embedding_t{new torch_nn_linear_t},
            /*24*/ new moshi_scaled_embedding_t{new torch_nn_linear_t},
            /*25*/ new moshi_scaled_embedding_t{new torch_nn_linear_t},
            /*26*/ new moshi_scaled_embedding_t{new torch_nn_linear_t},
            /*27*/ new moshi_scaled_embedding_t{new torch_nn_linear_t},
            /*28*/ new moshi_scaled_embedding_t{new torch_nn_linear_t},
            /*29*/ new moshi_scaled_embedding_t{new torch_nn_linear_t},
            /*30*/ new moshi_scaled_embedding_t{new torch_nn_linear_t},
        },
        /*.depformer_text_emb=*/ new moshi_scaled_embedding_demux_t{
            /*.num_embeddings=*/ 8001,
            /*.out1=*/ new torch_nn_linear_t,
            /*.out2=*/ new torch_nn_linear_t
            // NOTE: the original had low_rank for no reason, never gets used in demux
        },
        /*.depformer=*/ lm_depformer,

        /*.linears=*/ {
            /*0*/ new torch_nn_linear_t,
            /*1*/ new torch_nn_linear_t,
            /*2*/ new torch_nn_linear_t,
            /*3*/ new torch_nn_linear_t,
            /*4*/ new torch_nn_linear_t,
            /*5*/ new torch_nn_linear_t,
            /*6*/ new torch_nn_linear_t,
            /*7*/ new torch_nn_linear_t,
            /*8*/ new torch_nn_linear_t,
            /*9*/ new torch_nn_linear_t,
            /*10*/ new torch_nn_linear_t,
            /*11*/ new torch_nn_linear_t,
            /*12*/ new torch_nn_linear_t,
            /*13*/ new torch_nn_linear_t,
            /*14*/ new torch_nn_linear_t,
            /*15*/ new torch_nn_linear_t,
            /*16*/ new torch_nn_linear_t,
            /*17*/ new torch_nn_linear_t,
            /*18*/ new torch_nn_linear_t,
            /*19*/ new torch_nn_linear_t,
            /*20*/ new torch_nn_linear_t,
            /*21*/ new torch_nn_linear_t,
            /*22*/ new torch_nn_linear_t,
            /*23*/ new torch_nn_linear_t,
            /*24*/ new torch_nn_linear_t,
            /*25*/ new torch_nn_linear_t,
            /*26*/ new torch_nn_linear_t,
            /*27*/ new torch_nn_linear_t,
            /*28*/ new torch_nn_linear_t,
            /*29*/ new torch_nn_linear_t,
            /*30*/ new torch_nn_linear_t,
            /*31*/ new torch_nn_linear_t
        },

        /*.num_codebooks=*/ 33, // n_q + 1
        /*.num_audio_codebooks=*/ 32, // n_q
        /*.audio_offset=*/ 1
    };
}


moshi_mimi_t * moshi_mimi_alloc_default() {
    //mimi.quantizer.
    auto mimi_quantizer = new moshi_split_rvq_t{
        /*.n_q_semantic=*/ 1,
        /*.rvq_first=*/ new moshi_rvq_t{
            /*.vq=*/new moshi_residual_vq_t{ /*.layers=*/ {
                new moshi_vq_t{ new moshi_EuclideanCodebook_t },
            }},
            /*.output_proj=*/ new torch_nn_conv1d_t
        }, /*.rvq_rest=*/ new moshi_rvq_t{
            /*.vq=*/new moshi_residual_vq_t{ /*.layers=*/ {
                new moshi_vq_t{ new moshi_EuclideanCodebook_t },
                new moshi_vq_t{ new moshi_EuclideanCodebook_t },
                new moshi_vq_t{ new moshi_EuclideanCodebook_t },
                new moshi_vq_t{ new moshi_EuclideanCodebook_t },
                new moshi_vq_t{ new moshi_EuclideanCodebook_t },
                new moshi_vq_t{ new moshi_EuclideanCodebook_t },
                new moshi_vq_t{ new moshi_EuclideanCodebook_t },
                new moshi_vq_t{ new moshi_EuclideanCodebook_t },
                new moshi_vq_t{ new moshi_EuclideanCodebook_t },
                new moshi_vq_t{ new moshi_EuclideanCodebook_t },
                new moshi_vq_t{ new moshi_EuclideanCodebook_t },
                new moshi_vq_t{ new moshi_EuclideanCodebook_t },
                new moshi_vq_t{ new moshi_EuclideanCodebook_t },
                new moshi_vq_t{ new moshi_EuclideanCodebook_t },
                new moshi_vq_t{ new moshi_EuclideanCodebook_t },
                new moshi_vq_t{ new moshi_EuclideanCodebook_t },
                new moshi_vq_t{ new moshi_EuclideanCodebook_t },
                new moshi_vq_t{ new moshi_EuclideanCodebook_t },
                new moshi_vq_t{ new moshi_EuclideanCodebook_t },
                new moshi_vq_t{ new moshi_EuclideanCodebook_t },
                new moshi_vq_t{ new moshi_EuclideanCodebook_t },
                new moshi_vq_t{ new moshi_EuclideanCodebook_t },
                new moshi_vq_t{ new moshi_EuclideanCodebook_t },
                new moshi_vq_t{ new moshi_EuclideanCodebook_t },
                new moshi_vq_t{ new moshi_EuclideanCodebook_t },
                new moshi_vq_t{ new moshi_EuclideanCodebook_t },
                new moshi_vq_t{ new moshi_EuclideanCodebook_t },
                new moshi_vq_t{ new moshi_EuclideanCodebook_t },
                new moshi_vq_t{ new moshi_EuclideanCodebook_t },
                new moshi_vq_t{ new moshi_EuclideanCodebook_t },
                new moshi_vq_t{ new moshi_EuclideanCodebook_t },
            }},
            /*.output_proj=*/ new torch_nn_conv1d_t
        }
    };

    auto mimi_upsample_convtr = new moshi_streaming_conv_transpose_1d_t{
        /*.in_channels=*/ 512,
        /*.out_channels=*/ 512,
        /*.kernel_size=*/ 4,
        /*.stride=*/ 2,
        /*.groups=*/ 512,
    };

    // "mimi.decoder_.transformer."
    auto mimi_decoder__transformer =
    new moshi_streaming_transformer_t{ /*.layers=*/ {
        new moshi_streaming_transformer_layer_t{
            /*.norm1_rms=*/ NULL,
            /*.norm1=*/ new torch_nn_layer_norm_t{ /*.eps=*/ 0.000000 },
            /*self_attn=*/ new moshi_smha_t{
                /*.embed_dim=*/ 512,
                /*.num_heads=*/ 8,
                /*.cross_attention=*/ false,
                /*.cache_cross_attention=*/ true,
                /*.causal=*/ true,
                /*.rope_max_period=*/ 10000,
                /*.context=*/ 250,
                /*.weights_per_step=*/ 0,
                /*.weights_per_step_schedule=*/ {},
                /*.in_projs=*/ { new torch_nn_linear_t },
                /*.out_projs=*/ { new torch_nn_linear_t }
            }, /*.layer_scale_1=*/ new moshi_layer_scale_t,
            /*.norm_cross=*/ NULL,
            /*.cross_attention=*/ NULL,
            /*.norm2_rms=*/ NULL,
            /*.norm2=*/ new torch_nn_layer_norm_t{ /*.eps=*/ 0.000000 },
            /*.weights_per_step_schedule=*/ {},
            /*.gating=*/ {},
            /*.linear1=*/ new torch_nn_linear_t,
            /*.linear2=*/ new torch_nn_linear_t,
            /*.layer_scale_2=*/ new moshi_layer_scale_t
        }, new moshi_streaming_transformer_layer_t{
            /*.norm1_rms=*/ NULL,
            /*.norm1=*/ new torch_nn_layer_norm_t{ /*.eps=*/ 0.000000 },
            /*self_attn=*/ new moshi_smha_t{
                /*.embed_dim=*/ 512,
                /*.num_heads=*/ 8,
                /*.cross_attention=*/ false,
                /*.cache_cross_attention=*/ true,
                /*.causal=*/ true,
                /*.rope_max_period=*/ 10000,
                /*.context=*/ 250,
                /*.weights_per_step=*/ 0,
                /*.weights_per_step_schedule=*/ {},
                /*.in_projs=*/ { new torch_nn_linear_t },
                /*.out_projs=*/ { new torch_nn_linear_t }
            }, /*.layer_scale_1=*/ new moshi_layer_scale_t,
            /*.norm_cross=*/ NULL,
            /*.cross_attention=*/ NULL,
            /*.norm2_rms=*/ NULL,
            /*.norm2=*/ new torch_nn_layer_norm_t{ /*.eps=*/ 0.000000 },
            /*.weights_per_step_schedule=*/ {},
            /*.gating=*/ {},
            /*.linear1=*/ new torch_nn_linear_t,
            /*.linear2=*/ new torch_nn_linear_t,
            /*.layer_scale_2=*/ new moshi_layer_scale_t
        }, new moshi_streaming_transformer_layer_t{
            /*.norm1_rms=*/ NULL,
            /*.norm1=*/ new torch_nn_layer_norm_t{ /*.eps=*/ 0.000000 },
            /*self_attn=*/ new moshi_smha_t{
                /*.embed_dim=*/ 512,
                /*.num_heads=*/ 8,
                /*.cross_attention=*/ false,
                /*.cache_cross_attention=*/ true,
                /*.causal=*/ true,
                /*.rope_max_period=*/ 10000,
                /*.context=*/ 250,
                /*.weights_per_step=*/ 0,
                /*.weights_per_step_schedule=*/ {},
                /*.in_projs=*/ { new torch_nn_linear_t },
                /*.out_projs=*/ { new torch_nn_linear_t }
            }, /*.layer_scale_1=*/ new moshi_layer_scale_t,
            /*.norm_cross=*/ NULL,
            /*.cross_attention=*/ NULL,
            /*.norm2_rms=*/ NULL,
            /*.norm2=*/ new torch_nn_layer_norm_t{ /*.eps=*/ 0.000000 },
            /*.weights_per_step_schedule=*/ {},
            /*.gating=*/ {},
            /*.linear1=*/ new torch_nn_linear_t,
            /*.linear2=*/ new torch_nn_linear_t,
            /*.layer_scale_2=*/ new moshi_layer_scale_t
        }, new moshi_streaming_transformer_layer_t{
            /*.norm1_rms=*/ NULL,
            /*.norm1=*/ new torch_nn_layer_norm_t{ /*.eps=*/ 0.000000 },
            /*self_attn=*/ new moshi_smha_t{
                /*.embed_dim=*/ 512,
                /*.num_heads=*/ 8,
                /*.cross_attention=*/ false,
                /*.cache_cross_attention=*/ true,
                /*.causal=*/ true,
                /*.rope_max_period=*/ 10000,
                /*.context=*/ 250,
                /*.weights_per_step=*/ 0,
                /*.weights_per_step_schedule=*/ {},
                /*.in_projs=*/ { new torch_nn_linear_t },
                /*.out_projs=*/ { new torch_nn_linear_t }
            }, /*.layer_scale_1=*/ new moshi_layer_scale_t,
            /*.norm_cross=*/ NULL,
            /*.cross_attention=*/ NULL,
            /*.norm2_rms=*/ NULL,
            /*.norm2=*/ new torch_nn_layer_norm_t{ /*.eps=*/ 0.000000 },
            /*.weights_per_step_schedule=*/ {},
            /*.gating=*/ {},
            /*.linear1=*/ new torch_nn_linear_t,
            /*.linear2=*/ new torch_nn_linear_t,
            /*.layer_scale_2=*/ new moshi_layer_scale_t
        }, new moshi_streaming_transformer_layer_t{
            /*.norm1_rms=*/ NULL,
            /*.norm1=*/ new torch_nn_layer_norm_t{ /*.eps=*/ 0.000000 },
            /*self_attn=*/ new moshi_smha_t{
                /*.embed_dim=*/ 512,
                /*.num_heads=*/ 8,
                /*.cross_attention=*/ false,
                /*.cache_cross_attention=*/ true,
                /*.causal=*/ true,
                /*.rope_max_period=*/ 10000,
                /*.context=*/ 250,
                /*.weights_per_step=*/ 0,
                /*.weights_per_step_schedule=*/ {},
                /*.in_projs=*/ { new torch_nn_linear_t },
                /*.out_projs=*/ { new torch_nn_linear_t }
            }, /*.layer_scale_1=*/ new moshi_layer_scale_t,
            /*.norm_cross=*/ NULL,
            /*.cross_attention=*/ NULL,
            /*.norm2_rms=*/ NULL,
            /*.norm2=*/ new torch_nn_layer_norm_t{ /*.eps=*/ 0.000000 },
            /*.weights_per_step_schedule=*/ {},
            /*.gating=*/ {},
            /*.linear1=*/ new torch_nn_linear_t,
            /*.linear2=*/ new torch_nn_linear_t,
            /*.layer_scale_2=*/ new moshi_layer_scale_t
        }, new moshi_streaming_transformer_layer_t{
            /*.norm1_rms=*/ NULL,
            /*.norm1=*/ new torch_nn_layer_norm_t{ /*.eps=*/ 0.000000 },
            /*self_attn=*/ new moshi_smha_t{
                /*.embed_dim=*/ 512,
                /*.num_heads=*/ 8,
                /*.cross_attention=*/ false,
                /*.cache_cross_attention=*/ true,
                /*.causal=*/ true,
                /*.rope_max_period=*/ 10000,
                /*.context=*/ 250,
                /*.weights_per_step=*/ 0,
                /*.weights_per_step_schedule=*/ {},
                /*.in_projs=*/ { new torch_nn_linear_t },
                /*.out_projs=*/ { new torch_nn_linear_t }
            }, /*.layer_scale_1=*/ new moshi_layer_scale_t,
            /*.norm_cross=*/ NULL,
            /*.cross_attention=*/ NULL,
            /*.norm2_rms=*/ NULL,
            /*.norm2=*/ new torch_nn_layer_norm_t{ /*.eps=*/ 0.000000 },
            /*.weights_per_step_schedule=*/ {},
            /*.gating=*/ {},
            /*.linear1=*/ new torch_nn_linear_t,
            /*.linear2=*/ new torch_nn_linear_t,
            /*.layer_scale_2=*/ new moshi_layer_scale_t
        }, new moshi_streaming_transformer_layer_t{
            /*.norm1_rms=*/ NULL,
            /*.norm1=*/ new torch_nn_layer_norm_t{ /*.eps=*/ 0.000000 },
            /*self_attn=*/ new moshi_smha_t{
                /*.embed_dim=*/ 512,
                /*.num_heads=*/ 8,
                /*.cross_attention=*/ false,
                /*.cache_cross_attention=*/ true,
                /*.causal=*/ true,
                /*.rope_max_period=*/ 10000,
                /*.context=*/ 250,
                /*.weights_per_step=*/ 0,
                /*.weights_per_step_schedule=*/ {},
                /*.in_projs=*/ { new torch_nn_linear_t },
                /*.out_projs=*/ { new torch_nn_linear_t }
            }, /*.layer_scale_1=*/ new moshi_layer_scale_t,
            /*.norm_cross=*/ NULL,
            /*.cross_attention=*/ NULL,
            /*.norm2_rms=*/ NULL,
            /*.norm2=*/ new torch_nn_layer_norm_t{ /*.eps=*/ 0.000000 },
            /*.weights_per_step_schedule=*/ {},
            /*.gating=*/ {},
            /*.linear1=*/ new torch_nn_linear_t,
            /*.linear2=*/ new torch_nn_linear_t,
            /*.layer_scale_2=*/ new moshi_layer_scale_t
        }, new moshi_streaming_transformer_layer_t{
            /*.norm1_rms=*/ NULL,
            /*.norm1=*/ new torch_nn_layer_norm_t{ /*.eps=*/ 0.000000 },
            /*self_attn=*/ new moshi_smha_t{
                /*.embed_dim=*/ 512,
                /*.num_heads=*/ 8,
                /*.cross_attention=*/ false,
                /*.cache_cross_attention=*/ true,
                /*.causal=*/ true,
                /*.rope_max_period=*/ 10000,
                /*.context=*/ 250,
                /*.weights_per_step=*/ 0,
                /*.weights_per_step_schedule=*/ {},
                /*.in_projs=*/ { new torch_nn_linear_t },
                /*.out_projs=*/ { new torch_nn_linear_t }
            }, /*.layer_scale_1=*/ new moshi_layer_scale_t,
            /*.norm_cross=*/ NULL,
            /*.cross_attention=*/ NULL,
            /*.norm2_rms=*/ NULL,
            /*.norm2=*/ new torch_nn_layer_norm_t{ /*.eps=*/ 0.000000 },
            /*.weights_per_step_schedule=*/ {},
            /*.gating=*/ {},
            /*.linear1=*/ new torch_nn_linear_t,
            /*.linear2=*/ new torch_nn_linear_t,
            /*.layer_scale_2=*/ new moshi_layer_scale_t
        },
    }};

    auto mimi_decoder = new moshi_seanet_decoder_t{
        /*.model_0=*/ new moshi_streaming_conv_1d_t{
            /*.in_channels=*/ 512,
            /*.out_channels=*/ 1024,
            /*.kernel_size=*/ 7,
        },
        /*.model_2=*/ new moshi_streaming_conv_transpose_1d_t{
            /*.in_channels=*/ 1024,
            /*.out_channels=*/ 512,
            /*.kernel_size=*/ 16,
            /*.stride=*/ 8,
            /*.groups=*/ 1,

        },
        /*.model_3=*/ new moshi_seanet_resnet_block_t{
            /*.block_1=*/ new moshi_streaming_conv_1d_t{
                /*.in_channels=*/ 512,
                /*.out_channels=*/ 256,
                /*.kernel_size=*/ 3,
            },
            /*.block_3=*/ new moshi_stateless_conv_1d_t{
                /*.in_channels=*/ 256,
                /*.out_channels=*/ 512,
                /*.kernel_size=*/ 1,
            }
        },
        /*.model_5=*/ new moshi_streaming_conv_transpose_1d_t{
            /*.in_channels=*/ 512,
            /*.out_channels=*/ 256,
            /*.kernel_size=*/ 12,
            /*.stride=*/ 6,
            /*.groups=*/ 1,

        },
        /*.model_6=*/ new moshi_seanet_resnet_block_t{
            /*.block_1=*/ new moshi_streaming_conv_1d_t{
                /*.in_channels=*/ 256,
                /*.out_channels=*/ 128,
                /*.kernel_size=*/ 3,
            },
            /*.block_3=*/ new moshi_stateless_conv_1d_t{
                /*.in_channels=*/ 128,
                /*.out_channels=*/ 256,
                /*.kernel_size=*/ 1,
            }
        },
        /*.model_8=*/ new moshi_streaming_conv_transpose_1d_t{
            /*.in_channels=*/ 256,
            /*.out_channels=*/ 128,
            /*.kernel_size=*/ 10,
            /*.stride=*/ 5,
            /*.groups=*/ 1,

        },
        /*.model_9=*/ new moshi_seanet_resnet_block_t{
            /*.block_1=*/ new moshi_streaming_conv_1d_t{
                /*.in_channels=*/ 128,
                /*.out_channels=*/ 64,
                /*.kernel_size=*/ 3,
            },
            /*.block_3=*/ new moshi_stateless_conv_1d_t{
                /*.in_channels=*/ 64,
                /*.out_channels=*/ 128,
                /*.kernel_size=*/ 1,
            }
        },
        /*.model_11=*/ new moshi_streaming_conv_transpose_1d_t{
            /*.in_channels=*/ 128,
            /*.out_channels=*/ 64,
            /*.kernel_size=*/ 8,
            /*.stride=*/ 4,
            /*.groups=*/ 1,

        },
        /*.model_12=*/ new moshi_seanet_resnet_block_t{
            /*.block_1=*/ new moshi_streaming_conv_1d_t{
                /*.in_channels=*/ 64,
                /*.out_channels=*/ 32,
                /*.kernel_size=*/ 3,
            },
            /*.block_3=*/ new moshi_stateless_conv_1d_t{
                /*.in_channels=*/ 32,
                /*.out_channels=*/ 64,
                /*.kernel_size=*/ 1,
            }
        },
        /*.model_14=*/ new moshi_streaming_conv_1d_t{
            /*.in_channels=*/ 64,
            /*.out_channels=*/ 1,
            /*.kernel_size=*/ 3,
        }
    };

    auto mimi = new moshi_mimi_t;
    mimi->quantizer = mimi_quantizer;
    mimi->upsample = mimi_upsample_convtr;
    mimi->decoder_transformer = mimi_decoder__transformer;
    mimi->decoder = mimi_decoder;
    mimi->sample_rate = 24000;
    return mimi;
}


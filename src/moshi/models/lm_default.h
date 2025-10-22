#pragma once

moshi_lmmodel_t * moshi_lmmodel_alloc_default( config_t * config ) {
    //int64_t dim_feedforward = config->hidden_scale * config->dim;
    
    assert( config->positional_embedding == "rope" );
    auto lm_transformer = new moshi_streaming_transformer_t;
    lm_transformer->layers.resize( config->num_layers );
    for ( int64_t i = 0; i < config->num_layers; i++ ) {
        moshi_smha_t * cross_attention = NULL;
        torch_nn_layer_norm_t * norm_cross = NULL;
        if ( config->cross_attention ) {
            cross_attention = new moshi_smha_t{
                /*.embed_dim=*/ (int)config->dim,
                /*.num_heads=*/ (int)config->num_heads,
                /*.cross_attention=*/ true,
                /*.cache_cross_attention=*/ true,
                /*.causal=*/ false,
                /*.rope_max_period=*/ 0,
                /*.context=*/ 0,
                /*.weights_per_step=*/ 0,
                /*.weights_per_step_schedule=*/ {},
                /*.in_projs=*/ {new torch_nn_linear_t},
                /*.out_projs=*/ {new torch_nn_linear_t}
            };
            norm_cross = new torch_nn_layer_norm_t{ /*.eps=*/ 0.000000 };
        }
        auto layer = new moshi_streaming_transformer_layer_t{
            /*.norm1_rms=*/ new moshi_rms_norm_t{ /*.eps=*/ 0.000000 },
            /*.norm1=*/ NULL,
            /*self_attn=*/ new moshi_smha_t{
                /*.embed_dim=*/ (int)config->dim,
                /*.num_heads=*/ (int)config->num_heads,
                /*.cross_attention=*/ false,
                /*.cache_cross_attention=*/ true,
                /*.causal=*/ config->causal,
                /*.rope_max_period=*/ (int)config->max_period,
                /*.context=*/ (int)config->context,
                /*.weights_per_step=*/ 0,
                /*.weights_per_step_schedule=*/ {},
                /*.in_projs=*/ {new torch_nn_linear_t},
                /*.out_projs=*/ {new torch_nn_linear_t}
            },
            /*.layer_scale_1=*/ NULL,
            /*.norm_cross=*/ norm_cross,
            /*.cross_attention=*/ cross_attention,
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
        };
        lm_transformer->layers[i] = layer;
    }

    int depformer_num_weights = 1;
    if ( config->depformer_weights_per_step_schedule.size() ) {
        auto max = config->depformer_weights_per_step_schedule[0];
        for ( size_t i = 0; i < config->depformer_weights_per_step_schedule.size(); i++ )
            if ( max < config->depformer_weights_per_step_schedule[i] )
                max = config->depformer_weights_per_step_schedule[i];
        depformer_num_weights = max + 1;
    }

    auto lm_depformer = new moshi_streaming_transformer_t;
    lm_depformer->layers.resize( config->depformer_num_layers );
    for ( int64_t i = 0; i < config->depformer_num_layers; i++ ) {
        auto layer = new moshi_streaming_transformer_layer_t{
            /*.norm1_rms=*/ new moshi_rms_norm_t{ /*.eps=*/ 0.000000 },
            /*.norm1=*/ NULL,
            /*self_attn=*/ new moshi_smha_t{
                /*.embed_dim=*/ (int)config->depformer_dim,
                /*.num_heads=*/ (int)config->depformer_num_heads,
                /*.cross_attention=*/ false,
                /*.cache_cross_attention=*/ true,
                /*.causal=*/ config->causal,
                /*.rope_max_period=*/ 0,
                /*.context=*/ 0,
                /*.weights_per_step=*/ (int)config->depformer_weights_per_step_schedule.size(),
                /*.weights_per_step_schedule=*/ {},
                /*.in_projs=*/ {},
                /*.out_projs=*/ {}
            },
            /*.layer_scale_1=*/ NULL,
            /*.norm_cross=*/ NULL,
            /*.cross_attention=*/ NULL,
            /*.norm2_rms=*/ new moshi_rms_norm_t{ /*.eps=*/ 0.000000 },
            /*.norm2=*/ NULL,
            /*.weights_per_step_schedule=*/ {},
            /*.gating=*/ {},
            /*.linear1=*/ NULL,
            /*.linear2=*/ NULL,
            /*.layer_scale_2=*/ NULL
        };
        layer->self_attn->weights_per_step_schedule.resize( config->depformer_weights_per_step_schedule.size() );
        layer->weights_per_step_schedule.resize( config->depformer_weights_per_step_schedule.size() );
        for ( size_t j =0; j < config->depformer_weights_per_step_schedule.size(); j++ ) {
            layer->self_attn->weights_per_step_schedule[j] = config->depformer_weights_per_step_schedule[j];
            layer->weights_per_step_schedule[j] = config->depformer_weights_per_step_schedule[j];
        }
        layer->self_attn->in_projs.resize( depformer_num_weights );
        layer->self_attn->out_projs.resize( depformer_num_weights );
        layer->gating.resize( depformer_num_weights );
        for (int j = 0; j < depformer_num_weights; j++ ) {
            layer->self_attn->in_projs[j] = new torch_nn_linear_t;
            layer->self_attn->out_projs[j] = new torch_nn_linear_t;
            layer->gating[j] = new moshi_activation_gating_t{
                /*.linear_in=*/new torch_nn_linear_t,
                /*.linear_out=*/new torch_nn_linear_t
            };
        }

        lm_depformer->layers[i] = layer;
    }
    auto lmmodel = new moshi_lmmodel_t;
    lmmodel->n_q = config->n_q;
    lmmodel->dep_q = config->dep_q;
    lmmodel->card = config->card;
    lmmodel->text_card = config->text_card;
    lmmodel->delays.resize( config->delays.size() );
    int max_delay = config->delays[0];
    for (size_t i = 0; i < config->delays.size(); i++ ) {
        if ( config->delays[i] > max_delay )
            max_delay = config->delays[i];
        lmmodel->delays[i] = config->delays[i];
    }
    lmmodel->max_delay = max_delay;
    lmmodel->dim = config->dim;
    lmmodel->depformer_weights_per_step_schedule.resize( config->depformer_weights_per_step_schedule.size() );
    for (size_t i = 0; i < config->depformer_weights_per_step_schedule.size(); i++ )
        lmmodel->depformer_weights_per_step_schedule[i] = config->depformer_weights_per_step_schedule[i];
    lmmodel->emb.resize( config->n_q );
    for (int64_t i = 0; i < config->n_q; i++ )
        lmmodel->emb[i] = new moshi_scaled_embedding_t{NULL};
    lmmodel->text_emb = new moshi_scaled_embedding_demux_t{
        /*.num_embeddings=*/ (int)config->text_card + 1,
        /*.out1=*/ new torch_nn_linear_t,
        /*.out2=*/ new torch_nn_linear_t
    };
    lmmodel->text_linear = new torch_nn_linear_t;
    lmmodel->transformer = lm_transformer;
    lmmodel->out_norm = new moshi_rms_norm_t{1e-08};
    lmmodel->depformer_multi_linear = config->depformer_multi_linear;
    lmmodel->depformer_in.resize( depformer_num_weights );
    for ( int64_t i = 0; i < depformer_num_weights; i++ )
        lmmodel->depformer_in[i] = new torch_nn_linear_t;
    lmmodel->depformer_emb.resize( config->n_q - 1 );
    for ( int64_t i = 0; i < config->n_q - 1; i++ )
        lmmodel->depformer_emb[i] = new moshi_scaled_embedding_t{new torch_nn_linear_t};
    lmmodel->depformer_text_emb = new moshi_scaled_embedding_demux_t{
        /*.num_embeddings=*/ 8001,
        /*.out1=*/ new torch_nn_linear_t,
        /*.out2=*/ new torch_nn_linear_t
        // NOTE: the original had low_rank for no reason, never gets used in demux
    };
    lmmodel->depformer = lm_depformer;
    lmmodel->linears.resize( config->dep_q );
    for ( int64_t i = 0; i < config->dep_q; i++ )
        lmmodel->linears[i] = new torch_nn_linear_t;
    lmmodel->num_codebooks = config->n_q + 1;
    lmmodel->num_audio_codebooks= config->n_q;
    lmmodel->audio_offset= 1;
    
    return lmmodel;
}


moshi_mimi_t * moshi_mimi_alloc_default( int n_q, bool encoder = true ) {
    //mimi.quantizer.
    auto mimi_quantizer = new moshi_split_rvq_t{
        /*.n_q_semantic=*/ 1,
        /*.rvq_first=*/ new moshi_rvq_t{
            /*.n_q=*/1, // n_q_semantic
            /*.vq=*/new moshi_residual_vq_t{ /*.layers=*/ {
                new moshi_vq_t{ new moshi_EuclideanCodebook_t },
            }},
            /*.output_proj=*/ new torch_nn_conv1d_t,
            /*.input_proj=*/ new torch_nn_conv1d_t
        }, /*.rvq_rest=*/ new moshi_rvq_t{
            /*.n_q=*/n_q - 1, // n_q - n_q_semantic
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
            /*.output_proj=*/ new torch_nn_conv1d_t,
            /*.input_proj=*/ new torch_nn_conv1d_t
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
            /*.stride=*/ 1,
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
                /*.stride=*/ 1,
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
                /*.stride=*/ 1,
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
                /*.stride=*/ 1,
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
                /*.stride=*/ 1,
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
            /*.stride=*/ 1,
        }
    };

    auto mimi_encoder__transformer = (moshi_streaming_transformer_t*)NULL;
    auto mimi_encoder = (moshi_seanet_encoder_t*)NULL;
    auto mimi_downsample_conv = (moshi_streaming_conv_1d_t*)NULL;
    if (encoder) {
        mimi_encoder = new moshi_seanet_encoder_t{
            /*.model_0=*/ new moshi_streaming_conv_1d_t{
                /*.in_channels=*/ 1,
                /*.out_channels=*/ 64,
                /*.kernel_size=*/ 7,
                /*.stride=*/ 1,
            },
            /*.model_1=*/ new moshi_seanet_resnet_block_t{
                /*.block_1=*/ new moshi_streaming_conv_1d_t{
                    /*.in_channels=*/ 64,
                    /*.out_channels=*/ 32,
                    /*.kernel_size=*/ 3,
                    /*.stride=*/ 1,
                },
                /*.block_3=*/ new moshi_stateless_conv_1d_t{
                    /*.in_channels=*/ 32,
                    /*.out_channels=*/ 64,
                    /*.kernel_size=*/ 1,
                }
            },
            /*.model_3=*/ new moshi_streaming_conv_1d_t{
                /*.in_channels=*/ 64,
                /*.out_channels=*/ 128,
                /*.kernel_size=*/ 8,
                /*.stride=*/ 4,
            },
            /*.model_4=*/ new moshi_seanet_resnet_block_t{
                /*.block_1=*/ new moshi_streaming_conv_1d_t{
                    /*.in_channels=*/ 128,
                    /*.out_channels=*/ 64,
                    /*.kernel_size=*/ 3,
                    /*.stride=*/ 1,
                },
                /*.block_3=*/ new moshi_stateless_conv_1d_t{
                    /*.in_channels=*/ 64,
                    /*.out_channels=*/ 128,
                    /*.kernel_size=*/ 1,
                }
            },
            /*.model_6=*/ new moshi_streaming_conv_1d_t{
                /*.in_channels=*/ 128,
                /*.out_channels=*/ 256,
                /*.kernel_size=*/ 10,
                /*.stride=*/ 5,
            },
            /*.model_7=*/ new moshi_seanet_resnet_block_t{
                /*.block_1=*/ new moshi_streaming_conv_1d_t{
                    /*.in_channels=*/ 256,
                    /*.out_channels=*/ 128,
                    /*.kernel_size=*/ 3,
                    /*.stride=*/ 1,
                },
                /*.block_3=*/ new moshi_stateless_conv_1d_t{
                    /*.in_channels=*/ 128,
                    /*.out_channels=*/ 256,
                    /*.kernel_size=*/ 1,
                }
            },
            /*.model_9=*/ new moshi_streaming_conv_1d_t{
                /*.in_channels=*/ 256,
                /*.out_channels=*/ 512,
                /*.kernel_size=*/ 12,
                /*.stride=*/ 6,
            },
            /*.model_10=*/ new moshi_seanet_resnet_block_t{
                /*.block_1=*/ new moshi_streaming_conv_1d_t{
                    /*.in_channels=*/ 512,
                    /*.out_channels=*/ 256,
                    /*.kernel_size=*/ 3,
                    /*.stride=*/ 1,
                },
                /*.block_3=*/ new moshi_stateless_conv_1d_t{
                    /*.in_channels=*/ 256,
                    /*.out_channels=*/ 512,
                    /*.kernel_size=*/ 1,
                }
            },
            /*.model_12=*/ new moshi_streaming_conv_1d_t{
                /*.in_channels=*/ 512,
                /*.out_channels=*/ 1024,
                /*.kernel_size=*/ 16,
                /*.stride=*/ 8,
            },
            /*.model_14=*/ new moshi_streaming_conv_1d_t{
                /*.in_channels=*/ 1024,
                /*.out_channels=*/ 512,
                /*.kernel_size=*/ 3,
                /*.stride=*/ 1,
            }
        };
        
        mimi_encoder__transformer = new moshi_streaming_transformer_t;
        mimi_encoder__transformer->layers.resize( 8 );
        for ( int64_t i = 0; i < 8; i++ ) {
            auto layer = new moshi_streaming_transformer_layer_t{
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
                },
                /*.layer_scale_1=*/ new moshi_layer_scale_t,
                /*.norm_cross=*/ NULL,
                /*.cross_attention=*/ NULL,
                /*.norm2_rms=*/ NULL,
                /*.norm2=*/ new torch_nn_layer_norm_t{ /*.eps=*/ 0.000000 },
                /*.weights_per_step_schedule=*/ {},
                /*.gating=*/ {},
                /*.linear1=*/ new torch_nn_linear_t,
                /*.linear2=*/ new torch_nn_linear_t,
                /*.layer_scale_2=*/ new moshi_layer_scale_t
            };
            mimi_encoder__transformer->layers[i] = layer;
        }
        mimi_downsample_conv = new moshi_streaming_conv_1d_t{
            /*.in_channels=*/ 512,
            /*.out_channels=*/ 512,
            /*.kernel_size=*/ 4,
            /*.stride=*/ 2,
        };
    }

    auto mimi = new moshi_mimi_t;
    mimi->sample_rate = 24000;
    mimi->quantizer = mimi_quantizer;
    // decoder
    mimi->upsample = mimi_upsample_convtr;
    mimi->decoder_transformer = mimi_decoder__transformer;
    mimi->decoder = mimi_decoder;
    // encoder
    mimi->downsample = mimi_downsample_conv;
    mimi->encoder_transformer = mimi_encoder__transformer;
    mimi->encoder = mimi_encoder;
    return mimi;
}


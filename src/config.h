#pragma once

int util_json_object_log_skip( const_str_t & json, int offset, const char * label ) {
    printf("%s\n", label);
    offset = json_object_parse(json, offset, [](
            const_str_t & json, int offset,
            int key_start, int key_length
        ) {
		const char * key = json.s + key_start;
        std::string key2;
        key2.assign( key, key_length );
        printf( "        case %d:   if (strncmp( key, \"%s\", %d) == 0) {\n",
            key_length, key2.c_str(), key_length );
            
        return json_skip_value( json, offset );
    });
    return offset;
}

int config_fuser_parse( const_str_t & json, int offset, config_fuser_t & fuser ) {
    offset = json_object_parse(json, offset, [&fuser](
            const_str_t & json, int offset,
            int key_start, int key_length
        ) {
		const char * key = json.s + key_start;
        //offset = str_skip_whitespaces( json, offset );
        switch( key_length ) {
        case 23:   if (strncmp( key, "cross_attention_pos_emb", 23) == 0) {
                return json_bool_parse( json, offset, fuser.cross_attention_pos_emb );
            }
            break;
        case 29:   if (strncmp( key, "cross_attention_pos_emb_scale", 29) == 0) {
                return json_float_parse( json, offset, fuser.cross_attention_pos_emb_scale );
            }
            break;
        case 3:   if (strncmp( key, "sum", 3) == 0) {
                return json_string_array_parse( json, offset, fuser.sum );
            }
            break;
        case 7:   if (strncmp( key, "prepend", 7) == 0) {
                // skip, unknown/unused
            }
            break;
        case 5:   if (strncmp( key, "cross", 5) == 0) {
                return json_string_array_parse( json, offset, fuser.cross );
            }
            break;
        }
        return json_skip_value( json, offset );
    });
    return offset;
}

int config_tts_parse( const_str_t & json, int offset, config_tts_t & tts_config ) {
    offset = json_object_parse(json, offset, [&tts_config](
            const_str_t & json, int offset,
            int key_start, int key_length
        ) {
		const char * key = json.s + key_start;
        switch( key_length ) {
        case 11:   if (strncmp( key, "audio_delay", 11) == 0) {
                return json_float_parse( json, offset, tts_config.audio_delay );
            }
            break;
        case 19:   if (strncmp( key, "second_stream_ahead", 19) == 0) {
                return json_int64_parse( json, offset, tts_config.second_stream_ahead );
            }
            break;
        }
        return json_skip_value( json, offset );
    });
    return offset;
}

int config_stt_parse( const_str_t & json, int offset, config_stt_t & stt_config ) {
    offset = json_object_parse(json, offset, [&stt_config](
            const_str_t & json, int offset,
            int key_start, int key_length
        ) {
		const char * key = json.s + key_start;
        switch( key_length ) {
        case 19:   if (strncmp( key, "audio_delay_seconds", 19) == 0) {
                return json_float_parse( json, offset, stt_config.audio_delay_seconds );
            }
            break;
        case 28:   if (strncmp( key, "audio_silence_prefix_seconds", 28) == 0) {
                return json_float_parse( json, offset, stt_config.audio_silence_prefix_seconds );
            }
            break;
        }
        return json_skip_value( json, offset );
    });
    return offset;
}

int config_model_id_parse( const_str_t & json, int offset, config_model_id_t & model_id ) {
    offset = json_object_parse(json, offset, [&model_id](
            const_str_t & json, int offset,
            int key_start, int key_length
        ) {
		const char * key = json.s + key_start;
        //offset = str_skip_whitespaces( json, offset );
        switch( key_length ) {
        case 3:   if (strncmp( key, "sig", 3) == 0) {
                return json_string_parse( json, offset, model_id.sig );
            }
            break;
        case 5:   if (strncmp( key, "epoch", 5) == 0) {
                return json_int64_parse( json, offset, model_id.epoch );
            }
            break;
        }
        return json_skip_value( json, offset );
    });
    return offset;
}

int config_lm_gen_parse( const_str_t & json, int offset, config_lm_gen_t & lm_gen_config ) {
    offset = json_object_parse(json, offset, [&lm_gen_config](
            const_str_t & json, int offset,
            int key_start, int key_length
        ) {
		const char * key = json.s + key_start;
        //offset = str_skip_whitespaces( json, offset );
        switch( key_length ) {
        case 4:   if (strncmp( key, "temp", 4) == 0) {
                return json_float_parse( json, offset, lm_gen_config.temp );
            }
            break;
        case 9:   if (strncmp( key, "temp_text", 9) == 0) {
                return json_float_parse( json, offset, lm_gen_config.temp_text );
            }
            break;
        case 5:   if (strncmp( key, "top_k", 5) == 0) {
                return json_int64_parse( json, offset, lm_gen_config.top_k );
            }
            break;
        case 10:   if (strncmp( key, "top_k_text", 10) == 0) {
                return json_int64_parse( json, offset, lm_gen_config.top_k_text );
            }
            break;
        }
        return json_skip_value( json, offset );
    });
    return offset;
}

int moshi_get_config( moshi_config_t * config, const char * filename ) {
    config->demux_second_stream = false;
    config->moshi_name = "model.safetensors";
    config->stt_config.audio_silence_prefix_seconds = 1.0;
    config->stt_config.audio_delay_seconds = 5.0;
    config->extra_heads_num_heads = 0;
    //config->extra_heads_dim = 6;

	auto f = fopen( filename, "rb" );
	if ( ! f ) {
        fprintf( stderr, "error: failed to open %s\n", filename );
        return -1;
    }
    // get file length
	fseek( f, 0, SEEK_END );
	auto length = ftell( f );
	fseek( f, 0, SEEK_SET );
	if ( length <= 0 ) {
        fprintf( stderr, "error: empty file %s\n", filename );
        return -1;
    }
    // read file
	std::vector<char> raw( length );
	auto read = fread(raw.data(), length, 1, f);
	fclose( f );
    if ( read != 1 ) {
        fprintf( stderr, "error: failed to read %s\n", filename );
        return -1;
    }

	const_str_t json = {raw.data(), (int)length};

    int offset = str_skip_whitespaces(json, 0);
    if (offset >= length || json.s[offset] != '{') {
        fprintf( stderr, "error: did not find expected json object" );
        return -1;
	}

    offset = json_object_parse(json, offset, [config](
            const_str_t & json, int offset,
            int key_start, int key_length
        ) {
		const char * key = json.s + key_start;
        switch( key_length ) {
        case 3:    if (strncmp( key, "n_q", 3) == 0 ) {
                return json_int64_parse( json, offset, config->n_q );
            } else if (strncmp( key, "dim", 3) == 0 ) {
                return json_int64_parse( json, offset, config->dim );
            }
            break;
        case 4:    if (strncmp( key, "card", 4) == 0 ) {
                return json_int64_parse( json, offset, config->card );
            } else if (strncmp( key, "norm", 4) == 0 ) {
                return json_string_parse( json, offset, config->norm );
            }
            break;
        case 5:    if (strncmp( key, "dep_q", 5) == 0 ) {
                return json_int64_parse( json, offset, config->dep_q );
            } else if (strncmp( key, "fuser", 5) == 0 ) {
                return config_fuser_parse( json, offset, config->fuser );
            }
            break;
        case 6:    if (strncmp( key, "delays", 6) == 0 ) {
                return json_int64_array_parse( json, offset, config->delays );
            } else if (strncmp( key, "causal", 6) == 0 ) {
                return json_bool_parse( json, offset, config->causal );
            } else if (strncmp( key, "gating", 6) == 0 ) {
                return json_string_parse( json, offset, config->gating );
            }
            break;
        case 7:    if (strncmp( key, "context", 7) == 0 ) {
                return json_int64_parse( json, offset, config->context );
            }
            break;
        case 8:    if (strncmp( key, "model_id", 8) == 0 ) {
                return config_model_id_parse( json, offset, config->model_id );
            }
            break;
        case 9:    if (strncmp( key, "text_card", 9) == 0 ) {
                return json_int64_parse( json, offset, config->text_card );
            } else if (strncmp( key, "num_heads", 9) == 0 ) {
                return json_int64_parse( json, offset, config->num_heads );
            } else if (strncmp( key, "mimi_name", 9) == 0 ) {
                return json_string_parse( json, offset, config->mimi_name );
            }
            break;
        case 10:   if (strncmp( key, "num_layers", 10) == 0 ) {
                return json_int64_parse( json, offset, config->num_layers );
            } else if (strncmp( key, "max_period", 10) == 0 ) {
                return json_int64_parse( json, offset, config->max_period );
            } else if (strncmp( key, "tts_config", 10) == 0 ) {
                return config_tts_parse( json, offset, config->tts_config );
            } else if (strncmp( key, "stt_config", 10) == 0 ) {
                return config_stt_parse( json, offset, config->stt_config );
            } else if (strncmp( key, "model_type", 10) == 0 ) {
                return json_string_parse( json, offset, config->model_type );
            } else if (strncmp( key, "moshi_name", 10) == 0 ) {
                return json_string_parse( json, offset, config->moshi_name );
            }
            break;
        case 11:   if (strncmp( key, "layer_scale", 11) == 0 ) {
                // unknown/unused skip for now
                return json_skip_value( json, offset );
            }
            break;
        case 12:   if (strncmp( key, "hidden_scale", 12) == 0 ) {
                return json_float_parse( json, offset, config->hidden_scale );
            } else if (strncmp( key, "conditioners", 12) == 0 ) {
                // TODO: object parser
            }
            break;
        case 13:   if (strncmp( key, "depformer_dim", 13) == 0 ) {
                return json_int64_parse( json, offset, config->depformer_dim );
            } else if (strncmp( key, "text_card_out", 13) == 0 ) {
                // seems this override text_card in one case if not null
                // but the data is baked in the weights so we can skip it
                return json_skip_value( json, offset );
            } else if (strncmp( key, "lm_gen_config", 13) == 0 ) {
                return config_lm_gen_parse( json, offset, config->lm_gen_config );
            }
            break;
        case 14:   if (strncmp( key, "tokenizer_name", 14) == 0 ) {
                return json_string_parse( json, offset, config->tokenizer_name );
            }
            break;
        case 15:   if (strncmp( key, "cross_attention", 15) == 0 ) {
                return json_bool_parse( json, offset, config->cross_attention );
            }/* else if (strncmp( key, "extra_heads_dim", 15) == 0 ) { // this will come from weights
                return json_int64_parse( json, offset, config->extra_heads_dim );
            }*/
            break;
        case 17:   if (strncmp( key, "depformer_pos_emb", 17) == 0 ) {
                return json_string_parse( json, offset, config->depformer_pos_emb );
            }
            break;
        case 19:   if (strncmp( key, "depformer_num_heads", 19) == 0 ) {
                return json_int64_parse( json, offset, config->depformer_num_heads );
            } else if (strncmp( key, "demux_second_stream", 19) == 0 ) {
                return json_bool_parse( json, offset, config->demux_second_stream );
            }
            break;
        case 21:   if (strncmp( key, "extra_heads_num_heads", 21) == 0 ) {
                return json_int64_parse( json, offset, config->extra_heads_num_heads );
            }
            break;
        case 20:   if (strncmp( key, "positional_embedding", 20) == 0 ) {
                return json_string_parse( json, offset, config->positional_embedding );
            } else if (strncmp( key, "depformer_num_layers", 20) == 0 ) {
                return json_int64_parse( json, offset, config->depformer_num_layers );
            }
            break;
        case 22:   if (strncmp( key, "depformer_multi_linear", 22) == 0 ) {
                return json_bool_parse( json, offset, config->depformer_multi_linear );
            }
            break;
        case 24:   if (strncmp( key, "existing_text_padding_id", 24) == 0 ) {
                return json_int64_parse( json, offset, config->existing_text_padding_id );
            }
            break;
        // this is not used anywhere, its in the dimension of the weight files
        //case 25:   if (strncmp( key, "depformer_dim_feedforward", 25) == 0 ) {
        //        return json_int64_parse( json, offset, config->depformer_dim_feedforward );
        //    }
        //    break;
        case 26:   if (strncmp( key, "depformer_weights_per_step", 26) == 0 ) {
                return json_bool_parse( json, offset, config->depformer_weights_per_step );
            }
            break;
        case 29:   if (strncmp( key, "depformer_low_rank_embeddings", 29) == 0 ) {
                return json_int64_parse( json, offset, config->depformer_low_rank_embeddings );
            }
            break;
        case 35:   if (strncmp( key, "depformer_weights_per_step_schedule", 35) == 0 ) {
                return json_int64_array_parse( json, offset, config->depformer_weights_per_step_schedule );
            }
            break;
        }

        return json_skip_value( json, offset );
    } );
    if ( offset == -1 ) {
        fprintf( stderr, "error: reading config %s\n", filename );
        return -1;
    }

    return 0;
}

moshi_config_t * get_config( const char * filename ) {
    auto config = new moshi_config_t;
    moshi_get_config( config, filename );
    return config;
}




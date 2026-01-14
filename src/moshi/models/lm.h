#pragma once

#include <deque>

class TokenIds {
public:
    int card;
    const int new_word = 0;
    const int pad = 3;
    const int main = 1;
    const int other = 2;
    const int zero = -1;
    const int ungenerated = -2;

    TokenIds(int card = 8001) {
        this->card = card;
    }
};

class State {
public:
    int remaining_padding;
    int forced_padding;
    int end_step;
    std::deque<Entry> entries;
    std::deque<int> queued;
    std::deque<int> lookahead_queued;

    std::vector<int> get_tokens_ahead(int lookahead) {
        // assert lookahead > 0
        for (auto entry : entries) {
            if (!entry.tokens.size())
                continue;
            lookahead -= 1;
            if (lookahead != 0)
                continue;
            return entry.tokens;
        }
        return {};
    }
    
    bool is_empty() {
        if (entries.size())
            return false;
        if (queued.size())
            return false;
        if (lookahead_queued.size())
            return false;
        return true;
    }
};

int64_t g_last_token_time;

class StateMachine {
public:
    TokenIds token_ids;
    int second_stream_ahead;
    int max_padding;
    int initial_padding;
    bool logging = false;

    StateMachine(
            int text_card,
            int second_stream_ahead = 0,
            int max_padding = 6,
            int initial_padding = 2) {
        token_ids.card = text_card;
        this->second_stream_ahead = second_stream_ahead;
        this->max_padding = max_padding;
        this->initial_padding = initial_padding;
    }

    State * new_state(std::deque<Entry> &entries) {
        auto state = new State(
            /*remaining_padding=*/initial_padding,
            /*forced_padding=*/initial_padding,
            /*end_step*/-1,
            entries
        );
        return state;
    }

    State * new_state() {
        auto state = new State(
            /*remaining_padding=*/initial_padding,
            /*forced_padding=*/initial_padding,
            /*end_step*/-1
        );
        return state;
    }
    
    void reset_state( State * state ) {
        state->remaining_padding = initial_padding;
        state->forced_padding = initial_padding;
        state->end_step = -1;
        state->entries.clear();
        state->queued.clear();
        state->lookahead_queued.clear();
    }

    int process(int step, State * state, int token) {

        if (token != token_ids.new_word && token != token_ids.pad)
            token = token_ids.pad;

        if (state->queued.size())
            // Some text tokens are yet to be fed, we must PAD.
            token = token_ids.pad;
        else if (state->forced_padding > 0)
            // We are forced to pad, we must PAD.
            token = token_ids.pad;
        else if (state->remaining_padding <= 0)
            // We are not allowed to pad, we must ask for a new WORD.
            token = token_ids.new_word;

        if (token == token_ids.new_word) {
            if (state->entries.size()) {
                auto entry = state->entries.front();
                state->entries.pop_front();

                if ( logging ) {
                    auto new_token_time = ggml_time_ms();
                    auto last_token_time = g_last_token_time;
                    if ( entry.time != 0 && entry.time > last_token_time )
                        last_token_time = entry.time;
                    printf("\"%s\" %.4f\n", entry.text.c_str(), (new_token_time - last_token_time) / 1000.f);
                    g_last_token_time = new_token_time;
                }

                if (entry.tokens.size()) {
                    // We queue the tokens to be fed to the model.
                    for (auto token : entry.tokens)
                        state->queued.push_back(token);
                    if (second_stream_ahead) {
                        // We queue the tokens for the N+lookahead word into the second text stream.
                        for (auto token : state->get_tokens_ahead(second_stream_ahead))
                            state->lookahead_queued.push_back(token);
                    }
                    // Entry contains a new word, we reset the max padding counter.
                    state->remaining_padding = max_padding;
                } else {
                    token = token_ids.pad;
                }
                state->forced_padding = entry.padding;
            } else {
                token = token_ids.pad;
                if (second_stream_ahead && state->end_step < 0)
                    token = token_ids.new_word;
                // Trying to consume past the last word, we reached the end.
                if (state->end_step < 0)
                    state->end_step = step;
            }
        }

        int output = 0;
        if (token == token_ids.pad) {
            // Decrement the counters for remaining and forced pads.
            if (state->remaining_padding > 0)
                state->remaining_padding -= 1;
            if (state->forced_padding > 0)
                state->forced_padding -= 1;
            if (state->queued.size()){
                // We have some text tokens to feed to the model.
                output = state->queued.front();
                state->queued.pop_front();
            } else {
                output = token_ids.pad;
            }
        } else if (token == token_ids.new_word) {
            output = token_ids.new_word;
        } else if (token == token_ids.zero) {
            output = token;
        }

        if (second_stream_ahead) {
            int second = -1;
            if (output == token_ids.new_word) {
                second = token_ids.new_word;
                if (state->queued.size()) {
                    output = state->queued.front();
                    state->queued.pop_front();
                } else {
                    output = token_ids.pad;
                }
            } else if (state->lookahead_queued.size()) {
                second = state->lookahead_queued.front();
                state->lookahead_queued.pop_front();
            }
            output = (second + 1) * token_ids.card + output;
        }
        return output;
    }
};

template<class T>
void script_to_entries(
    std::deque<Entry> &entries,
    T &tokenizer,
    TokenIds &token_ids,
    float frame_rate,
    std::vector<std::string> script,
    bool multi_speaker = true,
    int padding_between = 0
) {
    int speaker_tokens[] = {token_ids.main, token_ids.other};
    int last_speaker = -99;
    for (size_t idx = 0; idx < script.size(); idx++) {
        bool first_content = true;
        auto init_line = script[idx];
        std::string line;
        for (size_t i = 0; i < init_line.size(); i++) {
            char c = init_line[i];
            switch(c) {
            //case '’': line += '\''; break;
            case ':': line += ' '; break;
            case '(': break;
            case ')': break;
            default: line += c;
            }
        }
        // TODO: experiment with break
        // break is indicated as e.g. <break time="3s"/>
        // event_re = re.compile(r"(?:<break\s+time=\"([0-9]+(?:.[0-9]*)?)s\"\s*/?>)|(?:\s+)")
        std::string ws = " \t\r\n";
        auto cur = line.find_first_not_of(ws);
        while (cur != std::string::npos) {
            auto end = line.find_first_of(ws, cur);
            std::string word;
            if (end == std::string::npos) {
                word = line.substr(cur);
                cur = std::string::npos;
            } else {
                auto count = end - cur;
                word = line.substr(cur, count);
                cur = line.find_first_not_of(ws, end);
            }
            std::vector<int> tokens;
            tokenizer.Encode(word, &tokens);
            if (first_content) {
                int speaker = idx % 2; // len(speaker_tokens)
                if (multi_speaker && last_speaker != speaker) {
                    last_speaker = speaker;
                    std::vector<int> new_tokens(1 + tokens.size());
                    new_tokens[0] = speaker_tokens[speaker];
                    for (size_t i = 0; i < tokens.size(); i++)
                        new_tokens[i+1] = tokens[i];
                    tokens.swap(new_tokens);
                }
                first_content = false;
            }
            int padding = 0;
            if (padding_between > 0) {
                padding = padding_between + tokens.size() - 1;
                if (padding < 0) padding = 0;
            }
            entries.push_back(Entry(tokens, word, padding));
        }
    }
}

template<class T>
void script_to_state(
    State * state,
    T &tokenizer,
    TokenIds &token_ids,
    float frame_rate,
    std::vector<std::string> script,
    bool multi_speaker = true,
    int padding_between = 0
) {
    int speaker_tokens[] = {token_ids.main, token_ids.other};
    int last_speaker = -99;
    for (size_t idx = 0; idx < script.size(); idx++) {
        bool first_content = true;
        auto init_line = script[idx];
        std::string line;
        for (size_t i = 0; i < init_line.size(); i++) {
            char c = init_line[i];
            switch(c) {
            //case '’': line += '\''; break;
            case ':': line += ' '; break;
            case '(': break;
            case ')': break;
            default: line += c;
            }
        }
        // TODO: experiment with break
        // break is indicated as e.g. <break time="3s"/>
        // event_re = re.compile(r"(?:<break\s+time=\"([0-9]+(?:.[0-9]*)?)s\"\s*/?>)|(?:\s+)")
        std::string ws = " \t\r\n";
        auto cur = line.find_first_not_of(ws);
        while (cur != std::string::npos) {
            auto end = line.find_first_of(ws, cur);
            std::string word;
            if (end == std::string::npos) {
                word = line.substr(cur);
                cur = std::string::npos;
            } else {
                auto count = end - cur;
                word = line.substr(cur, count);
                cur = line.find_first_not_of(ws, end);
            }
            std::vector<int> tokens;
            tokenizer.Encode(word, &tokens);
            if (first_content) {
                int speaker = idx % 2; // len(speaker_tokens)
                if (multi_speaker && last_speaker != speaker) {
                    last_speaker = speaker;
                    std::vector<int> new_tokens(1 + tokens.size());
                    new_tokens[0] = speaker_tokens[speaker];
                    for (size_t i = 0; i < tokens.size(); i++)
                        new_tokens[i+1] = tokens[i];
                    tokens.swap(new_tokens);
                }
                first_content = false;
            }
            int padding = 0;
            if (padding_between > 0) {
                padding = padding_between + (int) tokens.size() - 1;
                if (padding < 0) padding = 0;
            }
            state->entries.push_back(Entry(tokens, word, padding));
        }
    }
}

/*************************************************************\
 *  moshi.models.LMModel
\*************************************************************/

struct moshi_lmmodel_t {
    int n_q;
    int dep_q;
    int card;
    int text_card;
    std::vector<int> delays;
    int max_delay;
    int dim;
    std::vector<int> depformer_weights_per_step_schedule;
    own_ptr_vector<moshi_scaled_embedding_t> emb;
    bool demux_second_stream;
    own_ptr<moshi_scaled_embedding_demux_t> text_emb_demux;
    own_ptr<moshi_scaled_embedding_t> text_emb;

    own_ptr<torch_nn_linear_t> text_linear;
    own_ptr<moshi_streaming_transformer_t> transformer;
    own_ptr<moshi_rms_norm_t> out_norm;
    bool depformer_multi_linear;

    own_ptr_vector<torch_nn_linear_t> depformer_in;
    own_ptr_vector<moshi_scaled_embedding_t> depformer_emb;
    own_ptr<moshi_scaled_embedding_demux_t> depformer_text_emb_demux;
    own_ptr<moshi_scaled_embedding_t> depformer_text_emb;
    own_ptr<moshi_streaming_transformer_t> depformer;

    own_ptr_vector<torch_nn_linear_t> extra_heads;

    own_ptr_vector<torch_nn_linear_t> linears;

    int num_codebooks; // n_q + 1
    int num_audio_codebooks; // n_q
    int audio_offset; // 1
    int delay_steps;
    int text_initial_token_id;
    int initial_token_id;
};

void get_weights( WeightLoader * loader, std::string path, moshi_lmmodel_t * lm ) {
    for ( size_t i = 0; i < lm->depformer_in.size(); i++ )
        get_weights( loader, path + "depformer_in."+std::to_string(i)+".", lm->depformer_in[i] );
    if ( lm->depformer ) {
        get_weights( loader, path + "depformer.", lm->depformer );
        if ( lm->demux_second_stream )
            get_weights( loader, path + "depformer_text_emb.", lm->depformer_text_emb_demux );
        else
            get_weights( loader, path + "depformer_text_emb.", lm->depformer_text_emb );
        for ( size_t i = 0; i < lm->depformer_emb.size(); i++ )
            get_weights( loader, path + "depformer_emb."+std::to_string(i)+".", lm->depformer_emb[i] );
    }
    for ( size_t i = 0; i < lm->extra_heads.size(); i++ )
        get_weights( loader, path + "extra_heads."+std::to_string(i)+".", lm->extra_heads[i] );
    for ( size_t i = 0; i < lm->linears.size(); i++ )
        get_weights( loader, path + "linears."+std::to_string(i)+".", lm->linears[i] );
    for ( size_t i = 0; i < lm->emb.size(); i++ )
        get_weights( loader, path + "emb."+std::to_string(i)+".", lm->emb[i] );
    if ( lm->demux_second_stream )
        get_weights( loader, path + "text_emb.", lm->text_emb_demux );
    else
        get_weights( loader, path + "text_emb.", lm->text_emb );
    get_weights( loader, path + "transformer.", lm->transformer);
    get_weights( loader, path + "out_norm.", lm->out_norm);
    get_weights( loader, path + "text_linear.", lm->text_linear);
}

struct lmmodel_embed_t {
    embedding_demux_t text_demux;
    embedding_t text;
    std::vector<embedding_t> audio;
};

struct moshi_lmmodel_states_t {
    own_ptr<moshi_streaming_transformer_state_t> transformer;
    own_ptr<moshi_streaming_transformer_state_t> depformer;
    GraphContext * gctx = NULL;
    ggml_tensor * transformer_out;
    ggml_tensor * sampler_out;
    lmmodel_embed_t embed;
    int  transformer_T;
};

moshi_lmmodel_states_t * moshi_lmmodel_states( StateContext * state_ctx,
        moshi_lmmodel_t * lm, ggml_tensor * k_cross ) {
    auto state = new moshi_lmmodel_states_t;
    state->transformer = moshi_streaming_transformer_state( state_ctx, lm->transformer,
        k_cross );
    if ( lm->depformer ) {
        state->depformer = moshi_streaming_transformer_state( state_ctx, lm->depformer,
            NULL );
    } else {
        state->depformer = NULL;
    }
    state_ctx->fill(GGML_NE(lm->dim), 0.f, &state->transformer_out );
    return state;
}

void init( ScratchContext * ctx, moshi_lmmodel_states_t * state,
        moshi_lmmodel_t * lm,
        ggml_tensor * condition_cross ) {
    init( ctx, state->transformer, lm->transformer, condition_cross );
    if ( state->depformer )
        init( ctx, state->depformer, lm->depformer, NULL );
}

ggml_tensor * moshi_lmmodel_forward_depformer_transform(
        ScratchContext & ctx,
        moshi_lmmodel_t * lm,
        moshi_lmmodel_states_t * states,
        int depformer_cb_index,
        ggml_tensor * last_token_input,
        ggml_tensor * transformer_out
    ) {
    //ProfileScope profile(time_depformer_us);

    auto depformer_input = transformer_out;
    int in_index = 0;
    if ( lm->depformer_multi_linear ) {
        in_index = depformer_cb_index;
        if ( lm->depformer_weights_per_step_schedule.size() )
            in_index = lm->depformer_weights_per_step_schedule[in_index];
    }

    depformer_input = torch_nn_linear( ctx, lm->depformer_in[in_index], depformer_input );

    last_token_input = ggml_cast( ctx, last_token_input, GGML_TYPE_F32 );
    depformer_input = ggml_add( ctx, depformer_input, last_token_input );

    auto dep_output = moshi_streaming_transformer( ctx,
        lm->depformer, states->depformer, depformer_input );

    auto logits = torch_nn_linear( ctx, lm->linears[depformer_cb_index], dep_output );

    return logits;
}


//#define ENABLE_SINGLE_GRAPH

void moshi_lmmodel_depformer_step(
        ScratchContext & ctx,
        moshi_lmmodel_t * lm,
        moshi_lmmodel_states_t * state,
        int text_token,
        bool use_sampling,
        float temp,
        int top_k,
        std::vector<int> & depformer_tokens
    ) {
    depformer_tokens.resize( lm->dep_q );

    init( &ctx, state->depformer, lm->depformer, NULL );

    auto last_token_input = lm->demux_second_stream?
        moshi_scaled_embedding_demux( ctx, lm->depformer_text_emb_demux, text_token ) :
        moshi_scaled_embedding( ctx, lm->depformer_text_emb, text_token );

    auto logits = moshi_lmmodel_forward_depformer_transform( ctx, lm, state,
        0, last_token_input, state->transformer_out );

    ON_NTH( 8, ctx.set_name( "depformer_32" ) );
#ifdef ENABLE_SINGLE_GRAPH
    auto next_token = moshi_sample_token( ctx, logits, use_sampling, temp, top_k );
    ctx.build_forward_expand( next_token, &depformer_tokens[0] );
#else
    //ctx.compute(); this will be done by sampler
    auto next_token = moshi_sample_token_int( ctx, logits, use_sampling, temp, top_k );
    depformer_tokens[0] = next_token;
#endif

    for (int cb_index = 1; cb_index < lm->dep_q; cb_index++) {
        auto index = cb_index - 1;

#ifdef ENABLE_SINGLE_GRAPH
        last_token_input = moshi_scaled_embedding_chained( ctx,
            lm->depformer_emb[index], next_token );
#else
        assert( next_token >= 0 ); // hypothetically sampling should pass this
        last_token_input = moshi_scaled_embedding( ctx,
            lm->depformer_emb[index], next_token );
#endif

        logits = moshi_lmmodel_forward_depformer_transform(
            ctx, lm, state, cb_index,
            last_token_input,
            state->transformer_out );

#ifdef ENABLE_SINGLE_GRAPH
        next_token = moshi_sample_token( ctx, logits, use_sampling, temp, top_k );
        ctx.build_forward_expand( next_token, &depformer_tokens[cb_index] );
#else
        //ctx.compute(); this will be done by the sampler
        next_token = moshi_sample_token_int( ctx, logits, use_sampling, temp, top_k );
        depformer_tokens[cb_index] = next_token;
#endif
    }
#ifdef ENABLE_SINGLE_GRAPH
    ctx.compute();
#endif
}

ggml_tensor * moshi_lmmodel_text_token_embed_build(
        GraphContext & ctx,
        moshi_lmmodel_t * lm,
        lmmodel_embed_t * embed,
        ggml_tensor * sum_condition
    ) {

    ggml_tensor * input;
    if ( lm->demux_second_stream ) {
        input = moshi_scaled_embedding_demux_build( ctx,
            lm->text_emb_demux, &embed->text_demux );
    } else {
        input = moshi_scaled_embedding_build( ctx,
            lm->text_emb, &embed->text );
    }

    embed->audio.resize( lm->num_audio_codebooks );
    for (int cb_index = 0; cb_index < lm->num_audio_codebooks; cb_index++) {
        auto audio_emb = moshi_scaled_embedding_build( ctx,
            lm->emb[cb_index], &embed->audio[cb_index] );

        input = ggml_add( ctx, input, audio_emb );
    }

    if (sum_condition) {
        input = ggml_add( ctx, sum_condition, input );
    }

    return input;
}

void moshi_lmmodel_text_token_embed_step(
        GraphContext & ctx,
        moshi_lmmodel_t * lm,
        lmmodel_embed_t * embed,
        std::vector<int> & sequence
    ) {

    if ( lm->demux_second_stream ) {
        moshi_scaled_embedding_demux_step( ctx, lm->text_emb_demux,
            &embed->text_demux, sequence[0] );
    } else {
        moshi_scaled_embedding_step( ctx, lm->text_emb, &embed->text,
            sequence[0] );
    }

    for (int cb_index = 0; cb_index < lm->num_audio_codebooks; cb_index++) {
        moshi_scaled_embedding_step( ctx,
            lm->emb[cb_index],
            &embed->audio[cb_index],
            sequence[cb_index + lm->audio_offset] );
    }
}

ggml_tensor * moshi_lmmodel_text_token_embed(
        ScratchContext & ctx,
        moshi_lmmodel_t * lm,
        std::vector<int> & sequence,
        ggml_tensor * sum_condition
    ) {
    //ProfileScope profile(time_text_emb_us);

    auto input = lm->demux_second_stream?
        moshi_scaled_embedding_demux( ctx, lm->text_emb_demux, sequence[0] ) :
        moshi_scaled_embedding( ctx, lm->text_emb, sequence[0] );

    for (int cb_index = 0; cb_index < lm->num_audio_codebooks; cb_index++) {
        auto audio_emb = moshi_scaled_embedding( ctx, lm->emb[cb_index],
            sequence[cb_index + lm->audio_offset] );

        input = ggml_add( ctx, input, audio_emb );
    }

    if (sum_condition) {
        input = ggml_add( ctx, sum_condition, input );
    }

    return input;
}

// moshi.models.lm.LMModel.forward_text
std::tuple<ggml_tensor*, ggml_tensor*> moshi_lmmodel_forward_text(
        ScratchContext & ctx,
        moshi_lmmodel_t * lm,
        moshi_lmmodel_states_t * state,
        std::vector<int> & sequence,
        ggml_tensor * sum_condition
    ) {
    //ProfileScope profile(time_forward_text_us);
    //assert len(sequence) == lm.num_codebooks

    auto input = moshi_lmmodel_text_token_embed( ctx, lm, sequence, sum_condition );

    auto transformer_out = moshi_streaming_transformer_graph( ctx,
        lm->transformer, state->transformer, input );

    if ( lm->out_norm )
        transformer_out = moshi_rms_norm( ctx, lm->out_norm, transformer_out );

    auto text_logits = torch_nn_linear( ctx, lm->text_linear, transformer_out );

    return { transformer_out, text_logits };
}

std::tuple<ggml_tensor*, ggml_tensor*> moshi_lmmodel_forward_text_build(
        GraphContext & ctx,
        moshi_lmmodel_t * lm,
        moshi_lmmodel_states_t * state,
        ggml_tensor * sum_condition
    ) {
    auto input = moshi_lmmodel_text_token_embed_build( ctx, lm, &state->embed, sum_condition );

    state->transformer_T = input->ne[1];
    auto transformer_out = moshi_streaming_transformer_graph_build( ctx,
        lm->transformer, state->transformer, input );

    if ( lm->out_norm )
        transformer_out = moshi_rms_norm( ctx, lm->out_norm, transformer_out );

    auto text_logits = torch_nn_linear( ctx, lm->text_linear, transformer_out );

    return { transformer_out, text_logits };
}

void moshi_lmmodel_forward_text_step(
        GraphContext & gctx,
        ScratchContext & ctx,
        moshi_lmmodel_t * lm,
        moshi_lmmodel_states_t * state,
        std::vector<int> & sequence
    ) {
    moshi_lmmodel_text_token_embed_step( gctx, lm, &state->embed, sequence );

    moshi_streaming_transformer_graph_step( ctx,
        lm->transformer, state->transformer, state->transformer_T );
}


// moshi.models.lm.LMGen

const int lm_ungenerated_token_id = -2;

struct moshi_lmgen_state_t {
    int offset;
    int skip;
    std::vector<std::vector<int>> cache;
    std::vector<int> initial;
};

moshi_lmgen_state_t * moshi_lmgen_state( moshi_lmmodel_t * lm ) {
    auto state = new moshi_lmgen_state_t {
        0, // offset
        0, // skip
    };
    const int cache_capacity = lm->max_delay + 2;
    state->cache.resize( cache_capacity );
    for (int c = 0; c < cache_capacity; c++) {
        auto & cache = state->cache[c];
        cache.resize( lm->num_codebooks );
        for (int k = 0; k < lm->num_codebooks; k++) {
            cache[k] = lm_ungenerated_token_id;
        }
    }
    state->initial.resize( lm->num_codebooks );
    state->initial[0] = lm->text_initial_token_id;
    for ( int i = 1; i < lm->num_codebooks; i++ )
        state->initial[i] = lm->initial_token_id;
    return state;
}

struct moshi_lmgen_t {
    moshi_lmmodel_t * lm;
    bool use_sampling;
    float temp;
    float temp_text;
    int top_k;
    int top_k_text;

    // these are from the TTSModel callback on_text
    StateMachine * machine;
    State * machine_state;

    // these are from LMGenState
    ggml_tensor * condition_sum;

    // these are from the TTSModel callbacks on_text, on_audio
    std::deque<int> * text_prefixes;
    std::deque<std::vector<int>> * audio_prefixes;
};

bool moshi_lmgen_step(
        ScratchContext & scratch,
        moshi_lmgen_t * lmgen,
        moshi_lmgen_state_t * state,
        moshi_lmmodel_states_t * lm_states,
        bool depformer_replace_tokens,
        int & int_text_token,
        std::vector<int> & int_audio_tokens,
        float * vad = NULL,
        int skip_prefix = 2 // for debugging set to 0
) {
    auto lm = lmgen->lm;
    auto use_sampling = lmgen->use_sampling;
    auto temp = lmgen->temp;
    auto temp_text = lmgen->temp_text;
    auto top_k = lmgen->top_k;
    auto top_k_text = lmgen->top_k_text;
    auto machine = lmgen->machine;
    auto machine_state = lmgen->machine_state;
    auto condition_sum = lmgen->condition_sum;
    auto text_prefixes = lmgen->text_prefixes;
    auto audio_prefixes = lmgen->audio_prefixes;
    //ProfileScope profile(time_lmgen_step_us);
    int CT = (int) state->cache.size();
    int dep_q_1 = lm->dep_q + 1;

    auto needed_tokens = lm->num_codebooks - lm->dep_q - 1;
    if ( needed_tokens > 0 ) {
        assert( (int)int_audio_tokens.size() >= needed_tokens );
        assert( (int)lm->delays.size() >= needed_tokens );
        int start = dep_q_1;
        for ( int i = 0; i < needed_tokens; i++ ) {
            int write_position = (state->offset + lm->delays[start + i]) % CT;
            state->cache[write_position][start + i] = int_audio_tokens[i];
        }
    }
    /*
    it would possibly make sense to "warm up" the cache to avoid the branching
    logic below, and maybe have a more complete graph. may not be a performance
    benefit to it though.
    UPDATE: after investigating, there is additional logic later on that looks
    for -1 and zeroes out the results if present. that means to do that you
    need to modify data. see: scaled_embedding functions
    */
    int positions = state->offset % CT;
    std::vector<int> input( lm->num_codebooks );
    for ( int i = 0; i < lm->num_codebooks; i++ ) {
        auto is_init = state->offset <= lm->delays[i];
        if (is_init)
            input[i] = state->initial[i];
        else
            input[i] = state->cache[positions][i];
    }

    ONCE( scratch.set_name("text") );
    ON_NTH( 32, scratch.set_name( "text_32" ) );

#ifdef USE_SCRATCH
    auto [scratch_transformer_out, text_logits] = moshi_lmmodel_forward_text(
        scratch, lm, lm_states,
        input, condition_sum );

    auto cpy_transformer_out = ggml_cpy( scratch,
        scratch_transformer_out, lm_states->transformer_out );
    scratch.build_forward_expand( cpy_transformer_out );

    // note this does the compute
    auto text_token = moshi_sample_token_int( scratch, text_logits,
        use_sampling, temp_text, top_k_text );
#else
    if ( ! lm_states->gctx ) {
        lm_states->gctx = new GraphContext( 256, scratch.backend );
        GraphContext &graph = *lm_states->gctx;

        auto [graph_transformer_out, text_logits] = moshi_lmmodel_forward_text_build(
            graph, lm, lm_states, condition_sum );

        auto cpy_transformer_out = ggml_cpy( graph,
            graph_transformer_out, lm_states->transformer_out );
        graph.build_forward_expand( cpy_transformer_out );

        lm_states->sampler_out = moshi_sample_token( graph, text_logits,
            use_sampling, temp_text, top_k_text );

        graph.build_forward_expand( lm_states->sampler_out );
        graph.alloc();
    }

    GraphContext &graph = *lm_states->gctx;
    moshi_lmmodel_forward_text_step( graph, scratch, lm, lm_states, input );

    scratch.compute();
    graph.compute();

    int text_token;
    ggml_backend_tensor_get( lm_states->sampler_out, &text_token, 0, 4 );
#endif

    // on_text_hook
    if ( machine ) {
        if ( text_prefixes && text_prefixes->size() ) {
            text_token = text_prefixes->front();
            text_prefixes->pop_front();
        } else {
            static int prev_in_token = 3;
            static int prev_text_token = 3;
            auto in_token = text_token;

            text_token = machine->process(state->offset, machine_state, text_token);

            if (machine->logging && ( prev_in_token != in_token || prev_text_token != text_token ) ) {
                printf( "%d {%d, %d}\n", in_token, text_token % 8001, text_token / 8001 - 1 );
                prev_in_token = in_token;
                prev_text_token = text_token;
            }
        }
    }

    int_audio_tokens.resize( lm->dep_q );
    if ( lm->depformer ) {
        if (!depformer_replace_tokens) {
            //ProfileScope profile(time_depformer_step_us);
            moshi_lmmodel_depformer_step(
                scratch, lm, lm_states,
                text_token, use_sampling, temp, top_k,
                int_audio_tokens );
        } else {
            for (int i = 0; i < (int)int_audio_tokens.size(); i++) {
                int_audio_tokens[i] = -1;
            }
        }
        // on_audio_hook
        const int delay_steps = lm->delay_steps;
        if ( delay_steps ) {
            for (int q = 0; q < (int)int_audio_tokens.size(); q++) {
                if (state->offset < lm->delays[q + 1] + delay_steps)
                    int_audio_tokens[q] = -1; // token_ids.zero
            }
        }
        if ( audio_prefixes && audio_prefixes->size() ) {
            state->skip = skip_prefix;
            auto audio_codes = audio_prefixes->front();
            for (int q = 0; q < (int)int_audio_tokens.size(); q++) {
                if (audio_codes[q] != lm_ungenerated_token_id)
                    int_audio_tokens[q] = audio_codes[q];
            }
            audio_prefixes->pop_front();
        }
    }

    state->offset++;

    int position = state->offset % CT;
    state->cache[position][0] = text_token;
    if ( lm->depformer ) {
        for (int q = 0; q < (int)int_audio_tokens.size(); q++) {
            state->cache[position][q + 1] = int_audio_tokens[q];
        }
    }
    
    if ( state->skip > 0 ) {
        --state->skip;
        return false;
    }

    if (state->offset <= lm->max_delay || depformer_replace_tokens)
        return false;

    //int_audio_tokens[0] = state->cache[(state->offset - lm->max_delay) % state->cache.size()][1];
    int index = (state->offset - lm->max_delay + lm->delays[0]) % CT;
    int_text_token = state->cache[index][0];
    for ( int i = 1; i < dep_q_1; i++ ) {
        index = (state->offset - lm->max_delay + lm->delays[i]) % CT;
        int_audio_tokens[i - 1] = state->cache[index][i];
    }

    for (auto x : int_audio_tokens) {
        if (x == -1)
            return false;
    }
    
    if ( vad ) {
        if ( lm->extra_heads.size() > 2 ) {
            auto linear = torch_nn_linear( scratch, lm->extra_heads[2], lm_states->transformer_out );
            auto soft_max = ggml_soft_max( scratch, linear );
            auto view = ggml_view_1d( scratch, soft_max, 1, 0 );
            scratch.build_forward_expand( view, vad );
            scratch.compute();
        } else {
            *vad = 0;
        }
    }

    return true;
}


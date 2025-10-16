#pragma once

// TODO: remove prefix
std::tuple<std::string, std::string> split_first( const std::string& input, char c ) {
    size_t pos = input.find(c);
    if (pos == std::string::npos)
        return {input, ""};
    return {input.substr(0, pos), input.substr(pos + 1)};
}

class WeightLoader {
    public:
    struct binding_t {
        ggml_tensor ** target;
        std::string name;
    };
    typedef std::vector<binding_t> bindings_t;

    struct alloc_request_t {
        ggml_tensor ** result;
        int n_dims;
        NE ne;
        ggml_type type;
        std::string name;
    };

    SafeTensorFile * stf;
    ScratchContext * scratch;
    ggml_backend * backend;
    ggml_context * ctx;
    ggml_backend_buffer_t buffer;
    std::vector<alloc_request_t> alloc_requests;
    std::vector<std::function<void(WeightLoader*)>> init_requests;

    WeightLoader(SafeTensorFile * stf, ScratchContext * scratch, ggml_backend * backend = NULL) {
        assert( !scratch->backend ); // only cpu backends supported
        this->stf = stf;
        this->scratch = scratch;
        this->backend = backend;
        ctx = NULL;
        buffer = NULL;
    }

    ~WeightLoader() {
        if (buffer)
            ggml_backend_buffer_free( buffer );
        if (ctx)
            ggml_free( ctx );
    }

    safetensor_t * find( std::string name ) {
        // TODO: remove the the prefix
        auto [_, _name] = split_first(name, '.');
        return stf->find( _name );
    }

    void init( safetensor_t * safetensor, ggml_tensor * tensor ) {
        stf->init( safetensor, tensor, backend );
    }

    void add_alloc( ggml_tensor ** result, int n_dims, NE ne, ggml_type type, std::string name ) {
        assert( ctx == NULL );
        alloc_requests.push_back({ result, n_dims, {ne[0], ne[1], ne[2], ne[3]}, type, name });
    }

    void add_init( std::function<void(WeightLoader*)> on_init ) {
        init_requests.push_back( on_init );
    }

    bool fetch( bindings_t &bindings, std::function<bool(WeightLoader*)> on_bind ) {
        return on_bind( this );
    }

    bool fetch( ggml_tensor ** result, std::string name, void *func = NULL, int offset = 0 ) {
        safetensor_t * safetensor =  find( name );
        *result = NULL;
        if (!safetensor)
            return false;
        ggml_type src_type = safetensor_get_type( safetensor->dtype );
        NE ne;
        int n_dims = safetensor_get_shape(safetensor, ne, offset);
        // get destination type
        ggml_type dst_type = src_type;
        if (func == ggml_mul) dst_type = GGML_TYPE_F32;
        else if (func == ggml_add) dst_type = GGML_TYPE_F32;
        else if (func == ggml_rms_norm) dst_type = GGML_TYPE_F32;
        else if (func == ggml_conv_1d) dst_type = GGML_TYPE_F16;
        add_alloc( result, n_dims, ne, dst_type, name );
        if (dst_type == src_type) {
            add_init([ safetensor, result ]( WeightLoader * loader ) {
                loader->init( safetensor, *result );
            } );
        } else {
            add_init([ safetensor, result ]( WeightLoader * loader ) {
                auto & scratch_ctx = *loader->scratch;
                auto original = scratch_ctx.load( loader->stf, safetensor );
                auto cast = ggml_cast( scratch_ctx, original, (*result)->type );
                scratch_ctx.build_forward_expand( cast, *result );
                scratch_ctx.compute();
            } );
        }
        return true;
    }

    void alloc() {
        assert( ctx == NULL );
        size_t nbytes = ggml_tensor_overhead() * alloc_requests.size();
        if (backend) {
            ctx = ggml_init({ nbytes, NULL, true });
        } else {
            for (auto req : alloc_requests) {
                int64_t ne = req.ne[0] * req.ne[1] * req.ne[2] * req.ne[3];
                nbytes += ggml_row_size(req.type, ne);
            }
            ctx = ggml_init({ nbytes, NULL, false });
        }
        for (auto req : alloc_requests) {
            *req.result = ggml_new_tensor( ctx, req.type, req.n_dims, req.ne );
            if (req.name.size())
                ggml_set_name( *req.result, req.name.c_str() );
        }
        alloc_requests.clear();
        if (backend)
            buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    }

    void init() {
        assert( ctx );
        assert( backend == NULL || buffer != NULL);
        for (auto req : init_requests) {
            req( this );
        }
        init_requests.clear();
    }

    void load() {
        alloc();
        init();
    }
};



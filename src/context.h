#pragma once

typedef int64_t NE[GGML_MAX_DIMS]; // number of elements per dimension
class GGML_NE {
public:
    int64_t ne[GGML_MAX_DIMS];
    GGML_NE( int64_t ne0=1, int64_t ne1=1, int64_t ne2=1, int64_t ne3=1 ) {
        ne[0] = ne0;
        ne[1] = ne1;
        ne[2] = ne2;
        ne[3] = ne3;
    }

    operator int64_t* () {
        return ne;
    }
};

ggml_type safetensor_get_type(std::string dtype) {
    if (dtype == "F32")
        return GGML_TYPE_F32;
    if (dtype == "F16")
        return GGML_TYPE_F16;
    if (dtype == "BF16")
        return GGML_TYPE_BF16;
    assert(false);
    return (ggml_type)-1;
}

int safetensor_get_shape(safetensor_t * safetensor, NE &ne, int offset = 0) {
    // dimensions are inverted
    int last_index = safetensor->shape.size() - 1;
    assert( last_index + offset < 4 );
    for (int i = 0; i < offset; i++)
        ne[i] = 1;
    for (int i = 0; i <= last_index; i++)
        ne[i + offset] = safetensor->shape[last_index-i];
    for (int i = offset + last_index + 1; i < 4; i++) {
        ne[i] = 1;
    }
    return safetensor->shape.size() + offset;
}

ggml_tensor * safetensor_alloc( ggml_context * ctx, safetensor_t * safetensor) {
    auto type = safetensor_get_type(safetensor->dtype);
    // dimensions are inverted
    int last_index = safetensor->shape.size() - 1;
    NE ne = {1, 1, 1, 1};
    for (int i = 0; i <= last_index; i++)
        ne[i] = safetensor->shape[last_index-i];
    return ggml_new_tensor_4d(ctx, type, ne[0], ne[1], ne[2], ne[3]);
}

class SafeTensorFile {
    public:
    FILE * f;
    int64_t header_length;
    safetensors_t tensors;
    SafeTensorFile() {}
    ~SafeTensorFile() {fclose(f);}

    static SafeTensorFile * from_file(const char * filename) {
        FILE * f = fopen(filename, "rb");
        if (!f)
            return NULL;
        int64_t length;
        size_t r;
        r = fread(&length, sizeof(length), 1, f);
        if (r != 1 || length == 0) {
            fclose(f);
            return NULL;
        }
        std::vector<char> data(length+1);
        r = fread(data.data(), length, 1, f);
        if (r != 1) {
            fclose(f);
            return NULL;
        }
        data[length] = 0;

        const_str_t json = {data.data(), (int)length};

        safetensors_t tensors;
        if (!safetensor_parse(json, tensors)) {
            fclose(f);
            return NULL;
        }

        auto stf = new SafeTensorFile();
        stf->f = f;
        stf->header_length = length + 8;
        stf->tensors.swap(tensors);
        return stf;
    }

    safetensor_t * find(std::string name) {
        auto it = tensors.find(name);
        if (it == tensors.end())
            return NULL;
        return & it->second;
    }

    void init( safetensor_t * safetensor, ggml_tensor * tensor, ggml_backend * backend = NULL ) {
        int64_t nbytes = ggml_nbytes(tensor);
        int64_t offset = safetensor->data_offsets[0] + header_length;
        int64_t size = safetensor->data_offsets[1] - safetensor->data_offsets[0];
        if (nbytes > size) {
            printf("data is smaller than expected, got %ld needed %ld\n", size, nbytes);
            exit(-1);
        }
        fseek(f, offset, SEEK_SET);
        if (backend) {
            std::vector<char*> data(nbytes);
            int64_t r = fread(data.data(), nbytes, 1, f);
            if (r != 1) {
                printf("failed to read tensor %s\n", safetensor->key.c_str());
                exit(-1);
            }
            ggml_backend_tensor_set(tensor, data.data(), 0, nbytes);
        } else {
            int64_t r = fread(tensor->data, nbytes, 1, f);
            if (r != 1) {
                printf("failed to read tensor %s\n", safetensor->key.c_str());
                exit(-1);
            }
        }
    }
};

class ScratchContext {
    public:
    ggml_backend * backend;
    ggml_context * ctx;
    ggml_cgraph * gf;

    // to load tensors
    struct load_t {
        SafeTensorFile * src;
        safetensor_t * safetensor;
        ggml_tensor * tensor;
    };
    std::vector<load_t> loaders;
    // to load 32 constant
    struct constant_32_t {
        ggml_tensor * tensor;
        int32_t value;
    };
    std::vector<constant_32_t> constants32;
    // to load a vector constant
    struct constant_t {
        ggml_tensor * tensor;
        std::vector<uint8_t> data;
    };
    std::vector<constant_t> constants;
    // input convert
    struct input_convert_t {
        ggml_tensor * dst;
        ggml_tensor * src;
    };
    std::vector<input_convert_t> input_converts;
    // to manually copy tensors, from backend to local context
    struct copy_tensor_t {
        ggml_tensor * src;
        ggml_context * ctx;
        ggml_tensor ** dst;
    };
    std::vector<copy_tensor_t> tensor_copies;
    //
    struct backend_tensor_t {
        ggml_tensor * src;
        ggml_tensor * dst;
    };
    std::vector<backend_tensor_t> backend_copies;
    //
    struct copy_t {
        ggml_tensor * src;
        void * dst;
    };
    std::vector<copy_t> copies;

    ScratchContext( size_t mb, ggml_backend * backend = NULL ) {
        ctx = ggml_init({
            /*.mem_size   =*/ mb * 1024 * 1024,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ backend? true : false, // NOTE: this should be false when using the legacy API
        });
        gf = NULL;
        this->backend = backend;
    }
    ~ScratchContext() { ggml_free(ctx); }

    bool can_cast() {
        return true;
    }

    operator ggml_context * () {
        return ctx;
    }

    ggml_tensor * load( SafeTensorFile * src, safetensor_t * safetensor ) {
        auto tensor = safetensor_alloc( ctx, safetensor );
        if (backend) {
            loaders.push_back({ src, safetensor, tensor });
            return tensor;
        }
        src->init(safetensor, tensor);
        return tensor;
    }

    ggml_tensor * constant( int32_t i32 ) {
        if (backend) {
            auto tensor = ggml_new_tensor_1d( ctx, GGML_TYPE_I32, 1 );
            constants32.push_back({ tensor, i32 });
            return tensor;
        }
        return ggml_new_i32( ctx, i32 );
    }

    ggml_tensor * constant( float f32 ) {
        if (backend) {
            auto tensor = ggml_new_tensor_1d( ctx, GGML_TYPE_F32, 1 );
            constants32.push_back({ tensor, *(int32_t*)&f32 });
            return tensor;
        }
        return ggml_new_f32( ctx, f32 );
    }

    ggml_tensor * fill( int count, float value ) {
        if (backend) {
            auto tensor = ggml_new_tensor_1d( ctx, GGML_TYPE_F32, count );
            constants.push_back({tensor});
            auto & constant = constants.back();
            constant.data.resize( ggml_nbytes( tensor ) );
            float * data = (float*)constant.data.data();
            for (int64_t i = 0; i < count; i++) {
                data[i] = value;
            }
            return tensor;
        }
        assert(false); // TODO
    }

    // allow another to write to the backend of ours
    // primarily for copying from scratch cpu to scratch gpu
    ggml_tensor * dup_constant( ggml_tensor * src_tensor, void * &data ) {
        assert( backend ); // TODO maybe support non-backend
        auto tensor = ggml_dup_tensor( ctx, src_tensor );
        constants.push_back({tensor});
        auto & constant = constants.back();
        constant.data.resize( ggml_nbytes( tensor ) );
        data = constant.data.data();
        return tensor;
    }

    ggml_tensor * arange( float start, float stop, float step ) {
        if (backend) {
            const int64_t steps = (int64_t) ceilf((stop - start) / step);
            auto tensor = ggml_new_tensor_1d( ctx, GGML_TYPE_F32, steps );
            constants.push_back({tensor});
            auto & constant = constants.back();
            constant.data.resize( ggml_nbytes( tensor ) );
            float * data = (float*)constant.data.data();
            for (int64_t i = 0; i < steps; i++) {
                data[i] = start + step * i;
            }
            return tensor;
        }
        return ggml_arange( ctx, start, stop, step );
    }

    ggml_tensor * exponential( NE ne, float lambd = 1.f ) {
        auto tensor = ggml_new_tensor( ctx, GGML_TYPE_F32, 4, ne );
        int n = ggml_nelements( tensor );
        float * data;
        if (backend) {
            constants.push_back({tensor});
            auto & constant = constants.back();
            constant.data.resize( ggml_nbytes( tensor ) );
            data = (float*)constant.data.data();
        } else {
            data = (float*)tensor->data;
        }
#ifdef DISABLE_RAND
        for (int i = 0; i < n; i++)
            data[i] = -logf(0.5) / lambd;
#else
        for (int i = 0; i < n; i++)
            data[i] = -logf(rand() / (float)RAND_MAX) / lambd;
#endif
        return tensor;
    }

    ggml_tensor * input( NE ne, std::vector<int> & i32 ) {
        auto tensor = ggml_new_tensor( ctx, GGML_TYPE_I32, 4, ne );
        size_t nelements = ggml_nelements( tensor );
        assert( nelements == i32.size() );
        int * data;
        if (backend) {
            constants.push_back({tensor});
            auto & constant = constants.back();
            constant.data.resize( ggml_nbytes( tensor ) );
            data = (int*)constant.data.data();
        } else {
            data = (int*)tensor->data;
        }
        memcpy( data, i32.data(), ggml_nbytes( tensor ) );
        return tensor;
    }

    ggml_tensor * input( ggml_tensor * tensor ) {
        if (tensor == NULL)
            return NULL;
        if (backend) {
            if (!tensor->buffer) {
                assert( tensor->data );
                assert( ggml_is_contiguous( tensor ) );
                input_convert_t convert;
                convert.src = tensor;
                convert.dst = ggml_dup_tensor( ctx, tensor );
                input_converts.push_back( convert );
                tensor = convert.dst;
            }
            return tensor;
        }
        assert( tensor->data );
        return tensor;
    }

    std::string name;
    void set_name(std::string name) {
        this->name = name;
    }

    void build_forward_expand( ggml_tensor * tensor ) {
        assert( tensor->op == GGML_OP_CPY ); // scratch context will not store data
        if (!gf)
            gf = ggml_new_graph_custom( ctx, GGML_DEFAULT_GRAPH_SIZE * 2, false );
        ggml_build_forward_expand( gf, tensor );
    }

    void build_forward_expand( ggml_tensor * tensor,
            ggml_context * copy_ctx, ggml_tensor ** copy_tensor ) {
        assert( !ggml_get_no_alloc( copy_ctx ) );
        if (backend) {
            if (!gf)
                gf = ggml_new_graph_custom( ctx, GGML_DEFAULT_GRAPH_SIZE * 2, false );
            ggml_build_forward_expand( gf, tensor );
            tensor_copies.push_back({ tensor, copy_ctx, copy_tensor });
            return;
        }
        // wrap in a copy op
        *copy_tensor = ggml_dup_tensor( copy_ctx, tensor );
        tensor = ggml_cpy( ctx, tensor, *copy_tensor );
        if (!gf)
            gf = ggml_new_graph_custom( ctx, GGML_DEFAULT_GRAPH_SIZE * 2, false );
        ggml_build_forward_expand( gf, tensor );
    }

    void build_forward_expand( ggml_tensor * tensor, ggml_tensor * copy_tensor ) {
        assert( copy_tensor->buffer ); // copy to a backend
        assert( ggml_nbytes(tensor) == ggml_nbytes(copy_tensor) );
        if (!gf)
            gf = ggml_new_graph_custom( ctx, GGML_DEFAULT_GRAPH_SIZE * 2, false );
        ggml_build_forward_expand( gf, tensor );
        backend_copies.push_back({ tensor, copy_tensor });
    }

    void build_forward_expand( ggml_tensor * tensor, void * dst ) {
        if (!gf)
            gf = ggml_new_graph_custom( ctx, GGML_DEFAULT_GRAPH_SIZE * 2, false );
        ggml_build_forward_expand( gf, tensor );
        copies.push_back({ tensor, dst });
    }

    struct debug_sum_t {
        const char * label;
        ggml_tensor * src;
    };
    std::vector<debug_sum_t> debug_sums;
    void debug( const char * label, ggml_tensor * src ) {
        if (src->type != GGML_TYPE_F32)
            src = ggml_cast( ctx, src, GGML_TYPE_F32 );
        auto sum = ggml_sum( ctx, src );
        if (!gf)
            gf = ggml_new_graph_custom( ctx, GGML_DEFAULT_GRAPH_SIZE * 2, false );
        ggml_build_forward_expand( gf, sum );
        debug_sums.push_back({label, sum});
    }

    void reset() {
        backend_copies.clear();
        copies.clear();
        tensor_copies.clear();
        input_converts.clear();
        constants.clear();
        constants32.clear();
        loaders.clear();
        ggml_reset(ctx);
        gf = NULL;
        name = "";
    }

    void compute() {
        if (backend) {
            // initialize tensors
            auto buffer = ggml_backend_alloc_ctx_tensors( ctx, backend );
            for (auto load : loaders) {
                load.src->init( load.safetensor, load.tensor, backend );
            }
            for (auto i32 : constants32) {
                ggml_backend_tensor_set(i32.tensor, &i32.value, 0, ggml_nbytes(i32.tensor));
            }
            for (auto constant : constants) {
                ggml_backend_tensor_set(constant.tensor, constant.data.data(), 0, ggml_nbytes(constant.tensor));
            }
            for (auto convert : input_converts) {
                ggml_backend_tensor_set(convert.dst, convert.src->data, 0, ggml_nbytes(convert.dst));
            }
            // compute
            if (name.size()) {CAPTURE(name, gf);}
            ggml_backend_graph_compute( backend, gf );
            // debug
            for (auto sum : debug_sums) {
                float fsum;
                ggml_backend_tensor_get( sum.src, &fsum, 0, 4 );
                printf( "%s %f\n", sum.label, fsum );
            }
            debug_sums.clear();
            // copy results
            for (auto copy : tensor_copies) {
                auto tensor = ggml_dup_tensor( copy.ctx, copy.src );
                ggml_backend_tensor_get(copy.src, tensor->data, 0, ggml_nbytes(tensor));
                *copy.dst = tensor;
            }
            for (auto copy : copies) {
                size_t nbytes = ggml_nbytes(copy.src);
                ggml_backend_tensor_get(copy.src, copy.dst, 0, nbytes);
            }
            for (auto copy : backend_copies) {
                int nbytes = ggml_nbytes( copy.dst );
                std::vector<uint8_t> buf( nbytes );
                ggml_backend_tensor_get( copy.src, buf.data(), 0, nbytes );
                ggml_backend_tensor_set( copy.dst, buf.data(), 0, nbytes );
            }
            // cleanup
            ggml_backend_buffer_free( buffer );
            backend_copies.clear();
            copies.clear();
            tensor_copies.clear();
            input_converts.clear();
            constants.clear();
            constants32.clear();
            loaders.clear();
        } else {
            if (name.size()) {CAPTURE(name, gf);}
            ggml_graph_compute_with_ctx(ctx, gf, 1);
            for (auto copy : copies) {
                size_t nbytes = ggml_nbytes(copy.src);
                memcpy(copy.dst, copy.src->data, nbytes);
            }
            for (auto copy : backend_copies) {
                int nbytes = ggml_nbytes( copy.dst );
                std::vector<uint8_t> buf( nbytes );
                ggml_backend_tensor_set( copy.dst, copy.src->data, 0, nbytes );
            }
            backend_copies.clear();
            copies.clear();
        }
        ggml_reset(ctx);
        gf = NULL;
        name = "";
    }
};

class StateContext {
    public:
    ggml_backend * backend;
    ggml_context * ctx;
    ggml_backend_buffer_t buffer;

    struct state_tensor_t {
        ggml_tensor ** ptensor;
        ggml_type type;
        NE ne;
        std::vector<uint8_t> data;
    };
    std::vector<state_tensor_t> states;

    StateContext( ggml_backend * backend = NULL ) {
        ctx = NULL;
        buffer = NULL;
        this->backend = backend;
    }

    ~StateContext() {
        if (buffer)
            ggml_backend_buffer_free( buffer );
        if (ctx)
            ggml_free( ctx );
    }

    void fill32( NE ne, ggml_type type, int32_t value, ggml_tensor ** ptensor ) {
        assert( type == GGML_TYPE_F32 || type == GGML_TYPE_I32 );
        states.push_back({ ptensor, type });
        auto & state = states.back();
        int64_t nelements = 1;
        for ( int i = 0; i < GGML_MAX_DIMS; i++ ) {
            state.ne[i] = ne[i];
            nelements *= ne[i];
        }
        state.data.resize( nelements * 4 );
        int32_t * data = (int32_t*)state.data.data();
        for ( int i = 0; i < nelements; i++)
            data[i] = value;
        *ptensor = NULL;
    }

    void fill( NE ne, float value, ggml_tensor ** ptensor ) {
        fill32( ne, GGML_TYPE_F32, *(int32_t*)&value, ptensor );
    }

    void fill( NE ne, int32_t value, ggml_tensor ** ptensor ) {
        fill32( ne, GGML_TYPE_I32, value, ptensor );
    }

    void alloc() {
        assert( ctx == NULL ); // can only alloc once!
        size_t nbytes = ggml_tensor_overhead() * states.size();
        if (backend) {
            ctx = ggml_init({ nbytes, NULL, true });
        } else {
            for ( auto state : states )
                nbytes += state.data.size();
            ctx = ggml_init({ nbytes, NULL, false });
        }
        for ( auto state : states )
            *state.ptensor = ggml_new_tensor( ctx, state.type, 4, state.ne );
        if (backend)
            buffer = ggml_backend_alloc_ctx_tensors( ctx, backend );
    }

    void init() {
        if (backend) {
            for ( auto state : states )
                ggml_backend_tensor_set( *state.ptensor, state.data.data(), 0,
                    state.data.size() );
        } else {
            for ( auto state : states )
                memcpy( (*state.ptensor)->data, state.data.data(), state.data.size() );
        }
    }
};


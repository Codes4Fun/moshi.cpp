#pragma once

#include <inttypes.h>

#include <stdexcept>

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
    
    int64_t nelements() {
        return ne[0] * ne[1] * ne[2] * ne[3];
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
    int last_index = (int) safetensor->shape.size() - 1;
    assert( last_index + offset < 4 );
    for (int i = 0; i < offset; i++)
        ne[i] = 1;
    for (int i = 0; i <= last_index; i++)
        ne[i + offset] = safetensor->shape[last_index-i];
    for (int i = offset + last_index + 1; i < 4; i++) {
        ne[i] = 1;
    }
    return (int) safetensor->shape.size() + offset;
}

ggml_tensor * safetensor_alloc( ggml_context * ctx, safetensor_t * safetensor) {
    auto type = safetensor_get_type(safetensor->dtype);
    // dimensions are inverted
    int last_index = (int) safetensor->shape.size() - 1;
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

    int ref_count = 1;
    void ref() {
        ref_count++;
    }
    void unref() {
        --ref_count;
        if (ref_count < 1)
            delete this;
    }

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
            printf("data is smaller than expected, got %" PRId64 " needed %" PRId64 "\n", size, nbytes);
            exit(-1);
        }
#ifdef _WIN32
        //fseeko64(f, offset, SEEK_SET);
        auto e = _fseeki64(f, offset, SEEK_SET);
#else
        auto e = fseek(f, offset, SEEK_SET);
#endif
        assert( e == 0 );
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

SafeTensorFile * ref( SafeTensorFile * stf ) {
    stf->ref();
    return stf;
}

void unref( SafeTensorFile * stf ) {
    stf->unref();
}

// single tensor context
class own_ctx_tensor {
public:
    ggml_context * ctx;
    ggml_tensor * tensor;
    ggml_backend_buffer * buffer;
    own_ctx_tensor() {
        ctx = NULL;
        tensor = NULL;
        buffer = NULL;
    }
    void reset() {
        if ( buffer )
            ggml_backend_buffer_free( buffer );
        if ( ctx )
            ggml_free( ctx );
        buffer = NULL;
        ctx = NULL;
    }
    ~own_ctx_tensor() {
        reset();
    }
    void new_tensor( NE ne, ggml_type type, ggml_backend * backend ) {
        assert( backend ); // TODO: support non-backend options
        reset();
        if ( backend ) {
            ctx = ggml_init({
                /*.mem_size   =*/ ggml_tensor_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            });
            assert( ctx );
            tensor = ggml_new_tensor( ctx, type, 4, ne );
            assert( tensor );
            buffer = ggml_backend_alloc_ctx_tensors( ctx, backend );
            assert( buffer );
        }
    }
    operator ggml_tensor* () {
        return tensor;
    }
    ggml_tensor * operator->() {
        return tensor;
    }
    own_ctx_tensor( const own_ctx_tensor& ) = delete; 
    own_ctx_tensor & operator=( own_ctx_tensor& ) = delete;
};

class GraphContext {
    public:
    ggml_backend * backend;
    ggml_context * ctx;
    ggml_cgraph * gf;
    ggml_backend_buffer * buffer;

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

    GraphContext( size_t mb, ggml_backend * backend = NULL ) {
        this->backend = backend;
        ctx = ggml_init({
            /*.mem_size   =*/ mb * 1024 * 1024,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ backend? true : false, // NOTE: this should be false when using the legacy API
        });
        gf = NULL;
        buffer = NULL;
    }

    ~GraphContext() {
        if ( buffer )
            ggml_backend_buffer_free( buffer );
        ggml_free(ctx);
    }

    operator ggml_context * () {
        return ctx;
    }

    ggml_tensor * new_tensor( ggml_type type, NE ne ) {
        auto tensor = ggml_new_tensor( ctx, type, 4, ne );
        return tensor;
    }

    virtual void tensor_set( ggml_tensor * tensor, int32_t value ) {
        assert( buffer );
        assert( tensor->type == GGML_TYPE_I32 && ggml_nelements( tensor ) == 1 );
        ggml_backend_tensor_set( tensor, &value, 0, 4 );
    }

    virtual void tensor_set( ggml_tensor * tensor, float value ) {
        assert( buffer );
        assert( tensor->type == GGML_TYPE_F32 && ggml_nelements( tensor ) == 1 );
        ggml_backend_tensor_set( tensor, &value, 0, 4 );
    }

    virtual void tensor_set( ggml_tensor * tensor, std::vector<int32_t> & value ) {
        assert( buffer );
        assert( tensor->type == GGML_TYPE_I32 && ggml_nelements( tensor ) == value.size() );
        ggml_backend_tensor_set( tensor, value.data(), 0, 4 * value.size() );
    }

    virtual void tensor_set( ggml_tensor * tensor, std::vector<float> & value ) {
        assert( buffer );
        assert( tensor->type == GGML_TYPE_F32 && ggml_nelements( tensor ) == value.size() );
        ggml_backend_tensor_set( tensor, value.data(), 0, 4 * value.size() );
    }

    ggml_tensor * constant( int32_t i32 ) {
        if (backend) {
            auto tensor = ggml_new_tensor_1d( ctx, GGML_TYPE_I32, 1 );
            constants32.push_back({ tensor, i32 });
            return tensor;
        }
        assert(false);
        //return ggml_new_i32( ctx, i32 );
        return NULL;
    }

    ggml_tensor * constant( float f32 ) {
        if (backend) {
            auto tensor = ggml_new_tensor_1d( ctx, GGML_TYPE_F32, 1 );
            constants32.push_back({ tensor, *(int32_t*)&f32 });
            return tensor;
        }
        assert(false);
        //return ggml_new_f32( ctx, f32 );
        return NULL;
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
        assert(false);
    }

    ggml_tensor * fill( NE ne, float value ) {
        auto tensor = ggml_new_tensor( ctx, GGML_TYPE_F32, 4, ne );
        auto nelements = ggml_nelements( tensor );
        float * data;
        if (backend) {
            constants.push_back({tensor});
            auto & constant = constants.back();
            constant.data.resize( ggml_nbytes( tensor ) );
            data = (float*)constant.data.data();
        } else {
            assert( tensor->data );
            data = (float*)tensor->data;
        }
        for (int64_t i = 0; i < nelements; i++) {
            data[i] = value;
        }
        return tensor;
    }

    ggml_tensor * fill( NE ne, int32_t value ) {
        auto tensor = ggml_new_tensor( ctx, GGML_TYPE_I32, 4, ne );
        auto nelements = ggml_nelements( tensor );
        int32_t * data;
        if (backend) {
            constants.push_back({tensor});
            auto & constant = constants.back();
            constant.data.resize( ggml_nbytes( tensor ) );
            data = (int32_t*)constant.data.data();
        } else {
            assert( tensor->data );
            data = (int32_t*)tensor->data;
        }
        for (int64_t i = 0; i < nelements; i++) {
            data[i] = value;
        }
        return tensor;
    }

    // arange was not supported by all backends
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

    // probability density function
    ggml_tensor * exponential( NE ne, float lambd = 1.f ) {
        auto tensor = ggml_new_tensor( ctx, GGML_TYPE_F32, 4, ne );
        int64_t n = ggml_nelements( tensor );
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
        for (int64_t i = 0; i < n; i++)
            data[i] = -logf(0.5) / lambd;
#else
        for (int64_t i = 0; i < n; i++)
            data[i] = -logf(rand() / (float)RAND_MAX) / lambd;
#endif
        return tensor;
    }

    std::string name;
    void set_name(std::string name) {
        this->name = name;
    }

    ggml_cgraph * get_graph() {
        if (!gf)
            gf = ggml_new_graph_custom( ctx, GGML_DEFAULT_GRAPH_SIZE * 4, false );
        return gf;
    }

    void build_forward_expand( ggml_tensor * tensor ) {
        ggml_build_forward_expand( get_graph(), tensor );
    }

    void alloc() {
        assert( ! buffer );
        buffer = ggml_backend_alloc_ctx_tensors( ctx, backend );
        assert( buffer );
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
    }

    void compute() {
        assert( backend );
        if (name.size()) {CAPTURE(name, gf);}
        ggml_backend_graph_compute( backend, gf );
    }
};

class ScratchContext : public GraphContext {
    public:
    ScratchContext( size_t mb, ggml_backend * backend = NULL )
        : GraphContext( mb, backend ) {}

    ggml_tensor * load( SafeTensorFile * src, safetensor_t * safetensor ) {
        auto tensor = safetensor_alloc( ctx, safetensor );
        if (backend) {
            loaders.push_back({ src, safetensor, tensor });
            return tensor;
        }
        src->init(safetensor, tensor);
        return tensor;
    }

    virtual void tensor_set( ggml_tensor * tensor, int32_t value ) {
        assert( tensor->type == GGML_TYPE_I32 && ggml_nelements( tensor ) == 1 );
        constants32.push_back({ tensor, value });
    }

    virtual void tensor_set( ggml_tensor * tensor, float value ) {
        assert( tensor->type == GGML_TYPE_F32 && ggml_nelements( tensor ) == 1 );
            constants32.push_back({ tensor, *(int32_t*)&value });
    }

    virtual void tensor_set( ggml_tensor * tensor, std::vector<int32_t> & value ) {
        assert( tensor->type == GGML_TYPE_I32 && ggml_nelements( tensor ) == value.size() );
        constants.push_back({tensor});
        auto & constant = constants.back();
        constant.data.resize( ggml_nbytes( tensor ) );
        memcpy( constant.data.data(), value.data(), ggml_nbytes( tensor ) );
    }

    virtual void tensor_set( ggml_tensor * tensor, std::vector<float> & value ) {
        assert( tensor->type == GGML_TYPE_F32 && ggml_nelements( tensor ) == value.size() );
        constants.push_back({tensor});
        auto & constant = constants.back();
        constant.data.resize( ggml_nbytes( tensor ) );
        memcpy( constant.data.data(), value.data(), ggml_nbytes( tensor ) );
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

    ggml_tensor * input( NE ne, std::vector<float> & f32 ) {
        auto tensor = ggml_new_tensor( ctx, GGML_TYPE_F32, 4, ne );
        size_t nelements = ggml_nelements( tensor );
        assert( nelements == f32.size() );
        float * data;
        if (backend) {
            constants.push_back({tensor});
            auto & constant = constants.back();
            constant.data.resize( ggml_nbytes( tensor ) );
            data = (float*)constant.data.data();
        } else {
            data = (float*)tensor->data;
        }
        memcpy( data, f32.data(), ggml_nbytes( tensor ) );
        return tensor;
    }

    void build_forward_expand( ggml_tensor * tensor ) {
        assert( tensor->op == GGML_OP_CPY ); // scratch context will not store data
        ggml_build_forward_expand( get_graph(), tensor );
    }

    // this should only be used for cpu to gpu copies, otherwise use ggml_cpy
    void build_forward_expand( ggml_tensor * tensor, ggml_tensor * copy_tensor ) {
        assert( copy_tensor->buffer ); // copy to a backend
        assert( ggml_nbytes(tensor) == ggml_nbytes(copy_tensor) );
        ggml_build_forward_expand( get_graph(), tensor );
        backend_copies.push_back({ tensor, copy_tensor });
    }

    void build_forward_expand( ggml_tensor * tensor, int32_t * dst ) {
        ggml_build_forward_expand( get_graph(), tensor );
        copies.push_back({ tensor, dst });
    }

    void build_forward_expand( ggml_tensor * tensor, float * dst ) {
        ggml_build_forward_expand( get_graph(), tensor );
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
        ggml_build_forward_expand( get_graph(), sum );
        debug_sums.push_back({label, sum});
    }

    void clear() {
        debug_sums.clear();
        backend_copies.clear();
        copies.clear();
        //tensor_copies.clear();
        input_converts.clear();
        constants.clear();
        constants32.clear();
        loaders.clear();
        ggml_backend_buffer_free( buffer );
        buffer = NULL;
        ggml_reset(ctx);
        gf = NULL;
        name = "";
    }

    void compute() {
        assert( backend );
        alloc();
        // compute
        if (name.size()) {CAPTURE(name, gf);}
        ggml_backend_graph_compute( backend, gf );
        // debug
        for (auto sum : debug_sums) {
            float fsum;
            ggml_backend_tensor_get( sum.src, &fsum, 0, 4 );
            printf( "%s %f\n", sum.label, fsum );
        }
        // copy results
        for (auto copy : copies) {
            size_t nbytes = ggml_nbytes(copy.src);
            ggml_backend_tensor_get(copy.src, copy.dst, 0, nbytes);
        }
        for (auto copy : backend_copies) {
            int64_t nbytes = ggml_nbytes( copy.dst );
            std::vector<uint8_t> buf( nbytes );
            ggml_backend_tensor_get( copy.src, buf.data(), 0, nbytes );
            ggml_backend_tensor_set( copy.dst, buf.data(), 0, nbytes );
        }
        // cleanup
        clear();
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

    void new_tensor( NE ne, ggml_type type, ggml_tensor ** ptensor ) {
        // will be initialized later
        states.push_back({ ptensor, type });
        auto & state = states.back();
        for ( int i = 0; i < GGML_MAX_DIMS; i++ ) {
            state.ne[i] = ne[i];
        }
    }
    
    void new_tensor( NE ne, std::vector<float> & src, ggml_tensor ** ptensor ) {
        states.push_back({ ptensor, GGML_TYPE_F32 });
        auto & state = states.back();
        int64_t nelements = 1;
        for ( int i = 0; i < GGML_MAX_DIMS; i++ ) {
            state.ne[i] = ne[i];
            nelements *= ne[i];
        }
        assert( nelements == (int64_t)src.size() );
        state.data.resize( nelements * 4 );
        float * dst = (float*)state.data.data();
        for ( int i = 0; i < nelements; i++)
            dst[i] = src[i];
        *ptensor = NULL;
    }

    void fill16( NE ne, ggml_type type, int16_t value, ggml_tensor ** ptensor ) {
        assert( type == GGML_TYPE_F16 || type == GGML_TYPE_BF16 );
        states.push_back({ ptensor, type });
        auto & state = states.back();
        int64_t nelements = 1;
        for ( int i = 0; i < GGML_MAX_DIMS; i++ ) {
            state.ne[i] = ne[i];
            nelements *= ne[i];
        }
        state.data.resize( nelements * 2 );
        int16_t * data = (int16_t*)state.data.data();
        for ( int i = 0; i < nelements; i++)
            data[i] = value;
        *ptensor = NULL;
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
            for ( auto state : states ) {
                if ( ! state.data.size() )
                    continue;
                ggml_backend_tensor_set( *state.ptensor, state.data.data(), 0,
                    state.data.size() );
            }
        } else {
            for ( auto state : states ) {
                if ( ! state.data.size() )
                    continue;
                memcpy( (*state.ptensor)->data, state.data.data(), state.data.size() );
            }
        }
    }
};


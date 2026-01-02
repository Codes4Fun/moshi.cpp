#pragma once

#include <functional>
#include <gguf.h>
#include "crc-bbf.h"

// TODO: remove prefix
std::tuple<std::string, std::string> split_first( const std::string& input, char c ) {
    size_t pos = input.find(c);
    if (pos == std::string::npos)
        return {input, ""};
    return {input.substr(0, pos), input.substr(pos + 1)};
}

class WeightLoader {
    public:

    struct alloc_request_t {
        ggml_tensor ** result;
        int n_dims;
        NE ne;
        ggml_type type;
        std::string name;
    };

    std::string filename;

    unref_ptr<SafeTensorFile> stf;
    gguf_context * gguf;

    ScratchContext * scratch;
    ggml_backend * backend;
    ggml_context * ctx;
    ggml_backend_buffer_t buffer;
    std::vector<alloc_request_t> alloc_requests;
    std::vector<std::function<void(WeightLoader*)>> init_requests;
    bool quantize;
    ggml_type qtype;

    bool is_gguf;
    std::map<std::string,ggml_tensor*> tensors;

private:
    WeightLoader(const char * filename, SafeTensorFile * stf, ScratchContext * scratch, ggml_backend * backend = NULL) {
        this->filename = filename;
        this->stf = stf;
        this->gguf = NULL;
        this->scratch = scratch;
        this->backend = backend;
        this->ctx = NULL;
        buffer = NULL;
        quantize = false;
        qtype = GGML_TYPE_Q4_0;
        is_gguf = false;
    }
    WeightLoader(const char * filename, gguf_context * gguf, ggml_context * ctx, ScratchContext * scratch, ggml_backend * backend = NULL) {
        this->filename = filename;
        this->stf = NULL;
        this->gguf = gguf;
        this->scratch = scratch;
        this->backend = backend;
        this->ctx = ctx;
        buffer = NULL;
        quantize = false;
        qtype = GGML_TYPE_Q4_0;
        is_gguf = true;
    }
public:
    ~WeightLoader() {
        if (buffer)
            ggml_backend_buffer_free( buffer );
        if (ctx)
            ggml_free( ctx );
        if (gguf)
            gguf_free( gguf );
    }

    static WeightLoader * from_safetensor( const char * filename, ScratchContext * scratch, ggml_backend * backend = NULL ) {
        auto stf = SafeTensorFile::from_file( filename );
        if ( ! stf )
            return NULL;
        return new WeightLoader( filename, stf, scratch, backend );
    }

    static WeightLoader * from_gguf( const char * filename, ScratchContext * scratch, ggml_backend * backend = NULL ) {
        ggml_context * ctx;
        gguf_init_params params;
        params.no_alloc = true;
        params.ctx = &ctx;

        auto gguf = gguf_init_from_file( filename, params );
        if ( ! gguf ) {
            return NULL;
        }
        assert( ctx );
        auto loader = new WeightLoader( filename, gguf, ctx, scratch, backend );

        return loader;
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

    std::string tensor_name( std::string & name ) {
        auto name_size = name.size();
        if ( name_size < GGML_MAX_NAME )
            return name;
        crc_t crc;
        crc = crc_init();
        crc = crc_update(crc, (unsigned char *)name.c_str(), name_size );
        crc = crc_finalize(crc);
        std::string crc_name;
        crc_name.resize(8);
        static const char * hex = "0123456789abcdef";
        for ( int i = 0; i < 8; i++ ) {
            crc_name[i] = hex[ (crc >> 4) & 0xf ];
            crc_name[i] = hex[ crc & 0xf ];
            crc >>= 8;
        }
        return crc_name;
    }

    ggml_tensor * get_tensor( std::string & name ) {
        if ( ! gguf )
            return NULL;
        name = tensor_name( name );
        auto it = tensors.find( name );
        if ( it == tensors.end() )
            return NULL;
        return it->second;
    }

    bool fetch( ggml_tensor ** result, std::string name, ggml_type dst_type, int offset = 0 ) {
        if ( gguf ) {
            *result = get_tensor( name );
            return *result? true : false;
        }
        safetensor_t * safetensor =  find( name );
        *result = NULL;
        if (!safetensor)
            return false;
        // get source info
        ggml_type src_type = safetensor_get_type( safetensor->dtype );
        NE ne;
        int n_dims = safetensor_get_shape(safetensor, ne, offset);
        if ( dst_type == GGML_TYPE_Q4_K && ne[0] % 256 ) {
            dst_type = GGML_TYPE_Q4_0;
        }
        if ( dst_type == GGML_TYPE_Q4_0 && ne[0] % 32 ) {
            dst_type = src_type;
        }
        if ( dst_type == GGML_TYPE_Q8_K && ne[0] % 256 ) {
            dst_type = GGML_TYPE_Q8_0;
        }
        if ( dst_type == GGML_TYPE_Q8_0 && ne[0] % 32 ) {
            dst_type = src_type;
        }
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

    bool fetch( ggml_tensor ** result, std::string name, void *func = NULL, int offset = 0 ) {
        if ( gguf ) {
            *result = get_tensor( name );
            return *result? true : false;
        }
        safetensor_t * safetensor =  find( name );
        *result = NULL;
        if (!safetensor)
            return false;
        // get source info
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

    void save_gguf( const char * filename ) {
        auto gguf = gguf_init_empty();
        for ( auto tensor = ggml_get_first_tensor( ctx ); tensor;
                   tensor = ggml_get_next_tensor( ctx, tensor ) )
            gguf_add_tensor( gguf, tensor );
        gguf_write_to_file( gguf, filename, false );
    }

    bool load_gguf() {

        assert( backend );
        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);

        auto f = fopen( filename.c_str(), "rb" );

        std::vector<char*> data;
        auto data_offset = gguf_get_data_offset( gguf );
        int n_tensors = (int) gguf_get_n_tensors( gguf );
        for (int i = 0; i < n_tensors; i++) {
            std::string name = gguf_get_tensor_name( gguf, i );
            auto tensor      = ggml_get_tensor( ctx, name.c_str() );
            auto offset      = data_offset + gguf_get_tensor_offset( gguf, i );
            auto nbytes      = gguf_get_tensor_size( gguf, i );

            if ( data.size() < nbytes ) data.resize( nbytes );
#ifdef _WIN32
            auto e = _fseeki64(f, offset, SEEK_SET);
#else
            auto e = fseek(f, offset, SEEK_SET);
#endif
            assert( e == 0 );
            int64_t r = fread(data.data(), nbytes, 1, f);
            if (r != 1) {
                printf("failed to read tensor %s\n", name.c_str());
                exit(-1);
            }
            ggml_backend_tensor_set(tensor, data.data(), 0, nbytes);

            tensors[name] = tensor;
        }

        fclose( f );

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
            auto name_size = req.name.size();
            assert( name_size );
            if ( name_size ) {
                ggml_set_name( *req.result, tensor_name( req.name ).c_str() );
            }
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



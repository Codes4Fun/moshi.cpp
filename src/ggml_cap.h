#pragma once

#include <math.h>
#include <iomanip> // For std::setprecision
#include <set>
#include <map>
#include <string.h>

class GGMLContext;
class GGMLCGraph;
class GGMLTensorGroup;
class GGMLTensor;

class GGMLContext {
    public:
    ggml_context * context;
    std::vector<GGMLCGraph*> cgraphs;
    std::vector<GGMLTensor*> tensors;

    GGMLContext( ggml_context * context = NULL ) {
        this->context = context;
    }

    std::string id() {
        return std::to_string((ptrdiff_t)this);
    }
};

class GGMLCGraph {
    public:
    GGMLContext * context;
    ggml_cgraph * cgraph;
    std::vector<GGMLTensor*> tensors;
    std::string dump_filename;

    GGMLCGraph( GGMLContext * context, ggml_cgraph * cgraph = NULL ) {
        this->context = context;
        this->cgraph = cgraph;
    }

    std::string id() {
        return std::to_string((ptrdiff_t)this);
    }
};

class GGMLTensor {
    public:
    GGMLContext * context;
    ggml_tensor * tensor;
    std::vector<GGMLTensor*> src;
    std::string name;
    std::string caller;
    GGMLTensorGroup * group;
    bool side_effect; // if we are modified via cpy or inplace

    GGMLTensor( GGMLContext * context,
            GGMLTensor * src0 = NULL,
            GGMLTensor * src1 = NULL,
            GGMLTensor * src2 = NULL ) {
        this->context = context;
        this->tensor = NULL;
        if (src0)
            src.push_back( src0 );
        if (src1)
            src.push_back( src1 );
        if (src2)
            src.push_back( src2 );
        context->tensors.push_back( this );
        group = NULL;
        side_effect = false;
    }
    
    virtual ~GGMLTensor() {}

    virtual void set_side_effect() {
        side_effect = true;
    }

    std::string id() {
        return std::to_string((ptrdiff_t)this);
    }

    virtual ggml_tensor * get() = 0;

    virtual std::string & op_name() = 0;

    virtual void serialize( std::ostringstream & out ) = 0;
};

class GGMLNewTensor : public GGMLTensor {
    public:
    ggml_type type;
    int n_dims;
    int64_t ne[GGML_MAX_DIMS];
    GGMLNewTensor( GGMLContext * context, ggml_type type,
            int n_dims, const int64_t *ne ) : GGMLTensor( context ) {
        this->type = type;
        this->n_dims = n_dims;
        assert( n_dims >=1 && n_dims <= GGML_MAX_DIMS );
        for (int i = 0; i < n_dims; i++)
            this->ne[i] = ne[i];
    }
    virtual void set_side_effect() {
        side_effect = true;
    }
    virtual ggml_tensor * get() {
        if (!tensor) {
            auto ctx = context->context;
            tensor = ggml_new_tensor( ctx, type, n_dims, ne );
        }
        return tensor;
    }

    virtual std::string & op_name() {
        static std::string s_op_name = "new_tensor";
        return s_op_name;
    }

    virtual void serialize( std::ostringstream & out ) {
        /*out << "{\"type\":\"";
        out << ggml_type_name(type);
        out << ",\"ne\":[";
        const char * tail = "";
        for (int i = 0; i < n_dims; i++) {
            out << tail + std::to_string(ne[i]);
            tail = ",";
        }
        out << "]}";*/
        /*out << "[\"" << ggml_type_name(type) << "\",[";
        const char * tail = "";
        for (int i = 0; i < n_dims; i++) {
            out << tail << ne[i];
            tail = ",";
        }
        out << "]]";*/
        out << "null";
    }
};
class GGMLUnoTensor : public GGMLTensor {
    public:
    GGMLUnoTensor( GGMLContext * cx, GGMLTensor * ax ) : GGMLTensor( cx, ax ) {}

    virtual ggml_tensor * create( ggml_context * ctx, ggml_tensor * a ) = 0;

    virtual ggml_tensor * get() {
        if (!tensor) {
            auto ctx = context->context;
            auto a = src[0]->get();
            tensor = create( ctx, a );
        }
        return tensor;
    }

    virtual std::string & op_name() = 0;

    virtual void serialize( std::ostringstream & out ) {
        //out = "{\"right\":\"" + src[0]->id() + "\"}";
        out << "null";
    }
};
class GGMLPairTensor : public GGMLTensor {
    public:
    GGMLPairTensor( GGMLContext * cx, GGMLTensor * ax, GGMLTensor * bx )
        : GGMLTensor( cx, ax, bx ) {}

    virtual ggml_tensor * create( ggml_context * ctx, ggml_tensor * a, ggml_tensor * b ) = 0;

    virtual ggml_tensor * get() {
        if (!tensor) {
            auto ctx = context->context;
            auto a = src[0]->get();
            auto b = src[1]->get();
            tensor = create( ctx, a, b );
        }
        return tensor;
    }

    virtual std::string & op_name() = 0;

    virtual void serialize( std::ostringstream & out ) {
        //out = "{\"left\":\"" + src[0]->id();
        //out += "\",\"right\":\"" + src[1]->id();
        //out += "\"}";
        out << "null";
    }
};
class GGMLPairInplaceTensor : public GGMLPairTensor {
    public:
    bool inplace;
    GGMLPairInplaceTensor( GGMLContext * cx, GGMLTensor * ax, GGMLTensor * bx,
            bool inplace = false ) : GGMLPairTensor( cx, ax, bx ) {
        this->inplace = inplace;
    }

    virtual void serialize( std::ostringstream & out ) {
        //out = "{\"left\":\"" + src[0]->id();
        //out += "\",\"right\":\"" + src[1]->id();
        //out += inplace? "\",\"inplace\":true}" : "\",\"inplace\":false}";
        out << (inplace? "true" : "false");
    }
};

#define GGMLUNO_CLASS(op) \
class GGML_##op : public GGMLUnoTensor {\
    public:\
    GGML_##op ( GGMLContext * cx, GGMLTensor * ax )\
         : GGMLUnoTensor( cx, ax ) {}\
\
    virtual ggml_tensor * create( ggml_context * ctx, ggml_tensor * a ) {\
        return ggml_##op( ctx, a );\
    }\
\
    virtual std::string & op_name() {\
        static std::string s_op_name = #op;\
        return s_op_name;\
    }\
};

#define GGMLPAIR_CLASS(op) \
class GGML_##op : public GGMLPairTensor {\
    public:\
    GGML_##op ( GGMLContext * cx, GGMLTensor * ax, GGMLTensor * bx )\
         : GGMLPairTensor( cx, ax, bx ) {}\
\
    virtual ggml_tensor * create( ggml_context * ctx, ggml_tensor * a,\
            ggml_tensor * b ) {\
        return ggml_##op( ctx, a, b );\
    }\
\
    virtual std::string & op_name() {\
        static std::string s_op_name = #op;\
        return s_op_name;\
    }\
};

#define GGMLPAIRINPLACE_CLASS(op) \
class GGML_##op : public GGMLPairInplaceTensor {\
    public:\
    GGML_##op ( GGMLContext * cx, GGMLTensor * ax, GGMLTensor * bx,\
            bool inplace = false ) : GGMLPairInplaceTensor( cx, ax, bx, inplace ) {}\
\
    virtual ggml_tensor * create( ggml_context * ctx, ggml_tensor * a,\
            ggml_tensor * b ) {\
        if (inplace)\
            return ggml_##op##_inplace( ctx, a, b );\
        return ggml_##op( ctx, a, b );\
    }\
\
    virtual std::string & op_name() {\
        static std::string s_op_name = #op;\
        return s_op_name;\
    }\
};

GGMLPAIRINPLACE_CLASS(add);
GGMLPAIRINPLACE_CLASS(sub);
GGMLPAIRINPLACE_CLASS(mul);
GGMLPAIRINPLACE_CLASS(div);
GGMLUNO_CLASS(neg);
GGMLUNO_CLASS(sum);
// ggml_repeat_4d custom
// ggml_concat custom
GGMLUNO_CLASS(elu);
GGMLUNO_CLASS(gelu);
GGMLUNO_CLASS(silu);
// ggml_norm custom
// ggml_rms_norm custom
GGMLPAIR_CLASS(mul_mat);
GGMLUNO_CLASS(argmax);
// ggml_scale custom
GGMLPAIR_CLASS(cpy);
// ggml_cast custom
GGMLUNO_CLASS(cont);
// ggml_reshape_2d custom
// ggml_reshape_3d custom
// ggml_reshape_4d custom
// ggml_view_1d custom
// ggml_view_2d custom
// ggml_view_3d custom
// ggml_view_4d custom
// ggml_permute custom
GGMLUNO_CLASS(transpose);
GGMLUNO_CLASS(soft_max);
// ggml_soft_max_ext custom
GGMLPAIR_CLASS(get_rows);
// ggml_clamp custom
// ggml_conv_1d custom
// ggml_conv_transpose_1d custom
// ggml_arange custom
// ggml_top_k custom
// ggml_timestep_embedding custom


class GGML_repeat_4d : public GGMLTensor {
    public:
    int64_t ne[4];
    GGML_repeat_4d( GGMLContext * cx, GGMLTensor * tx,
            const int64_t ne[4] ) : GGMLTensor( cx, tx ) {
        for (int i = 0; i < 4; i++) this->ne[i] = ne[i];
    }
    virtual ggml_tensor * get() {
        if (!tensor) {
            auto ctx = context->context;
            auto a = src[0]->get();
            tensor = ggml_repeat_4d( ctx, a, ne[0], ne[1], ne[2], ne[3] );
        }
        return tensor;
    }
    virtual std::string & op_name() {
        static std::string s_op_name = "repeat_4d";
        return s_op_name;
    }
    virtual void serialize( std::ostringstream & out ) {
        out << "[" << ne[0];
        out << "," << ne[1];
        out << "," << ne[2];
        out << "," << ne[3] << "]";
    }
};

class GGML_concat : public GGMLTensor {
    public:
    int dim;
    GGML_concat( GGMLContext * cx, GGMLTensor * ax, GGMLTensor * bx, int dim )
        : GGMLTensor( cx, ax, bx ) {
        this->dim = dim;
    }
    virtual ggml_tensor * get() {
        if (!tensor) {
            auto ctx = context->context;
            auto a = src[0]->get();
            auto b = src[1]->get();
            tensor = ggml_concat( ctx, a, b, dim );
        }
        return tensor;
    }
    virtual std::string & op_name() {
        static std::string s_op_name = "concat";
        return s_op_name;
    }
    virtual void serialize( std::ostringstream & out ) {
        //char temp[512];
        //snprintf(temp, sizeof(temp),
        //    R"({"left":"%s","right":"%s","dim":%d})",
        //    src[0]->id().c_str(), src[1]->id().c_str(), dim );
        //out = temp;
        out << dim;
    }
};

class GGML_norm : public GGMLTensor {
    public:
    float eps;
    GGML_norm( GGMLContext * cx, GGMLTensor * tx,
            float eps ) : GGMLTensor( cx, tx ) {
        this->eps = eps;
    }
    virtual ggml_tensor * get() {
        if (!tensor) {
            auto ctx = context->context;
            auto a = src[0]->get();
            tensor = ggml_norm( ctx, a, eps );
        }
        return tensor;
    }
    virtual std::string & op_name() {
        static std::string s_op_name = "norm";
        return s_op_name;
    }
    virtual void serialize( std::ostringstream & out ) {
        //char temp[512];
        //snprintf(temp, sizeof(temp),
        //    R"({"src":"%s","eps":%f})",
        //    src[0]->id().c_str(), eps );
        //out = temp;
        out << eps;
    }
};

class GGML_rms_norm : public GGMLTensor {
    public:
    float eps;
    GGML_rms_norm( GGMLContext * cx, GGMLTensor * tx,
            float eps ) : GGMLTensor( cx, tx ) {
        this->eps = eps;
    }
    virtual ggml_tensor * get() {
        if (!tensor) {
            auto ctx = context->context;
            auto a = src[0]->get();
            tensor = ggml_rms_norm( ctx, a, eps );
        }
        return tensor;
    }
    virtual std::string & op_name() {
        static std::string s_op_name = "rms_norm";
        return s_op_name;
    }
    virtual void serialize( std::ostringstream & out ) {
        //char temp[512];
        //snprintf(temp, sizeof(temp),
        //    R"({"src":"%s","eps":%f})",
        //    src[0]->id().c_str(), eps );
        //out = temp;
        out << eps;
    }
};

class GGML_scale : public GGMLTensor {
    public:
    float s;
    GGML_scale( GGMLContext * cx, GGMLTensor * tx,
            float s ) : GGMLTensor( cx, tx ) {
        this->s = s;
    }
    virtual ggml_tensor * get() {
        if (!tensor) {
            auto ctx = context->context;
            auto a = src[0]->get();
            tensor = ggml_scale( ctx, a, s );
        }
        return tensor;
    }
    virtual std::string & op_name() {
        static std::string s_op_name = "scale";
        return s_op_name;
    }
    virtual void serialize( std::ostringstream & out ) {
        //char temp[512];
        //snprintf(temp, sizeof(temp),
        //    R"({"src":"%s","s":%f})",
        //    src[0]->id().c_str(), s );
        //out = temp;
        out << s;
    }
};

class GGML_cast : public GGMLTensor {
    public:
    ggml_type type;
    GGML_cast( GGMLContext * cx, GGMLTensor * tx,
            ggml_type type ) : GGMLTensor( cx, tx ) {
        this->type = type;
    }
    virtual ggml_tensor * get() {
        if (!tensor) {
            auto ctx = context->context;
            auto a = src[0]->get();
            tensor = ggml_cast( ctx, a, type );
        }
        return tensor;
    }
    virtual std::string & op_name() {
        static std::string s_op_name = "cast";
        return s_op_name;
    }
    virtual void serialize( std::ostringstream & out ) {
        //char temp[512];
        //snprintf(temp, sizeof(temp),
        //    R"({"src":"%s","type":"%s"})",
        //    src[0]->id().c_str(), ggml_type_name( type ) );
        //out = temp;
        out << '"' << ggml_type_name( type ) << '"';
    }
};

class GGML_reshape : public GGMLTensor {
    public:
    int n_dims;
    int64_t ne[GGML_MAX_DIMS];
    GGML_reshape( GGMLContext * cx, GGMLTensor * tx,
            int n_dims, const int64_t * ne ) : GGMLTensor( cx, tx ) {
        assert( n_dims >= 2 && n_dims <= 4 );
        this->n_dims = n_dims;
        for (int i = 0; i < n_dims; i++)
            this->ne[i] = ne[i];
    }
    virtual ggml_tensor * get() {
        if (!tensor) {
            auto ctx = context->context;
            auto a = src[0]->get();
            switch(n_dims) {
            case 2: tensor = ggml_reshape_2d( ctx, a, ne[0], ne[1] ); break;
            case 3: tensor = ggml_reshape_3d( ctx, a, ne[0], ne[1], ne[2] ); break;
            case 4: tensor = ggml_reshape_4d( ctx, a, ne[0], ne[1], ne[2], ne[3] ); break;
            }
        }
        return tensor;
    }
    virtual std::string & op_name() {
        static std::string s_op_name = "reshape";
        return s_op_name;
    }
    virtual void serialize( std::ostringstream & out ) {
        //out = R"({"src":")" + src[0]->id() + R"(","ne":[)";
        //const char * tail = "";
        //for (int i = 0; i < n_dims; i++ ) {
        //    out += tail + std::to_string(ne[i]);
        //    tail = ",";
        //}
        //out += "]}";
        out << '[';
        const char * tail = "";
        for (int i = 0; i < n_dims; i++ ) {
            out << tail << ne[i];
            tail = ",";
        }
        out << ']';
    }
};

class GGML_view : public GGMLTensor {
    public:
    int n_dims;
    int64_t ne[GGML_MAX_DIMS];
    size_t stride[GGML_MAX_DIMS-1];
    int64_t offset;
    GGML_view( GGMLContext * cx, GGMLTensor * tx, int n_dims,
            const int64_t * ne, const size_t * stride, int64_t offset )
             : GGMLTensor( cx, tx ) {
        assert( n_dims >= 1 && n_dims <= 4 );
        this->n_dims = n_dims;
        for (int i = 0; i < n_dims; i++)
            this->ne[i] = ne[i];
        for (int i = 0; i < n_dims-1; i++)
            this->stride[i] = stride[i];
        this->offset = offset;
    }
    virtual void set_side_effect() {
        src[0]->set_side_effect();
    }
    virtual ggml_tensor * get() {
        if (!tensor) {
            auto ctx = context->context;
            auto a = src[0]->get();
            switch(n_dims) {
            case 1: tensor = ggml_view_1d( ctx, a, ne[0], offset ); break;
            case 2: tensor = ggml_view_2d( ctx, a, ne[0], ne[1], stride[0], offset ); break;
            case 3: tensor = ggml_view_3d( ctx, a, ne[0], ne[1], ne[2], stride[0], stride[1], offset ); break;
            case 4: tensor = ggml_view_4d( ctx, a, ne[0], ne[1], ne[2], ne[3], stride[0], stride[1], stride[2], offset ); break;
            }
        }
        return tensor;
    }
    virtual std::string & op_name() {
        static std::string s_op_name = "view";
        return s_op_name;
    }
    virtual void serialize( std::ostringstream & out ) {
        /*out = R"({"src":")" + src[0]->id() + R"(","ne":[)";
        const char * tail = "";
        for (int i = 0; i < n_dims; i++ ) {
            out += tail + std::to_string(ne[i]);
            tail = ",";
        }
        if (n_dims > 1) {
            out += "],\"stride\":[";
            tail = "";
            for (int i = 0; i < n_dims-1; i++ ) {
                out += tail + std::to_string(stride[i]);
                tail = ",";
            }
        }
        out += "],\"offset\":" + std::to_string(offset) + "}";*/
        out << '[';
        const char * tail = "";
        for (int i = 0; i < n_dims; i++ ) {
            out << tail << ne[i];
            tail = ",";
        }
        if (n_dims > 1) {
            for (int i = 0; i < n_dims-1; i++ ) {
                out << tail << stride[i];
                tail = ",";
            }
        }
        out << tail << offset << ']';
    }
};


class GGML_permute : public GGMLTensor {
    public:
    int axis[4];
    GGML_permute( GGMLContext * cx, GGMLTensor * tx, const int axis[4] )
            : GGMLTensor( cx, tx ) {
        for (int i = 0; i < 4; i++)
            this->axis[i] = axis[i];
    }
    virtual void set_side_effect() {
        src[0]->set_side_effect();
    }
    virtual ggml_tensor * get() {
        if (!tensor) {
            auto ctx = context->context;
            auto a = src[0]->get();
            tensor = ggml_permute( ctx, a, axis[0], axis[1], axis[2], axis[3] );
        }
        return tensor;
    }
    virtual std::string & op_name() {
        static std::string s_op_name = "permute";
        return s_op_name;
    }
    virtual void serialize( std::ostringstream & out ) {
        /*out = R"({"src":")" + src[0]->id() + R"(","axis":[)";
        const char * tail = "";
        for (int i = 0; i < 4; i++ ) {
            out += tail + std::to_string(axis[i]);
            tail = ",";
        }
        out += "]}";*/
        out << '[' << axis[0];
        out << ',' << axis[1];
        out << ',' << axis[2];
        out << ',' << axis[3] << ']';
    }
};

class GGML_soft_max_ext : public GGMLTensor {
    public:
    float scale;
    float max_bias;
    GGML_soft_max_ext( GGMLContext * cx, GGMLTensor * tx, GGMLTensor * mask,
            float scale, float max_bias ) : GGMLTensor( cx, tx, mask ) {
        this->scale = scale;
        this->max_bias = max_bias;
    }
    virtual ggml_tensor * get() {
        if (!tensor) {
            auto ctx = context->context;
            auto a = src[0]->get();
            auto mask = src.size() > 1? src[1]->get() : (ggml_tensor*)NULL;
            tensor = ggml_soft_max_ext( ctx, a, mask, scale, max_bias );
        }
        return tensor;
    }
    virtual std::string & op_name() {
        static std::string s_op_name = "soft_max_ext";
        return s_op_name;
    }
    virtual void serialize( std::ostringstream & out ) {
        /*char temp[1024];
        if (src.size() > 1) {
            snprintf(temp, sizeof(temp),
                R"({"src":"%s","mask":"%s","scale":%f,"max_bias":%f})",
                src[0]->id().c_str(), src[1]->id().c_str(), scale, max_bias );
        } else {
            snprintf(temp, sizeof(temp),
                R"({"src":"%s","scale":%f,"max_bias":%f})",
                src[0]->id().c_str(), scale, max_bias );
        }
        out = temp;*/
        out << '[' << scale << ',' << max_bias << ']';
    }
};

class GGML_clamp : public GGMLTensor {
    public:
    float min;
    float max;
    GGML_clamp( GGMLContext * cx, GGMLTensor * tx, float min, float max )
            : GGMLTensor( cx, tx ) {
        this->min = min;
        this->max = max;
    }
    virtual ggml_tensor * get() {
        if (!tensor) {
            auto ctx = context->context;
            auto a = src[0]->get();
            tensor = ggml_clamp( ctx, a, min, max );
        }
        return tensor;
    }
    virtual std::string & op_name() {
        static std::string s_op_name = "clamp";
        return s_op_name;
    }
    virtual void serialize( std::ostringstream & out ) {
        /*char temp[1024];
        snprintf(temp, sizeof(temp),
            R"({"src":"%s","min":%f,"max":%f})",
            src[0]->id().c_str(), min, max );
        out = temp;*/
        out << '[' << min << ',' << max << ']';
    }
};

class GGML_conv_1d : public GGMLTensor {
    public:
    int s0;
    int p0;
    int d0;
    GGML_conv_1d( GGMLContext * cx, GGMLTensor * ax, GGMLTensor * bx, int s0,
            int p0, int d0 ) : GGMLTensor( cx, ax, bx ) {
        this->s0 = s0;
        this->p0 = p0;
        this->d0 = d0;
    }
    virtual ggml_tensor * get() {
        if (!tensor) {
            auto ctx = context->context;
            auto a = src[0]->get();
            auto b = src[1]->get();
            tensor = ggml_conv_1d( ctx, a, b, s0, p0, d0 );
        }
        return tensor;
    }
    virtual std::string & op_name() {
        static std::string s_op_name = "conv_1d";
        return s_op_name;
    }
    virtual void serialize( std::ostringstream & out ) {
        /*char temp[1024];
        snprintf(temp, sizeof(temp),
            R"({"input":"%s","weight":"%s","s0":%d,"p0":%d,"d0":%d})",
            src[0]->id().c_str(), src[1]->id().c_str(), s0, p0, d0 );
        out = temp;*/
        out << '[' << s0 << ',' << p0 << ',' << d0 << ']';
    }
};

class GGML_conv_transpose_1d : public GGMLTensor {
    public:
    int s0;
    int p0;
    int d0;
    GGML_conv_transpose_1d( GGMLContext * cx, GGMLTensor * ax, GGMLTensor * bx, int s0,
            int p0, int d0 ) : GGMLTensor( cx, ax, bx ) {
        this->s0 = s0;
        this->p0 = p0;
        this->d0 = d0;
    }
    virtual ggml_tensor * get() {
        if (!tensor) {
            auto ctx = context->context;
            auto a = src[0]->get();
            auto b = src[1]->get();
            tensor = ggml_conv_transpose_1d( ctx, a, b, s0, p0, d0 );
        }
        return tensor;
    }
    virtual std::string & op_name() {
        static std::string s_op_name = "conv_transpose_1d";
        return s_op_name;
    }
    virtual void serialize( std::ostringstream & out ) {
        /*char temp[1024];
        snprintf(temp, sizeof(temp),
            R"({"input":"%s","weight":"%s","s0":%d,"p0":%d,"d0":%d})",
            src[0]->id().c_str(), src[1]->id().c_str(), s0, p0, d0 );
        out = temp;*/
        out << '[' << s0 << ',' << p0 << ',' << d0 << ']';
    }
};

class GGML_arange : public GGMLTensor {
    public:
    float start;
    float stop;
    float step;
    GGML_arange( GGMLContext * cx, float start, float stop, float step )
            : GGMLTensor( cx ) {
        this->start = start;
        this->stop = stop;
        this->step = step;
    }
    virtual ggml_tensor * get() {
        if (!tensor) {
            auto ctx = context->context;
            tensor = ggml_arange( ctx, start, stop, step );
        }
        return tensor;
    }
    virtual std::string & op_name() {
        static std::string s_op_name = "arange";
        return s_op_name;
    }
    virtual void serialize( std::ostringstream & out ) {
        /*char temp[1024];
        snprintf(temp, sizeof(temp),
            R"({"start":%f,"stop":%f,"step":%f})",
            start, stop, step );
        out = temp;*/
        out << '[' << start << ',' << stop << ',' << step << ']';
    }
};

class GGML_top_k : public GGMLTensor {
    public:
    int k;
    GGML_top_k( GGMLContext * cx, GGMLTensor * tx, int k )
            : GGMLTensor( cx, tx ) {
        this->k = k;
    }
    virtual ggml_tensor * get() {
        if (!tensor) {
            auto ctx = context->context;
            auto a = src[0]->get();
            tensor = ggml_top_k( ctx, a, k );
        }
        return tensor;
    }
    virtual std::string & op_name() {
        static std::string s_op_name = "top_k";
        return s_op_name;
    }
    virtual void serialize( std::ostringstream & out ) {
        /*char temp[1024];
        snprintf(temp, sizeof(temp),
            R"({"src":"%s","k":%d})",
            src[0]->id().c_str(), k );
        out = temp;*/
        out << k;
    }
};

class GGML_timestep_embedding : public GGMLTensor {
    public:
    int dim;
    int max_period;
    GGML_timestep_embedding( GGMLContext * cx, GGMLTensor * tx, int dim, int max_period )
            : GGMLTensor( cx, tx ) {
        this->dim = dim;
        this->max_period = max_period;
    }
    virtual ggml_tensor * get() {
        if (!tensor) {
            auto ctx = context->context;
            auto a = src[0]->get();
            tensor = ggml_timestep_embedding( ctx, a, dim, max_period );
        }
        return tensor;
    }
    virtual std::string & op_name() {
        static std::string s_op_name = "timestep_embedding";
        return s_op_name;
    }
    virtual void serialize( std::ostringstream & out ) {
        /*char temp[1024];
        snprintf(temp, sizeof(temp),
            R"({"timesteps":"%s","dim":%d,"max_period":%d})",
            src[0]->id().c_str(), dim, max_period );
        out = temp;*/
        out << '[' << dim << ',' << max_period << ']';
    }
};

class GGML_sum_rows : public GGMLTensor {
    public:
    int dim;
    int max_period;
    GGML_sum_rows( GGMLContext * cx, GGMLTensor * tx )
            : GGMLTensor( cx, tx ) { }
    virtual ggml_tensor * get() {
        if ( ! tensor ) {
            auto ctx = context->context;
            auto a = src[0]->get();
            tensor = ggml_sum_rows( ctx, a );
        }
        return tensor;
    }
    virtual std::string & op_name() {
        static std::string s_op_name = "sum_rows";
        return s_op_name;
    }
    virtual void serialize( std::ostringstream & out ) {
        out << "null";
    }
};



class GGMLTensorGroup {
    public:
    std::string name;
    GGMLTensorGroup * parent;
    bool locked;
    std::set<GGMLTensor*> tensors;
    std::set<GGMLTensorGroup*> children;

    GGMLTensorGroup( GGMLTensorGroup * parent, std::string name ) {
        this->parent = parent;
        this->name = name;
        locked = true;
        if (parent)
            parent->children.insert( this );
    }
    ~GGMLTensorGroup() {
        for (auto child : children)
            delete child;
        if (parent)
            parent->children.erase( this );
    }
    bool can_free() {
        if (locked)
            return false;
        if (tensors.size())
            return false;
        for (auto child : children)
            if (!child->can_free())
                return false;
        return true;
    }

    std::string id() {
        return std::to_string((ptrdiff_t)this);
    }
};

class GGMLTensorGroupRoot : public GGMLTensorGroup {
    public:
    std::vector<GGMLTensorGroup*> stack;
    GGMLTensorGroupRoot() : GGMLTensorGroup(NULL, "root") { }

    void push_group( std::string name ) {
        auto parent = stack.size()? stack.back() : NULL;
        auto group = new GGMLTensorGroup( parent, name );
        stack.push_back( group );
    }

    void pop_group() {
        assert( stack.size() > 0 );
        auto group = stack.back();
        stack.pop_back();
        group->locked = false;
        if (group->can_free()) {
            delete group;
        }
    }

    void add_tensor( GGMLTensor * tx ) {
        assert( !tx->group );
        if (stack.size()) {
            auto group = stack.back();
            group->tensors.insert( tx );
            tx->group = group;
        }
    }

    void remove_tensor( GGMLTensor * tx ) {
        if (!tx->group)
            return;
        auto group = tx->group;
        tx->group = NULL;
        group->tensors.erase( tx );
        while (group->can_free()) {
            auto parent = group->parent;
            delete group;
            if (!parent)
                break;
            group = parent;
        }
    }
};



class GGMLCapture {
    public:
    std::map<ggml_context*,GGMLContext*> contexts;
    std::map<ggml_cgraph*,GGMLCGraph*> cgraphs;
    std::map<ggml_tensor*,GGMLTensor*> tensors;

    GGMLContext * get( ggml_context * context ) {
        auto it = contexts.find( context );
        assert( it != contexts.end() );
        return it->second;
    }

    GGMLCGraph * get( ggml_cgraph * cgraph ) {
        auto it = cgraphs.find( cgraph );
        assert( it != cgraphs.end() );
        return it->second;
    }

    GGMLTensor * get( ggml_tensor * tensor ) {
        auto it = tensors.find( tensor );
        assert( it != tensors.end() );
        return it->second;
    }

    GGMLContext * create( ggml_context * context ) {
        assert( contexts.find( context ) == contexts.end() );
        auto cx = new GGMLContext( context );
        contexts[context] = cx;
        return cx;
    }

    GGMLCGraph * create( ggml_context * context, ggml_cgraph * cgraph ) {
        GGMLContext * cx = get( context );
        assert( cgraphs.find( cgraph ) == cgraphs.end() );
        auto gx = new GGMLCGraph( cx, cgraph );
        cgraphs[cgraph] = gx;
        cx->cgraphs.push_back( gx );
        return gx;
    }

    /*GGMLTensor * create( ggml_context * context, ggml_tensor * tensor ) {
        GGMLContext * cx = get( context );
        assert( tensors.find( tensor ) == tensors.end() );
        auto tx = new GGMLTensor( cx, tensor );
        tensors[tensor] = tx;
        cx->tensors.push_back( tx );
        return tx;
    }*/

    GGMLTensorGroupRoot group_root;
    void push_group( std::string name ) {
        group_root.push_group( name );
    }
    void pop_group() {
        group_root.pop_group();
    }

    void free( GGMLTensor * tx ) {
        auto it = tensors.find( tx->tensor );
        assert( it != tensors.end() );
        tensors.erase( it );
        group_root.remove_tensor( tx );
        delete tx;
    }

    void free( GGMLCGraph * gx ) {
        auto it = cgraphs.find( gx->cgraph );
        assert( it != cgraphs.end() );
        cgraphs.erase( it );
        delete gx;
    }

    void reset( GGMLContext * cx ) {
        for (auto gx : cx->cgraphs) {
            free( gx );
        }
        cx->cgraphs.clear();
        for (auto tx : cx->tensors) {
            free( tx );
        }
        cx->tensors.clear();
    }

    void reset( ggml_context * context ) {
        reset( get( context ) );
    }

    void free( ggml_context * context ) {
        auto it = contexts.find(context);
        assert( it != contexts.end() );
        reset( it->second );
        delete it->second;
        contexts.erase( it );
    }

    GGMLTensor * create_new_tensor( ggml_context * ctx, ggml_type type,
            int n_dims, const int64_t * ne ) {
        auto cx = get( ctx );
        auto tx = new GGMLNewTensor( cx, type, n_dims, ne );
        auto tensor = tx->get();
        tensors[tensor] = tx;
        group_root.add_tensor( tx );
        return tx;
    }

#define GGML_UNO_CREATE(op) \
    GGMLTensor * create_##op( ggml_context * ctx, ggml_tensor * a ) {\
        auto cx = get( ctx );\
        auto ax = get( a );\
        auto tx = new GGML_##op( cx, ax );\
        auto tensor = tx->get();\
        tensors[tensor] = tx;\
        group_root.add_tensor( tx );\
        return tx;\
    }

#define GGML_PAIR_CREATE(op) \
    GGMLTensor * create_##op( ggml_context * ctx, ggml_tensor * a, ggml_tensor * b ) {\
        auto cx = get( ctx );\
        auto ax = get( a );\
        auto bx = get( b );\
        auto tx = new GGML_##op( cx, ax, bx );\
        auto tensor = tx->get();\
        tensors[tensor] = tx;\
        group_root.add_tensor( tx );\
        return tx;\
    }

#define GGML_PAIR_INPLACE_CREATE(op) \
    GGMLTensor * create_##op( ggml_context * ctx, ggml_tensor * a, ggml_tensor * b, bool inplace = false ) {\
        auto cx = get( ctx );\
        auto ax = get( a );\
        auto bx = get( b );\
        if (inplace)\
            ax->set_side_effect();\
        auto tx = new GGML_##op( cx, ax, bx, inplace );\
        auto tensor = tx->get();\
        tensors[tensor] = tx;\
        group_root.add_tensor( tx );\
        return tx;\
    }

    GGML_PAIR_INPLACE_CREATE(add);
    GGML_PAIR_INPLACE_CREATE(sub);
    GGML_PAIR_INPLACE_CREATE(mul);
    GGML_PAIR_INPLACE_CREATE(div);
    GGML_UNO_CREATE(neg);
    GGML_UNO_CREATE(sum);
    // ggml_repeat_4d custom
    // ggml_concat custom
    GGML_UNO_CREATE(elu);
    GGML_UNO_CREATE(gelu);
    GGML_UNO_CREATE(silu);
    // ggml_norm custom
    // ggml_rms_norm custom
    GGML_PAIR_CREATE(mul_mat);
    GGML_UNO_CREATE(argmax);
    // ggml_scale custom
    //GGML_PAIR_CREATE(cpy);
    // ggml_cast custom
    GGML_UNO_CREATE(cont);
    // ggml_reshape_2d custom
    // ggml_reshape_3d custom
    // ggml_reshape_4d custom
    // ggml_view_1d custom
    // ggml_view_2d custom
    // ggml_view_3d custom
    // ggml_view_4d custom
    // ggml_permute custom
    GGML_UNO_CREATE(transpose);
    GGML_UNO_CREATE(soft_max);
    // ggml_soft_max_ext custom
    GGML_PAIR_CREATE(get_rows);
    // ggml_clamp custom
    // ggml_conv_1d custom
    // ggml_conv_transpose_1d custom
    // ggml_arange custom
    // ggml_top_k custom
    // ggml_timestep_embedding custom

    GGMLTensor * create_repeat_4d( ggml_context * ctx, ggml_tensor * a, const int64_t ne[4] ) {
        auto cx = get( ctx );
        auto ax = get( a );
        auto tx = new GGML_repeat_4d( cx, ax, ne );
        auto tensor = tx->get();
        tensors[tensor] = tx;
        group_root.add_tensor( tx );
        return tx;
    }

    GGMLTensor * create_concat( ggml_context * ctx, ggml_tensor * a, ggml_tensor * b, int64_t dim ) {
        auto cx = get( ctx );
        auto ax = get( a );
        auto bx = get( b );
        auto tx = new GGML_concat( cx, ax, bx, dim );
        auto tensor = tx->get();
        tensors[tensor] = tx;
        group_root.add_tensor( tx );
        return tx;
    }

    GGMLTensor * create_norm( ggml_context * ctx, ggml_tensor * a, float eps ) {
        auto cx = get( ctx );
        auto ax = get( a );
        auto tx = new GGML_norm( cx, ax, eps );
        auto tensor = tx->get();
        tensors[tensor] = tx;
        group_root.add_tensor( tx );
        return tx;
    }

    GGMLTensor * create_rms_norm( ggml_context * ctx, ggml_tensor * a, float eps ) {
        auto cx = get( ctx );
        auto ax = get( a );
        auto tx = new GGML_rms_norm( cx, ax, eps );
        auto tensor = tx->get();
        tensors[tensor] = tx;
        group_root.add_tensor( tx );
        return tx;
    }

    GGMLTensor * create_scale( ggml_context * ctx, ggml_tensor * a, float s ) {
        auto cx = get( ctx );
        auto ax = get( a );
        auto tx = new GGML_scale( cx, ax, s );
        auto tensor = tx->get();
        tensors[tensor] = tx;
        group_root.add_tensor( tx );
        return tx;
    }

    GGMLTensor * create_cpy( ggml_context * ctx, ggml_tensor * a, ggml_tensor * b ) {
        auto cx = get( ctx );
        auto ax = get( a );
        auto bx = get( b );
        bx->set_side_effect();
        auto tx = new GGML_cpy( cx, ax, bx );
        auto tensor = tx->get();
        tensors[tensor] = tx;
        group_root.add_tensor( tx );
        return tx;
    }

    GGMLTensor * create_cast( ggml_context * ctx, ggml_tensor * a, ggml_type type ) {
        auto cx = get( ctx );
        auto ax = get( a );
        auto tx = new GGML_cast( cx, ax, type );
        auto tensor = tx->get();
        tensors[tensor] = tx;
        group_root.add_tensor( tx );
        return tx;
    }

    GGMLTensor * create_reshape( ggml_context * ctx, ggml_tensor * a, int n_dims, const int64_t * ne ) {
        auto cx = get( ctx );
        auto ax = get( a );
        auto tx = new GGML_reshape( cx, ax, n_dims, ne );
        auto tensor = tx->get();
        tensors[tensor] = tx;
        group_root.add_tensor( tx );
        return tx;
    }

    GGMLTensor * create_view( ggml_context * ctx, ggml_tensor * a, int n_dims, const int64_t * ne, const size_t * stride, size_t offset ) {
        auto cx = get( ctx );
        auto ax = get( a );
        auto tx = new GGML_view( cx, ax, n_dims, ne, stride, offset );
        auto tensor = tx->get();
        tensors[tensor] = tx;
        group_root.add_tensor( tx );
        return tx;
    }

    GGMLTensor * create_permute( ggml_context * ctx, ggml_tensor * a, const int axis[4] ) {
        auto cx = get( ctx );
        auto ax = get( a );
        auto tx = new GGML_permute( cx, ax, axis );
        auto tensor = tx->get();
        tensors[tensor] = tx;
        group_root.add_tensor( tx );
        return tx;
    }

    GGMLTensor * create_soft_max_ext( ggml_context * ctx, ggml_tensor  * a,
            ggml_tensor  * mask, float scale, float max_bias ) {
        auto cx = get( ctx );
        auto ax = get( a );
        auto mx = mask? get( mask ) : (GGMLTensor*)NULL;
        auto tx = new GGML_soft_max_ext( cx, ax, mx, scale, max_bias );
        auto tensor = tx->get();
        tensors[tensor] = tx;
        group_root.add_tensor( tx );
        return tx;
    }

    GGMLTensor * create_clamp( ggml_context * ctx, ggml_tensor  * a,
            float min, float max ) {
        auto cx = get( ctx );
        auto ax = get( a );
        ax->set_side_effect();
        auto tx = new GGML_clamp( cx, ax, min, max );
        auto tensor = tx->get();
        tensors[tensor] = tx;
        group_root.add_tensor( tx );
        return tx;
    }

    GGMLTensor * create_conv_1d( ggml_context * ctx, ggml_tensor  * a,
            ggml_tensor * b, int s0, int p0, int d0 ) {
        auto cx = get( ctx );
        auto ax = get( a );
        auto bx = get( b );
        auto tx = new GGML_conv_1d( cx, ax, bx, s0, p0, d0 );
        auto tensor = tx->get();
        tensors[tensor] = tx;
        group_root.add_tensor( tx );
        return tx;
    }

    GGMLTensor * create_conv_transpose_1d( ggml_context * ctx, ggml_tensor  * a,
            ggml_tensor * b, int s0, int p0, int d0 ) {
        auto cx = get( ctx );
        auto ax = get( a );
        auto bx = get( b );
        auto tx = new GGML_conv_transpose_1d( cx, ax, bx, s0, p0, d0 );
        auto tensor = tx->get();
        tensors[tensor] = tx;
        group_root.add_tensor( tx );
        return tx;
    }

    GGMLTensor * create_arange( ggml_context * ctx, float start, float stop, float step ) {
        auto cx = get( ctx );
        auto tx = new GGML_arange( cx, start, stop, step );
        auto tensor = tx->get();
        tensors[tensor] = tx;
        group_root.add_tensor( tx );
        return tx;
    }

    GGMLTensor * create_top_k( ggml_context * ctx, ggml_tensor  * a, int k ) {
        auto cx = get( ctx );
        auto ax = get( a );
        auto tx = new GGML_top_k( cx, ax, k );
        auto tensor = tx->get();
        tensors[tensor] = tx;
        group_root.add_tensor( tx );
        return tx;
    }

    GGMLTensor * create_timestep_embedding( ggml_context * ctx,
            ggml_tensor  * timesteps, int dim, int max_period ) {
        auto cx = get( ctx );
        auto timestepx = get( timesteps );
        auto tx = new GGML_timestep_embedding( cx, timestepx, dim, max_period );
        auto tensor = tx->get();
        tensors[tensor] = tx;
        group_root.add_tensor( tx );
        return tx;
    }

    GGMLTensor * create_sum_rows( ggml_context * ctx, ggml_tensor  * a ) {
        auto cx = get( ctx );
        auto ax = get( a );
        auto tx = new GGML_sum_rows( cx, ax );
        auto tensor = tx->get();
        tensors[tensor] = tx;
        group_root.add_tensor( tx );
        return tx;
    }


    private:
    GGMLCapture() {}
    public:
    static GGMLCapture & Get() {
        static GGMLCapture singleton;
        return singleton;
    }
};


class GroupScope {
    public:
    GroupScope( std::string name ) {
        GGMLCapture::Get().push_group( name );
    }
    ~GroupScope() {
        GGMLCapture::Get().pop_group();
    }
};
#define CAPTURE_GROUP(name) GroupScope _group_scope(name)

class Collector {
    public:
    std::set<GGMLTensor*> tensors;
    std::set<GGMLTensorGroup*> groups;
    Collector() {}
    void insert( GGMLTensorGroup * group ) {
        if (!group)
            return;
        if (groups.find( group ) != groups.end())
            return;
        groups.insert( group );
        insert( group->parent );
    }
    void insert( GGMLTensor * tx ) {
        if (tensors.find( tx ) != tensors.end())
            return;
        tensors.insert( tx );
        for (auto src : tx->src) {
            insert( src );
        }
        insert( tx->group );
    }
};

void insert( std::set<GGMLTensor*> &set, GGMLTensor * tx ) {
    if (set.find( tx ) != set.end())
        return;
    set.insert( tx );
    for (auto src : tx->src) {
        insert( set, src );
    }
}

/*void graph_dump_precompute( std::string filename, ggml_cgraph * cgraph ) {
    Collector collector;
    auto gx = GGMLCapture::Get().get( cgraph );
    //std::set<GGMLTensor*> set;
    for (auto tx : gx->tensors) {
        //insert( set, tx );
        collector.insert( tx );
    }
    auto fbin = fopen((filename + ".tensors").c_str(), "wb");
}*/

void graph_dump( std::string filename, ggml_cgraph * cgraph ) {
    Collector collector;
    auto gx = GGMLCapture::Get().get( cgraph );
    //std::set<GGMLTensor*> set;
    for (auto tx : gx->tensors) {
        //insert( set, tx );
        collector.insert( tx );
    }
    auto fbin = fopen(("capture/" + filename + ".tensors").c_str(), "wb");
    assert( fbin );
    std::ostringstream out;
    out << "{\"tensor\":{\n";
    size_t total_nbytes = 0;
    std::vector<uint8_t> buf;
    const char * tail = "";
    for (auto tx : collector.tensors) {
        out << tail;
        tail = ",\n";
        out << '"' << tx->id() << "\":[";
        out << '"' << tx->op_name() << "\",";
        // parameters
        out << '[';
        const char * tail2 = "\"";
        for (auto src : tx->src) {
            out << tail2 << src->id() << "\"";
            tail2 = ",\"";
        }
        out << "],";
        tx->serialize(out);
        // file offset and size
        out << ",[\"" << ggml_type_name(tx->tensor->type);
        out << "\",[" << tx->tensor->ne[0];
        out << "," << tx->tensor->ne[1];
        out << "," << tx->tensor->ne[2];
        out << "," << tx->tensor->ne[3] << "],";
        if ( !tx->tensor->data || (tx->op_name() != "new_tensor" && tx->side_effect ) ) {
            out << "0,0]";
        } else if (tx->tensor->view_src) {
            // TODO: maybe add support for tensors that use views
            //out << "0,0]";
            assert( tx->tensor->buffer ); // TODO: support non-backend tensors
            auto view_src = tx->tensor->view_src;
            assert( view_src->type == GGML_TYPE_F32 || view_src->type == GGML_TYPE_I32 );
            auto src_nbytes = ggml_nbytes( view_src );
            if (buf.size() < src_nbytes) buf.resize(src_nbytes);
            ggml_backend_tensor_get( view_src, buf.data(), 0, src_nbytes );

            //values.resize( ggml_nelements( result ));
            //uint8_t * dst = (uint8_t*)values.data();

            size_t nbytes = ggml_nbytes( tx->tensor );
            out << total_nbytes << ',' << nbytes << ']';

            uint8_t * src = buf.data() + tx->tensor->view_offs;
            auto nb1 = tx->tensor->nb[1];
            auto nb2 = tx->tensor->nb[2];
            auto nb3 = tx->tensor->nb[3];
            auto cpy_nb1 = tx->tensor->ne[1] * tx->tensor->nb[0];
            size_t actual_nbytes = 0;
            for ( int ne3 = 0; ne3 < tx->tensor->ne[3]; ne3++ ) {
                for ( int ne2 = 0; ne2 < tx->tensor->ne[2]; ne2++ ) {
                    for ( int ne1 = 0; ne1 < tx->tensor->ne[1]; ne1++ ) {
                        int offset = ne1 * nb1 +ne2 * nb2 +ne3 * nb3;
                        //memcpy( dst, src + offset, cpy_nb1 );
                        auto w = fwrite(src + offset, cpy_nb1, 1, fbin);
                        assert( w == 1 );
                        actual_nbytes += cpy_nb1;
                        //dst += cpy_nb1;
                    }
                }
            }
            assert( actual_nbytes == nbytes );
            total_nbytes += nbytes;
        } else {
            if (tx->op_name() == "repeat_4d") {
                printf("repeat_4d\n");
            }
            size_t nbytes = ggml_nbytes( tx->tensor );
            out << total_nbytes << ',' << nbytes << ']';
            if (tx->tensor->buffer) {
                if (buf.size() < nbytes) buf.resize(nbytes);
                ggml_backend_tensor_get( tx->tensor, buf.data(), 0, nbytes );
                auto w = fwrite(buf.data(), nbytes, 1, fbin);
                assert( w == 1 );
            } else {
                auto w = fwrite(tx->tensor->data, nbytes, 1, fbin);
                assert( w == 1 );
            }
            total_nbytes += nbytes;
        }
        out << ",\"" << tx->name << '"';
        if (tx->group)
            out << ",\"" << tx->group->id() << '"';
        else
            out << ",\"0\"";
        out << ",\"" << tx->caller << '"';
        out << ']';
    }
    fclose( fbin );
    out << "},\n\"groups\":{\n";
    tail = "\"";
    for (auto group : collector.groups) {
        out << tail << group->id() << "\":[";
        tail = ",\n\"";
        out << '"' << group->name << '"';
        // parent
        if (group->parent)
            out << ",\"" << group->parent->id() << "\",[";
        else
            out << ",\"0\",[";
        // tensors
        const char * tail2 = "\"";
        for (auto tx : group->tensors) {
            if (collector.tensors.find( tx ) == collector.tensors.end())
                continue;
            out << tail2 << tx->id() << '"';
            tail2 = ",\"";
        }
        // children
        out << "],[";
        tail2 = "\"";
        for (auto child : group->children) {
            if (collector.groups.find( child ) == collector.groups.end())
                continue;
            out << tail2 << child->id() << '"';
            tail2 = ",\"";
        }
        out << "]]";
    }
    out << "},\n\"forward_expand\":[\n";
    tail = "\"";
    for (auto tx : gx->tensors) {
        out << tail << tx->id() << '"';
        tail = ",\n\"";
    }
    out << "],\n\"nbytes\":" << total_nbytes << "}";
    auto json = out.str();
    auto f = fopen(("capture/" + filename + ".json").c_str(), "wb");
    assert( f );
    auto w = fwrite( json.c_str(), json.size(), 1, f );
    assert( w == 1 );
    fclose( f );
}
//#define CAPTURE(filename, cgraph) graph_dump(filename, cgraph)

class GraphDumper {
    protected:
    GGMLCGraph * gx;
    std::set<GGMLTensor*> new_tensors;
    std::set<GGMLTensor*> tensors;
    std::set<GGMLTensorGroup*> groups;
    FILE * fbin;
    std::ostringstream out;
    size_t total_nbytes;
    std::vector<uint8_t> buf;

    void insert( GGMLTensorGroup * group ) {
        if (!group)
            return;
        if (groups.find( group ) != groups.end())
            return;
        groups.insert( group );
        insert( group->parent );
    }
    void insert( GGMLTensor * tx ) {
        if (tx->op_name() == "new_tensor") {
            new_tensors.insert( tx );
            assert( tx->src.size() == 0 );
        } else {
            if (tensors.find( tx ) != tensors.end())
                return;
            tensors.insert( tx );
            for (auto src : tx->src) {
                insert( src );
            }
        }
        insert( tx->group );
    }

    void write_bytes( ggml_tensor * tensor ) {
        if ( tensor->view_src ) {
            assert( tensor->buffer ); // TODO: support non-backend tensors
            auto view_src = tensor->view_src;
            assert( view_src->type == GGML_TYPE_F32 || view_src->type == GGML_TYPE_I32 || view_src->type == GGML_TYPE_BF16 );
            auto src_nbytes = ggml_nbytes( view_src );
            if (buf.size() < src_nbytes) buf.resize(src_nbytes);
            ggml_backend_tensor_get( view_src, buf.data(), 0, src_nbytes );

            //values.resize( ggml_nelements( result ));
            //uint8_t * dst = (uint8_t*)values.data();

            auto nelements = ggml_nelements( tensor );
            size_t nbe = 4;
            if ( view_src->type == GGML_TYPE_BF16 || view_src->type == GGML_TYPE_F16 )
                nbe = 2;
            size_t nbytes = nelements * nbe;
            out << total_nbytes << ',' << nbytes << ']';

            uint8_t * src = buf.data() + tensor->view_offs;
            auto nb0 = tensor->nb[0];
            auto nb1 = tensor->nb[1];
            auto nb2 = tensor->nb[2];
            auto nb3 = tensor->nb[3];
            //auto cpy_nb1 = tensor->ne[0] * tensor->nb[0];
            size_t actual_nbytes = 0;
            for ( int ne3 = 0; ne3 < tensor->ne[3]; ne3++ ) {
                for ( int ne2 = 0; ne2 < tensor->ne[2]; ne2++ ) {
                    for ( int ne1 = 0; ne1 < tensor->ne[1]; ne1++ ) {
                        for ( int ne0 = 0; ne0 < tensor->ne[0]; ne0++ ) {
                            int offset =
                                ne0 * nb0 +
                                ne1 * nb1 +
                                ne2 * nb2 +
                                ne3 * nb3;
                            //memcpy( dst, src + offset, cpy_nb1 );
                            auto w = fwrite(src + offset, nbe, 1, fbin);
                            assert( w == 1 );
                            actual_nbytes += nbe;
                            //dst += cpy_nb1;
                        }
                    }
                }
            }
            assert( actual_nbytes == nbytes );
            total_nbytes += nbytes;
            return;
        }
        size_t nbytes = ggml_nbytes( tensor );
        out << total_nbytes << ',' << nbytes << ']';
        if (tensor->buffer) {
            if (buf.size() < nbytes) buf.resize(nbytes);
            ggml_backend_tensor_get( tensor, buf.data(), 0, nbytes );
            auto w = fwrite(buf.data(), nbytes, 1, fbin);
            assert( w == 1 );
        } else {
            auto w = fwrite(tensor->data, nbytes, 1, fbin);
            assert( w == 1 );
        }
        total_nbytes += nbytes;
    }

    public:
    GraphDumper( GGMLCGraph * gx ) {
        this->gx = gx;
        for (auto tx : gx->tensors) {
            insert( tx );
        }
        fbin = fopen(("capture/" + gx->dump_filename + ".tensors").c_str(), "wb");
        assert( fbin );
        // start writing new_tensors and dumping their initial state
        out << "{\"tensor\":{\n";
        total_nbytes = 0;
        const char * tail = "";
        for (auto tx : new_tensors) {
            auto ntx = dynamic_cast<GGMLNewTensor*>(tx);
            out << tail;
            tail = ",\n";
            out << '"' << tx->id() << "\":[";
            out << "\"new_tensor\",[],null,[\"";
            out << ggml_type_name(tx->tensor->type);
            out << "\",[" << tx->tensor->ne[0];
            for (int i = 1; i < ntx->n_dims; i++) {
                out << "," << tx->tensor->ne[i];
            }
            out << "],";
            write_bytes( tx-> tensor );
            out << ",\"" << tx->name << '"';
            if (tx->group)
                out << ",\"" << tx->group->id() << '"';
            else
                out << ",\"0\"";
            out << ",\"" << tx->caller << '"';
            out << ']';
        }
    }

    void post_compute() {
        assert( fbin );
        const char * tail = new_tensors.size()? ",\n" : "";
        for (auto tx : tensors) {
            out << tail;
            tail = ",\n";
            out << '"' << tx->id() << "\":[";
            out << '"' << tx->op_name() << "\",";
            // parameters
            out << '[';
            const char * tail2 = "\"";
            for (auto src : tx->src) {
                out << tail2 << src->id() << "\"";
                tail2 = ",\"";
            }
            out << "],";
            tx->serialize(out);
            // file offset and size
            out << ",[\"" << ggml_type_name(tx->tensor->type);
            out << "\",[" << tx->tensor->ne[0];
            out << "," << tx->tensor->ne[1];
            out << "," << tx->tensor->ne[2];
            out << "," << tx->tensor->ne[3] << "],";
            // TODO: remove!
            if (tx->op_name() == "top_k") {
                printf("found it\n");
            }
            if ( !tx->tensor->data || tx->side_effect ) {
                out << "0,0]";
            //} else if (tx->tensor->view_src) {
                // TODO: maybe add support for tensors that use views
            //    out << "0,0]";
            } else if (tx->tensor->view_src && tx->tensor->op == GGML_OP_ADD) {
                // skipping inplace
                out << "0,0]";
            } else {
                write_bytes( tx->tensor );
            }
            out << ",\"" << tx->name << '"';
            if (tx->group)
                out << ",\"" << tx->group->id() << '"';
            else
                out << ",\"0\"";
            out << ",\"" << tx->caller << '"';
            out << ']';
        }
        fclose( fbin );
        fbin = NULL;
        out << "},\n\"groups\":{\n";
        tail = "\"";
        for (auto group : groups) {
            out << tail << group->id() << "\":[";
            tail = ",\n\"";
            out << '"' << group->name << '"';
            // parent
            if (group->parent)
                out << ",\"" << group->parent->id() << "\",[";
            else
                out << ",\"0\",[";
            // tensors
            const char * tail2 = "\"";
            for (auto tx : group->tensors) {
                if ( tensors.find( tx ) == tensors.end()
                 && new_tensors.find( tx ) == new_tensors.end() )
                    continue;
                out << tail2 << tx->id() << '"';
                tail2 = ",\"";
            }
            // children
            out << "],[";
            tail2 = "\"";
            for (auto child : group->children) {
                if (groups.find( child ) == groups.end())
                    continue;
                out << tail2 << child->id() << '"';
                tail2 = ",\"";
            }
            out << "]]";
        }
        out << "},\n\"forward_expand\":[\n";
        tail = "\"";
        for (auto tx : gx->tensors) {
            out << tail << tx->id() << '"';
            tail = ",\n\"";
        }
        out << "],\n\"nbytes\":" << total_nbytes << "}";
        auto json = out.str();
        auto f = fopen(("capture/" + gx->dump_filename + ".json").c_str(), "wb");
        assert( f );
        auto w = fwrite( json.c_str(), json.size(), 1, f );
        assert( w == 1 );
        fclose( f );
        gx->dump_filename = "";
    }
};

void graph_set_dump( std::string filename, ggml_cgraph * cgraph ) {
    auto gx = GGMLCapture::Get().get( cgraph );
    assert( gx );
    gx->dump_filename = filename;
}
#define CAPTURE(filename, cgraph) graph_set_dump(filename, cgraph)


enum ggml_status _ggml_backend_graph_compute (
    ggml_backend_t backend, struct ggml_cgraph * cgraph ) {
    auto gx = GGMLCapture::Get().get( cgraph );
    assert( gx );
    if (!gx->dump_filename.size())
        return ggml_backend_graph_compute( backend, cgraph );

    GraphDumper dumper( gx );
    auto status = ggml_backend_graph_compute( backend, cgraph );
    dumper.post_compute();
    return status;
}
#define ggml_backend_graph_compute(...) _ggml_backend_graph_compute(__VA_ARGS__)

enum ggml_status _ggml_graph_compute_with_ctx( struct ggml_context * ctx,
        struct ggml_cgraph * cgraph, int n_threads ) {
    auto gx = GGMLCapture::Get().get( cgraph );
    assert( gx );
    if (!gx->dump_filename.size())
        return ggml_graph_compute_with_ctx( ctx, cgraph, n_threads );

    GraphDumper dumper( gx );
    auto status = ggml_graph_compute_with_ctx( ctx, cgraph, n_threads );
    dumper.post_compute();
    gx->dump_filename = "";
    return status;
}
#define ggml_graph_compute_with_ctx(...) _ggml_graph_compute_with_ctx(__VA_ARGS__)



struct ggml_context * _ggml_init (struct ggml_init_params params) {
    auto context = ggml_init( params );
    GGMLCapture::Get().create( context );
    return context;
}
#define ggml_init(...) _ggml_init((ggml_init_params)__VA_ARGS__)

void                  _ggml_reset(struct ggml_context * ctx) {
    GGMLCapture::Get().reset( ctx );
    ggml_reset( ctx );
}
#define ggml_reset(ctx) _ggml_reset(ctx)

void                  _ggml_free (struct ggml_context * ctx) {
    GGMLCapture::Get().free( ctx );
    ggml_free( ctx );
}
#define ggml_free(ctx) _ggml_free(ctx)

struct ggml_cgraph * _ggml_new_graph       (struct ggml_context * ctx) {
    auto cgraph = ggml_new_graph( ctx );
    GGMLCapture::Get().create( ctx, cgraph );
    return cgraph;
}
#define ggml_new_graph(ctx) _ggml_new_graph(ctx)

struct ggml_cgraph * _ggml_new_graph_custom(
        struct ggml_context * ctx, size_t size, bool grads ) {
    auto cgraph = ggml_new_graph_custom( ctx, size, grads );
    GGMLCapture::Get().create( ctx, cgraph );
    return cgraph;
}
#define ggml_new_graph_custom(...) _ggml_new_graph_custom(__VA_ARGS__)



const char * get_filename( const char * text ) {
    int last = 0;
    for ( int i = 0; text[i]; i++ ) {
        if ( text[i] == '/' || text[i] == '\\' )
            last = i + 1;
    }
    return text + last;
}

/*bool visited( ggml_tensor * tensor ){
    if (tensor->name[GGML_MAX_NAME-1] == 1)
        return true;
    assert( tensor->name[GGML_MAX_NAME-1] == 0 );
    ((ggml_tensor*)tensor)->name[GGML_MAX_NAME-2] = 0;
    ((ggml_tensor*)tensor)->name[GGML_MAX_NAME-1] = 1;
    return false;
}*/
#define visited(...) {}

/*void check( const ggml_tensor * tensor ) {
    if (!tensor)
        return;
    assert( tensor->name[0] == '@' );
    if (visited( (ggml_tensor*)tensor ))
        return;
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (tensor->src[i] && tensor->src[i] != tensor)
            check( tensor->src[i] );
    }
}*/
#define check(...) {}

struct ggml_tensor * _ggml_new_tensor(
        struct ggml_context * ctx,
        enum   ggml_type type,
        int    n_dims,
        const int64_t *ne,
        const char * file, int line, const char * func ) {
    //auto tensor = ggml_new_tensor( ctx, type, n_dims, ne );
    auto tx = GGMLCapture::Get().create_new_tensor( ctx, type, n_dims, ne );
    auto tensor = tx->get();
    snprintf( tensor->name, GGML_MAX_NAME, "%s:%d:%s", get_filename(file), line, func);
    tx->caller = tensor->name;
    return tensor;
}
#define ggml_new_tensor(...) _ggml_new_tensor(__VA_ARGS__, __FILE__, __LINE__, __func__)

struct ggml_tensor * _ggml_new_tensor_1d(
        struct ggml_context * ctx,
        enum   ggml_type type,
        int64_t ne0,
        const char * file, int line, const char * func ) {
    //auto tensor = ggml_new_tensor_1d( ctx, type, ne0 );
    auto tx = GGMLCapture::Get().create_new_tensor( ctx, type, 1, &ne0 );
    auto tensor = tx->get();
    snprintf( tensor->name, GGML_MAX_NAME, "%s:%d:%s", get_filename(file), line, func);
    tx->caller = tensor->name;
    return tensor;
}
#define ggml_new_tensor_1d(...) _ggml_new_tensor_1d(__VA_ARGS__, __FILE__, __LINE__, __func__)

struct ggml_tensor * _ggml_new_tensor_2d(
        struct ggml_context * ctx,
        enum   ggml_type type,
        int64_t ne0,
        int64_t ne1,
        const char * file, int line, const char * func ) {
    //auto tensor = ggml_new_tensor_2d( ctx, type, ne0, ne1 );
    int64_t ne[] = {ne0, ne1};
    auto tx = GGMLCapture::Get().create_new_tensor( ctx, type, 2, ne );
    auto tensor = tx->get();
    snprintf( tensor->name, GGML_MAX_NAME, "%s:%d:%s", get_filename(file), line, func);
    tx->caller = tensor->name;
    return tensor;
}
#define ggml_new_tensor_2d(...) _ggml_new_tensor_2d(__VA_ARGS__, __FILE__, __LINE__, __func__)

struct ggml_tensor * _ggml_new_tensor_3d(
        struct ggml_context * ctx,
        enum   ggml_type type,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2,
        const char * file, int line, const char * func ) {
    //auto tensor = ggml_new_tensor_3d( ctx, type, ne0, ne1, ne2 );
    int64_t ne[] = {ne0, ne1, ne2};
    auto tx = GGMLCapture::Get().create_new_tensor( ctx, type, 3, ne );
    auto tensor = tx->get();
    snprintf( tensor->name, GGML_MAX_NAME, "%s:%d:%s", get_filename(file), line, func);
    tx->caller = tensor->name;
    return tensor;
}
#define ggml_new_tensor_3d(...) _ggml_new_tensor_3d(__VA_ARGS__, __FILE__, __LINE__, __func__)

struct ggml_tensor * _ggml_new_tensor_4d(
        struct ggml_context * ctx,
        enum   ggml_type type,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2,
        int64_t ne3,
        const char * file, int line, const char * func ) {
    //auto tensor = ggml_new_tensor_4d( ctx, type, ne0, ne1, ne2, ne3 );
    int64_t ne[] = {ne0, ne1, ne2, ne3};
    auto tx = GGMLCapture::Get().create_new_tensor( ctx, type, 4, ne );
    auto tensor = tx->get();
    snprintf( tensor->name, GGML_MAX_NAME, "%s:%d:%s", get_filename(file), line, func);
    tx->caller = tensor->name;
    return tensor;
}
#define ggml_new_tensor_4d(...) _ggml_new_tensor_4d(__VA_ARGS__, __FILE__, __LINE__, __func__)

struct ggml_tensor * _ggml_dup_tensor (
        struct ggml_context * ctx,
        const struct ggml_tensor * src,
        const char * file, int line, const char * func ) {
    //check( src );
    //auto tensor = ggml_dup_tensor( ctx, src );
    //snprintf( tensor->name, GGML_MAX_NAME, "@%s %d", file, line);
    //return tensor;
    auto tx = GGMLCapture::Get().create_new_tensor( ctx, src->type, 4, src->ne );
    auto tensor = tx->get();
    snprintf( tensor->name, GGML_MAX_NAME, "%s:%d:%s", get_filename(file), line, func);
    tx->caller = tensor->name;
    return tensor;
}
#define ggml_dup_tensor(...)  _ggml_dup_tensor (__VA_ARGS__, __FILE__, __LINE__, __func__)


#define GGML_UNO_CALL(op) \
struct ggml_tensor * _ggml_##op(\
        struct ggml_context * ctx,\
        struct ggml_tensor  * a,\
        const char * file, int line, const char * func ) {\
    check( a );\
    auto tx = GGMLCapture::Get().create_##op( ctx, a );\
    auto tensor = tx->get();\
    snprintf( tensor->name, GGML_MAX_NAME, "%s:%d:%s", get_filename(file), line, func);\
    tx->caller = tensor->name;\
    return tensor;\
}

#define GGML_PAIR_CALL(op) \
struct ggml_tensor * _ggml_##op(\
        struct ggml_context * ctx,\
        struct ggml_tensor  * a,\
        struct ggml_tensor  * b,\
        const char * file, int line, const char * func ) {\
    check( a );\
    check( b );\
    auto tx = GGMLCapture::Get().create_##op( ctx, a, b );\
    auto tensor = tx->get();\
    snprintf( tensor->name, GGML_MAX_NAME, "%s:%d:%s", get_filename(file), line, func);\
    tx->caller = tensor->name;\
    return tensor;\
}

#define GGML_PAIR_INPLACE_CALL(op) \
struct ggml_tensor * _ggml_##op(\
        struct ggml_context * ctx,\
        struct ggml_tensor  * a,\
        struct ggml_tensor  * b,\
        const char * file, int line, const char * func ) {\
    check( a );\
    check( b );\
    auto tx = GGMLCapture::Get().create_##op( ctx, a, b );\
    auto tensor = tx->get();\
    snprintf( tensor->name, GGML_MAX_NAME, "%s:%d:%s", get_filename(file), line, func);\
    tx->caller = tensor->name;\
    ggml_set_output(tensor);\
    return tensor;\
}\
struct ggml_tensor * _ggml_##op##_inplace(\
        struct ggml_context * ctx,\
        struct ggml_tensor  * a,\
        struct ggml_tensor  * b,\
        const char * file, int line, const char * func ) {\
    check( a );\
    check( b );\
    auto tx = GGMLCapture::Get().create_##op( ctx, a, b, true );\
    auto tensor = tx->get();\
    snprintf( tensor->name, GGML_MAX_NAME, "%s:%d:%s", get_filename(file), line, func);\
    tx->caller = tensor->name;\
    return tensor;\
}

GGML_PAIR_INPLACE_CALL(add);
#define ggml_add(...) _ggml_add(__VA_ARGS__, __FILE__, __LINE__,__func__)
#define ggml_add_inplace(...) _ggml_add_inplace(__VA_ARGS__, __FILE__, __LINE__,__func__)

GGML_PAIR_INPLACE_CALL(sub);
#define ggml_sub(...) _ggml_sub(__VA_ARGS__, __FILE__, __LINE__,__func__)
#define ggml_sub_inplace(...) _ggml_sub_inplace(__VA_ARGS__, __FILE__, __LINE__,__func__)

GGML_PAIR_INPLACE_CALL(mul);
#define ggml_mul(...) _ggml_mul(__VA_ARGS__, __FILE__, __LINE__,__func__)
#define ggml_mul_inplace(...) _ggml_mul_inplace(__VA_ARGS__, __FILE__, __LINE__,__func__)

GGML_PAIR_INPLACE_CALL(div);
#define ggml_div(...) _ggml_div(__VA_ARGS__, __FILE__, __LINE__,__func__)
#define ggml_div_inplace(...) _ggml_div_inplace(__VA_ARGS__, __FILE__, __LINE__,__func__)

GGML_UNO_CALL(neg);
#define ggml_neg(...) _ggml_neg(__VA_ARGS__, __FILE__, __LINE__,__func__)
GGML_UNO_CALL(sum);
#define ggml_sum(...) _ggml_sum(__VA_ARGS__, __FILE__, __LINE__,__func__)
// ggml_repeat_4d custom
// ggml_concat custom
GGML_UNO_CALL(elu);
#define ggml_elu(...) _ggml_elu(__VA_ARGS__, __FILE__, __LINE__,__func__)
GGML_UNO_CALL(gelu);
#define ggml_gelu(...) _ggml_gelu(__VA_ARGS__, __FILE__, __LINE__,__func__)
GGML_UNO_CALL(silu);
#define ggml_silu(...) _ggml_silu(__VA_ARGS__, __FILE__, __LINE__,__func__)
// ggml_norm custom
// ggml_rms_norm custom
GGML_PAIR_CALL(mul_mat);
#define ggml_mul_mat(...) _ggml_mul_mat(__VA_ARGS__, __FILE__, __LINE__,__func__)
GGML_UNO_CALL(argmax);
#define ggml_argmax(...) _ggml_argmax(__VA_ARGS__, __FILE__, __LINE__,__func__)
// ggml_scale custom
GGML_PAIR_CALL(cpy);
#define ggml_cpy(...) _ggml_cpy(__VA_ARGS__, __FILE__, __LINE__,__func__)
// ggml_cast custom
GGML_UNO_CALL(cont);
#define ggml_cont(...) _ggml_cont(__VA_ARGS__, __FILE__, __LINE__,__func__)
// ggml_reshape_2d custom
// ggml_reshape_3d custom
// ggml_reshape_4d custom
// ggml_view_1d custom
// ggml_view_2d custom
// ggml_view_3d custom
// ggml_view_4d custom
// ggml_permute custom
GGML_UNO_CALL(transpose);
#define ggml_transpose(...) _ggml_transpose(__VA_ARGS__, __FILE__, __LINE__,__func__)
GGML_UNO_CALL(soft_max);
#define ggml_soft_max(...) _ggml_soft_max(__VA_ARGS__, __FILE__, __LINE__,__func__)
// ggml_soft_max_ext custom
GGML_PAIR_CALL(get_rows);
#define ggml_get_rows(...) _ggml_get_rows(__VA_ARGS__, __FILE__, __LINE__,__func__)
// ggml_clamp custom
// ggml_conv_1d custom
// ggml_conv_transpose_1d custom
// ggml_arange custom
// ggml_top_k custom
// ggml_timestep_embedding custom

/*struct ggml_tensor * _ggml_add(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b,
        const char * file, int line ) {
    check( a );
    check( b );
    //auto tensor = ggml_add( ctx, a, b );
    auto tensor = GGMLCapture::Get().create_add( ctx, a, b );
    snprintf( tensor->name, GGML_MAX_NAME, "@%s %d", file, line);
    return tensor;
}
#define ggml_add(...) _ggml_add(__VA_ARGS__, __FILE__, __LINE__)

struct ggml_tensor * _ggml_add_inplace(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b,
        const char * file, int line ) {
    check( a );
    check( b );
    //auto tensor = ggml_add_inplace( ctx, a, b );
    auto tensor = GGMLCapture::Get().create_add( ctx, a, b, true );
    snprintf( tensor->name, GGML_MAX_NAME, "@%s %d", file, line);
    return tensor;
}
#define ggml_add_inplace(...) _ggml_add_inplace(__VA_ARGS__, __FILE__, __LINE__)

struct ggml_tensor * _ggml_sub(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b,
        const char * file, int line ) {
    check( a );
    check( b );
    //auto tensor = ggml_sub( ctx, a, b );
    auto tensor = GGMLCapture::Get().create_sub( ctx, a, b );
    snprintf( tensor->name, GGML_MAX_NAME, "@%s %d", file, line);
    return tensor;
}
#define ggml_sub(...) _ggml_sub(__VA_ARGS__, __FILE__, __LINE__)

struct ggml_tensor * _ggml_mul(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b,
        const char * file, int line ) {
    check( a );
    check( b );
    //auto tensor = ggml_mul( ctx, a, b );
    auto tensor = GGMLCapture::Get().create_mul( ctx, a, b );
    snprintf( tensor->name, GGML_MAX_NAME, "@%s %d", file, line);
    return tensor;
}
#define ggml_mul(...) _ggml_mul(__VA_ARGS__, __FILE__, __LINE__)

struct ggml_tensor * _ggml_div(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b,
        const char * file, int line ) {
    check( a );
    check( b );
    //auto tensor = ggml_div( ctx, a, b );
    auto tensor = GGMLCapture::Get().create_div( ctx, a, b );
    snprintf( tensor->name, GGML_MAX_NAME, "@%s %d", file, line);
    return tensor;
}
#define ggml_div(...) _ggml_div(__VA_ARGS__, __FILE__, __LINE__)

struct ggml_tensor * _ggml_neg(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        const char * file, int line ) {
    check( a );
    //auto tensor = ggml_neg( ctx, a );
    auto tensor = GGMLCapture::Get().create_neg( ctx, a );
    snprintf( tensor->name, GGML_MAX_NAME, "@%s %d", file, line);
    return tensor;
}
#define ggml_neg(...) _ggml_neg(__VA_ARGS__, __FILE__, __LINE__)

struct ggml_tensor * _ggml_sum(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        const char * file, int line ) {
    check( a );
    //auto tensor = ggml_sum( ctx, a );
    auto tensor = GGMLCapture::Get().create_sum( ctx, a );
    snprintf( tensor->name, GGML_MAX_NAME, "@%s %d", file, line);
    return tensor;
}
#define ggml_sum(...) _ggml_sum(__VA_ARGS__, __FILE__, __LINE__)
*/

struct ggml_tensor * _ggml_repeat_4d(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        int64_t    ne0,
        int64_t    ne1,
        int64_t    ne2,
        int64_t    ne3,
        const char * file, int line, const char * func ) {
    check( a );
    //auto tensor = ggml_repeat_4d( ctx, a, ne0, ne1, ne2, ne3 );
    int64_t ne[4] = {ne0, ne1, ne2, ne3};
    auto tx = GGMLCapture::Get().create_repeat_4d( ctx, a, ne );
    auto tensor = tx->get();
    snprintf( tensor->name, GGML_MAX_NAME, "%s:%d:%s", get_filename(file), line, func);
    tx->caller = tensor->name;
    ggml_set_output(tensor);
    return tensor;
}
#define ggml_repeat_4d(...) _ggml_repeat_4d(__VA_ARGS__, __FILE__, __LINE__,__func__)

struct ggml_tensor * _ggml_concat(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b,
        int                   dim,
        const char * file, int line, const char * func ) {
    check( a );
    check( b );
    //auto tensor = ggml_concat( ctx, a, b, dim );
    auto tx = GGMLCapture::Get().create_concat( ctx, a, b, dim );
    auto tensor = tx->get();
    snprintf( tensor->name, GGML_MAX_NAME, "%s:%d:%s", get_filename(file), line, func);
    tx->caller = tensor->name;
    return tensor;
}
#define ggml_concat(...) _ggml_concat(__VA_ARGS__, __FILE__, __LINE__,__func__)

/*
struct ggml_tensor * _ggml_elu(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        const char * file, int line ) {
    check( a );
    //auto tensor = ggml_elu( ctx, a );
    auto tensor = GGMLCapture::Get().create_elu( ctx, a );
    snprintf( tensor->name, GGML_MAX_NAME, "@%s %d", file, line);
    return tensor;
}
#define ggml_elu(...) _ggml_elu(__VA_ARGS__, __FILE__, __LINE__)

struct ggml_tensor * _ggml_gelu(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        const char * file, int line ) {
    check( a );
    //auto tensor = ggml_gelu( ctx, a );
    auto tensor = GGMLCapture::Get().create_gelu( ctx, a );
    snprintf( tensor->name, GGML_MAX_NAME, "@%s %d", file, line);
    return tensor;
}
#define ggml_gelu(...) _ggml_gelu(__VA_ARGS__, __FILE__, __LINE__)

struct ggml_tensor * _ggml_silu(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        const char * file, int line ) {
    check( a );
    //auto tensor = ggml_silu( ctx, a );
    auto tensor = GGMLCapture::Get().create_silu( ctx, a );
    snprintf( tensor->name, GGML_MAX_NAME, "@%s %d", file, line);
    return tensor;
}
#define ggml_silu(...) _ggml_silu(__VA_ARGS__, __FILE__, __LINE__)
*/

struct ggml_tensor * _ggml_norm(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        float                 eps,
        const char * file, int line, const char * func ) {
    check( a );
    //auto tensor = ggml_norm( ctx, a, eps );
    auto tx = GGMLCapture::Get().create_norm( ctx, a, eps );
    auto tensor = tx->get();
    snprintf( tensor->name, GGML_MAX_NAME, "%s:%d:%s", get_filename(file), line, func);
    tx->caller = tensor->name;
    return tensor;
}
#define ggml_norm(...) _ggml_norm(__VA_ARGS__, __FILE__, __LINE__,__func__)

struct ggml_tensor * _ggml_rms_norm(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        float                 eps,
        const char * file, int line, const char * func ) {
    check( a );
    //auto tensor = ggml_rms_norm( ctx, a, eps );
    auto tx = GGMLCapture::Get().create_rms_norm( ctx, a, eps );
    auto tensor = tx->get();
    snprintf( tensor->name, GGML_MAX_NAME, "%s:%d:%s", get_filename(file), line, func);
    tx->caller = tensor->name;
    return tensor;
}
#define ggml_rms_norm(...) _ggml_rms_norm(__VA_ARGS__, __FILE__, __LINE__,__func__)

/*
struct ggml_tensor * _ggml_mul_mat(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b,
        const char * file, int line ) {
    check( a );
    check( b );
    //auto tensor = ggml_mul_mat( ctx, a, b );
    auto tensor = GGMLCapture::Get().create_mul_mat( ctx, a, b );
    snprintf( tensor->name, GGML_MAX_NAME, "@%s %d", file, line);
    return tensor;
}
#define ggml_mul_mat(...) _ggml_mul_mat(__VA_ARGS__, __FILE__, __LINE__)

struct ggml_tensor * _ggml_argmax(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        const char * file, int line ) {
    check( a );
    //auto tensor = ggml_argmax( ctx, a );
    auto tensor = GGMLCapture::Get().create_argmax( ctx, a );
    snprintf( tensor->name, GGML_MAX_NAME, "@%s %d", file, line);
    return tensor;
}
#define ggml_argmax(...) _ggml_argmax(__VA_ARGS__, __FILE__, __LINE__)
*/

struct ggml_tensor * _ggml_scale(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        float                 s,
        const char * file, int line, const char * func ) {
    check( a );
    //auto tensor = ggml_scale( ctx, a, s );
    auto tx = GGMLCapture::Get().create_scale( ctx, a, s );
    auto tensor = tx->get();
    snprintf( tensor->name, GGML_MAX_NAME, "%s:%d:%s", get_filename(file), line, func);
    tx->caller = tensor->name;
    return tensor;
}
#define ggml_scale(...) _ggml_scale(__VA_ARGS__, __FILE__, __LINE__,__func__)

/*struct ggml_tensor * _ggml_cpy(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b,
        const char * file, int line ) {
    check( a );
    check( b );
    //auto tensor = ggml_cpy( ctx, a, b );
    auto tensor = GGMLCapture::Get().create_cpy( ctx, a, b );
    snprintf( tensor->name, GGML_MAX_NAME, "@%s %d", file, line);
    return tensor;
}
#define ggml_cpy(...) _ggml_cpy(__VA_ARGS__, __FILE__, __LINE__)*/

struct ggml_tensor * _ggml_cast(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        enum   ggml_type      type,
        const char * file, int line, const char * func ) {
    check( a );
    //auto tensor = ggml_cast( ctx, a, type );
    auto tx = GGMLCapture::Get().create_cast( ctx, a, type );
    auto tensor = tx->get();
    snprintf( tensor->name, GGML_MAX_NAME, "%s:%d:%s", get_filename(file), line, func);
    tx->caller = tensor->name;
    return tensor;
}
#define ggml_cast(...) _ggml_cast(__VA_ARGS__, __FILE__, __LINE__,__func__)

/*
struct ggml_tensor * _ggml_cont(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        const char * file, int line ) {
    check( a );
    //auto tensor = ggml_cont( ctx, a );
    auto tensor = GGMLCapture::Get().create_cont( ctx, a );
    snprintf( tensor->name, GGML_MAX_NAME, "@%s %d", file, line);
    return tensor;
}
#define ggml_cont(...) _ggml_cont(__VA_ARGS__, __FILE__, __LINE__)
*/

struct ggml_tensor * _ggml_reshape_2d(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        const char * file, int line, const char * func ) {
    check( a );
    //auto tensor = ggml_reshape_2d( ctx, a, ne0, ne1 );
    int64_t ne[] = {ne0, ne1};
    auto tx = GGMLCapture::Get().create_reshape( ctx, a, 2, ne );
    auto tensor = tx->get();
    snprintf( tensor->name, GGML_MAX_NAME, "%s:%d:%s", get_filename(file), line, func);
    tx->caller = tensor->name;
    return tensor;
}
#define ggml_reshape_2d(...) _ggml_reshape_2d(__VA_ARGS__, __FILE__, __LINE__,__func__)

struct ggml_tensor * _ggml_reshape_3d(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        const char * file, int line, const char * func ) {
    check( a );
    //auto tensor = ggml_reshape_3d( ctx, a, ne0, ne1, ne2 );
    int64_t ne[] = {ne0, ne1, ne2};
    auto tx = GGMLCapture::Get().create_reshape( ctx, a, 3, ne );
    auto tensor = tx->get();
    snprintf( tensor->name, GGML_MAX_NAME, "%s:%d:%s", get_filename(file), line, func);
    tx->caller = tensor->name;
    return tensor;
}
#define ggml_reshape_3d(...) _ggml_reshape_3d(__VA_ARGS__, __FILE__, __LINE__,__func__)

struct ggml_tensor * _ggml_reshape_4d(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3,
        const char * file, int line, const char * func ) {
    check( a );
    //auto tensor = ggml_reshape_4d( ctx, a, ne0, ne1, ne2, ne3 );
    int64_t ne[] = {ne0, ne1, ne2, ne3};
    auto tx = GGMLCapture::Get().create_reshape( ctx, a, 4, ne );
    auto tensor = tx->get();
    snprintf( tensor->name, GGML_MAX_NAME, "%s:%d:%s", get_filename(file), line, func);
    tx->caller = tensor->name;
    return tensor;
}
#define ggml_reshape_4d(...) _ggml_reshape_4d(__VA_ARGS__, __FILE__, __LINE__,__func__)

struct ggml_tensor * _ggml_view_1d(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        int64_t               ne0,
        size_t                offset,
        const char * file, int line, const char * func ) {
    check( a );
    //auto tensor = ggml_view_1d( ctx, a, ne0, offset );
    auto tx = GGMLCapture::Get().create_view( ctx, a, 1, &ne0, NULL, offset );
    auto tensor = tx->get();
    snprintf( tensor->name, GGML_MAX_NAME, "%s:%d:%s", get_filename(file), line, func);
    tx->caller = tensor->name;
    return tensor;
}
#define ggml_view_1d(...) _ggml_view_1d(__VA_ARGS__, __FILE__, __LINE__,__func__)

struct ggml_tensor * _ggml_view_2d(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        size_t                nb1, // row stride in bytes
        size_t                offset,
        const char * file, int line, const char * func ) {
    check( a );
    //auto tensor = ggml_view_2d( ctx, a, ne0, ne1, nb1, offset );
    int64_t ne[] = {ne0, ne1};
    size_t stride[] = {nb1};
    auto tx = GGMLCapture::Get().create_view( ctx, a, 2, ne, stride, offset );
    auto tensor = tx->get();
    snprintf( tensor->name, GGML_MAX_NAME, "%s:%d:%s", get_filename(file), line, func);
    tx->caller = tensor->name;
    return tensor;
}
#define ggml_view_2d(...) _ggml_view_2d(__VA_ARGS__, __FILE__, __LINE__,__func__)

struct ggml_tensor * _ggml_view_3d(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        size_t                nb1, // row   stride in bytes
        size_t                nb2, // slice stride in bytes
        size_t                offset,
        const char * file, int line, const char * func ) {
    check( a );
    //auto tensor = ggml_view_3d( ctx, a, ne0, ne1, ne2, nb1, nb2, offset );
    int64_t ne[] = {ne0, ne1, ne2};
    size_t stride[] = {nb1, nb2};
    auto tx = GGMLCapture::Get().create_view( ctx, a, 3, ne, stride, offset );
    auto tensor = tx->get();
    snprintf( tensor->name, GGML_MAX_NAME, "%s:%d:%s", get_filename(file), line, func);
    tx->caller = tensor->name;
    return tensor;
}
#define ggml_view_3d(...) _ggml_view_3d(__VA_ARGS__, __FILE__, __LINE__,__func__)

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
        const char * file, int line, const char * func ) {
    check( a );
    //auto tensor = ggml_view_4d( ctx, a, ne0, ne1, ne2, ne3, nb1, nb2, nb3, offset );
    int64_t ne[] = {ne0, ne1, ne2, ne3};
    size_t stride[] = {nb1, nb2, nb3};
    auto tx = GGMLCapture::Get().create_view( ctx, a, 4, ne, stride, offset );
    auto tensor = tx->get();
    snprintf( tensor->name, GGML_MAX_NAME, "%s:%d:%s", get_filename(file), line, func);
    tx->caller = tensor->name;
    return tensor;
}
#define ggml_view_4d(...) _ggml_view_4d(__VA_ARGS__, __FILE__, __LINE__,__func__)

struct ggml_tensor * _ggml_permute(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        int                   axis0,
        int                   axis1,
        int                   axis2,
        int                   axis3,
        const char * file, int line, const char * func ) {
    check( a );
    //auto tensor = ggml_permute( ctx, a, axis0, axis1, axis2, axis3 );
    int axis[] = {axis0, axis1, axis2, axis3};
    auto tx = GGMLCapture::Get().create_permute( ctx, a, axis );
    auto tensor = tx->get();
    snprintf( tensor->name, GGML_MAX_NAME, "%s:%d:%s", get_filename(file), line, func);
    tx->caller = tensor->name;
    return tensor;
}
#define ggml_permute(...) _ggml_permute(__VA_ARGS__, __FILE__, __LINE__,__func__)

/*
struct ggml_tensor * _ggml_transpose(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        const char * file, int line ) {
    check( a );
    //auto tensor = ggml_transpose( ctx, a );
    auto tensor = GGMLCapture::Get().create_transpose( ctx, a );
    snprintf( tensor->name, GGML_MAX_NAME, "@%s %d", file, line);
    return tensor;
}
#define ggml_transpose(...) _ggml_transpose(__VA_ARGS__, __FILE__, __LINE__)

struct ggml_tensor * _ggml_soft_max(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        const char * file, int line ) {
    check( a );
    //auto tensor = ggml_soft_max( ctx, a );
    auto tensor = GGMLCapture::Get().create_soft_max( ctx, a );
    snprintf( tensor->name, GGML_MAX_NAME, "@%s %d", file, line);
    return tensor;
}
#define ggml_soft_max(...) _ggml_soft_max(__VA_ARGS__, __FILE__, __LINE__)
*/

struct ggml_tensor * _ggml_soft_max_ext(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * mask,
        float                 scale,
        float                 max_bias,
        const char * file, int line, const char * func ) {
    check( a );
    check( mask );
    //auto tensor = ggml_soft_max_ext( ctx, a, mask, scale, max_bias );
    auto tx = GGMLCapture::Get().create_soft_max_ext( ctx, a, mask, scale, max_bias );
    auto tensor = tx->get();
    snprintf( tensor->name, GGML_MAX_NAME, "%s:%d:%s", get_filename(file), line, func);
    tx->caller = tensor->name;
    return tensor;
}
#define ggml_soft_max_ext(...) _ggml_soft_max_ext(__VA_ARGS__, __FILE__, __LINE__,__func__)

/*
struct ggml_tensor * _ggml_get_rows(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,  // data
        struct ggml_tensor  * b,
        const char * file, int line ) {
    check( a );
    check( b );
    //auto tensor = ggml_get_rows( ctx, a, b );
    auto tensor = GGMLCapture::Get().create_get_rows( ctx, a, b );
    snprintf( tensor->name, GGML_MAX_NAME, "@%s %d", file, line);
    return tensor;
}
#define ggml_get_rows(...) _ggml_get_rows(__VA_ARGS__, __FILE__, __LINE__)
*/

struct ggml_tensor * _ggml_clamp(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        float                 min,
        float                 max,
        const char * file, int line, const char * func ) {
    check( a );
    //auto tensor = ggml_clamp( ctx, a, min, max );
    auto tx = GGMLCapture::Get().create_clamp( ctx, a, min, max );
    auto tensor = tx->get();
    snprintf( tensor->name, GGML_MAX_NAME, "%s:%d:%s", get_filename(file), line, func);
    tx->caller = tensor->name;
    return tensor;
}
#define ggml_clamp(...) _ggml_clamp(__VA_ARGS__, __FILE__, __LINE__,__func__)

struct ggml_tensor * _ggml_conv_1d(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,   // convolution kernel
        struct ggml_tensor  * b,   // data
        int                   s0,  // stride
        int                   p0,  // padding
        int                   d0,
        const char * file, int line, const char * func ) {
    check( a );
    check( b );
    //auto tensor = ggml_conv_1d( ctx, a, b, s0, p0, d0 );
    auto tx = GGMLCapture::Get().create_conv_1d( ctx, a, b, s0, p0, d0 );
    auto tensor = tx->get();
    snprintf( tensor->name, GGML_MAX_NAME, "%s:%d:%s", get_filename(file), line, func);
    visited( tensor ); // produces other tensors
    tx->caller = tensor->name;
    return tensor;
}
#define ggml_conv_1d(...) _ggml_conv_1d(__VA_ARGS__, __FILE__, __LINE__,__func__)

struct ggml_tensor * _ggml_conv_transpose_1d(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,   // convolution kernel
        struct ggml_tensor  * b,   // data
        int                   s0,  // stride
        int                   p0,  // padding
        int                   d0,
        const char * file, int line, const char * func ) {
    check( a );
    check( b );
    //auto tensor = ggml_conv_transpose_1d( ctx, a, b, s0, p0, d0 );
    auto tx = GGMLCapture::Get().create_conv_transpose_1d( ctx, a, b, s0, p0, d0 );
    auto tensor = tx->get();
    snprintf( tensor->name, GGML_MAX_NAME, "%s:%d:%s", get_filename(file), line, func);
    tx->caller = tensor->name;
    return tensor;
}
#define ggml_conv_transpose_1d(...) _ggml_conv_transpose_1d(__VA_ARGS__, __FILE__, __LINE__,__func__)

struct ggml_tensor * _ggml_arange(
        struct ggml_context * ctx,
        float                 start,
        float                 stop,
        float                 step,
        const char * file, int line, const char * func ) {
    //auto tensor = ggml_arange( ctx, start, stop, step );
    auto tx = GGMLCapture::Get().create_arange( ctx, start, stop, step );
    auto tensor = tx->get();
    snprintf( tensor->name, GGML_MAX_NAME, "%s:%d:%s", get_filename(file), line, func);
    tx->caller = tensor->name;
    return tensor;
}
#define ggml_arange(...) _ggml_arange(__VA_ARGS__, __FILE__, __LINE__,__func__)

struct ggml_tensor * _ggml_top_k(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        int                   k,
        const char * file, int line, const char * func ) {
    check( a );
    //auto tensor = ggml_top_k( ctx, a, k );
    auto tx = GGMLCapture::Get().create_top_k( ctx, a, k );
    auto tensor = tx->get();
    snprintf( tensor->name, GGML_MAX_NAME, "%s:%d:%s", get_filename(file), line, func);
    visited( tensor ); // produces other tensors
    tx->caller = tensor->name;
    return tensor;
}
#define ggml_top_k(...) _ggml_top_k(__VA_ARGS__, __FILE__, __LINE__,__func__)

struct ggml_tensor * _ggml_timestep_embedding(
        struct ggml_context * ctx,
        struct ggml_tensor  * timesteps,
        int                   dim,
        int                   max_period,
        const char * file, int line, const char * func ) {
    check( timesteps );
    //auto tensor = ggml_timestep_embedding( ctx, timesteps, dim, max_period );
    auto tx = GGMLCapture::Get().create_timestep_embedding( ctx, timesteps, dim, max_period );
    auto tensor = tx->get();
    snprintf( tensor->name, GGML_MAX_NAME, "%s:%d:%s", get_filename(file), line, func);
    tx->caller = tensor->name;
    return tensor;
}
#define ggml_timestep_embedding(...) _ggml_timestep_embedding(__VA_ARGS__, __FILE__, __LINE__,__func__)

struct ggml_tensor * _ggml_sum_rows( ggml_context * ctx, ggml_tensor * a,
        const char * file, int line, const char * func ) {
    check( a );
    auto tx = GGMLCapture::Get().create_sum_rows( ctx, a );
    auto tensor = tx->get();
    snprintf( tensor->name, GGML_MAX_NAME, "%s:%d:%s", get_filename(file), line, func);
    tx->caller = tensor->name;
    return tensor;
}
#define ggml_sum_rows(...) _ggml_sum_rows(__VA_ARGS__, __FILE__, __LINE__,__func__)

void _ggml_build_forward_expand(
        struct ggml_cgraph * cgraph,
        struct ggml_tensor * tensor ) {
    check( tensor );
    auto gx = GGMLCapture::Get().get( cgraph );
    auto tx = GGMLCapture::Get().get( tensor );
    if ( tx->tensor->op == GGML_OP_CPY ) {
        tx = GGMLCapture::Get().get( tx->tensor->src[0] );
    }
    if ( tx->tensor->op == GGML_OP_VIEW ) {
        tx = GGMLCapture::Get().get( tx->tensor->view_src );
    }
    gx->tensors.push_back( tx );
    ggml_build_forward_expand( cgraph, tensor );
}
#define ggml_build_forward_expand(...) _ggml_build_forward_expand(__VA_ARGS__)

// this should probably be always last in case we need to call the real one
struct ggml_tensor * _ggml_set_name   (
        struct ggml_tensor * tensor,
        const char * name ) {
    auto tx = GGMLCapture::Get().get( tensor );
    tx->name = name;
    int len = strlen(tensor->name);
    int remaining = GGML_MAX_NAME - len;
    if (remaining > 0)
        snprintf( tensor->name + len, remaining, "%s", name);
    return tensor;
}
#define ggml_set_name(...)    _ggml_set_name   (__VA_ARGS__)




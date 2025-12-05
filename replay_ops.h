#pragma once

struct tensor_data_t {
	ggml_type type;
	std::vector<int64_t> ne;
	int64_t offset;
	int64_t nbytes;
};

struct tensor_t {
	std::string id;
	std::string op_name;
	std::vector<std::string> src;
	tensor_data_t data;
	std::string name;
	std::string group;
	std::string caller;
};

struct group_t {
	std::string id;
	std::string name;
	std::string parent;
	std::vector<std::string> tensors;
	std::vector<std::string> children;
};

struct cgraph_t {
	std::vector<std::string> forward_expand;
};

std::map<std::string,ggml_type> type_map = {
	{"i8", GGML_TYPE_I8},
	{"i16", GGML_TYPE_I16},
	{"i32", GGML_TYPE_I32},
	{"i64", GGML_TYPE_I64},
	{"f64", GGML_TYPE_F64},
	{"f32", GGML_TYPE_F32},
	{"f16", GGML_TYPE_F16},
	{"q4_0", GGML_TYPE_Q4_0},
	{"q4_1", GGML_TYPE_Q4_1},
	{"q5_0", GGML_TYPE_Q5_0},
	{"q5_1", GGML_TYPE_Q5_1},
	{"q8_0", GGML_TYPE_Q8_0},
	{"q8_1", GGML_TYPE_Q8_1},
	{"mxfp4", GGML_TYPE_MXFP4},
	{"q2_K", GGML_TYPE_Q2_K},
	{"q3_K", GGML_TYPE_Q3_K},
	{"q4_K", GGML_TYPE_Q4_K},
	{"q5_K", GGML_TYPE_Q5_K},
	{"q6_K", GGML_TYPE_Q6_K},
	{"iq2_xxs", GGML_TYPE_IQ2_XXS},
	{"iq2_xs", GGML_TYPE_IQ2_XS},
	{"iq3_xxs", GGML_TYPE_IQ3_XXS},
	{"iq3_s", GGML_TYPE_IQ3_S},
	{"iq2_s", GGML_TYPE_IQ2_S},
	{"iq1_s", GGML_TYPE_IQ1_S},
	{"iq1_m", GGML_TYPE_IQ1_M},
	{"iq4_nl", GGML_TYPE_IQ4_NL},
	{"iq4_xs", GGML_TYPE_IQ4_XS},
	{"q8_K", GGML_TYPE_Q8_K},
	{"bf16", GGML_TYPE_BF16},
	{"tq1_0", GGML_TYPE_TQ1_0},
	{"tq2_0", GGML_TYPE_TQ2_0}
};

int replay_parse_tensor_data(const_str_t & json, int offset, tensor_data_t & data) {
	offset = json_array_parse(json, offset, [&data](
			const_str_t & json, int offset, int index ) {
		switch(index) {
		case 0: {
			std::string type_name;
			offset = json_string_parse(json, offset, type_name);
			if (offset == -1)
				return offset;
			auto it = type_map.find(type_name);
			if (it == type_map.end())
				return replay_error("unknown type");
			data.type = it->second;
			return offset;
		}
		case 1: return json_int64_array_parse(json, offset, data.ne);
		case 2: return json_int64_parse(json, offset, data.offset);
		case 3: return json_int64_parse(json, offset, data.nbytes);
		}
		return replay_error("unexpected item in group");
	});
	return offset;
}

#include <cmath>
#include <iostream>
bool compareFloats(float a, float b, float epsilonBase) {
    // Calculate the absolute difference
    double diff = std::fabs(a - b);
	auto epsilon = epsilonBase;

	// Calculate the scaled tolerance based on the relative size of the numbers
	if (a != 0.f || b != 0.f) {
		epsilon *= std::max(std::fabs(a), std::fabs(b));
	}

    // Return true if the difference is within the allowed tolerance
    //return diff <= epsilon;// || (diff < std::numeric_limits<double>::min() && a == b);
    if ( diff <= epsilon )
        return true;
    return false;
}
bool compareFloats(float * a, float * b, int n, float epsilonBase) {
	if (n == 0)
		return true;

	// Calculate the scaled tolerance based on the relative size of the numbers
	float max = std::max(std::fabs(a[0]), std::fabs(b[0]));
	for (int i = 1; i < n; i++) {
		max = std::max(max, std::fabs(a[i]));
		max = std::max(max, std::fabs(b[i]));
	}
	auto epsilon = epsilonBase;
	if (max > 0.f)
		epsilon *= max;
	
    // Return true if the difference is within the allowed tolerance
	for (int i = 0; i < n; i++)  {
		auto diff = std::fabs(a[i] - b[i]);
		if (diff > epsilon)
			return false;
	}
    return true;
}

class replay_op_base : public tensor_t {
public:
	bool implemented;
	ggml_tensor * tensor;
	bool leaf;
	replay_op_base(std::string id, std::string op_name) {
		this->id = id;
		this->op_name = op_name;
		implemented = false;
		tensor = NULL;
	}
	virtual int parse(const_str_t & json, int offset) {
		return json_skip_value(json, offset);
	}

	virtual ggml_tensor * alloc(ggml_context * ctx, ggml_tensor ** src) {
		assert( false );
		return NULL;
	}

	void reset() { tensor = NULL; };
	ggml_tensor * alloc(ggml_context * ctx) {
		tensor = ggml_new_tensor(ctx, data.type, data.ne.size(), data.ne.data());
		leaf = true;
		return tensor;
	}
	bool can_load() {
		return data.nbytes != 0;
	}
	void load(FILE * f, void * data) {
#ifdef _WIN32
		//assert( fseeko64(f, this->data.offset, SEEK_SET) == 0 );
		assert( _fseeki64(f, this->data.offset, SEEK_SET) == 0 );
#else
		assert( fseek(f, this->data.offset, SEEK_SET) == 0 );
#endif
		assert( fread(data, this->data.nbytes, 1, f) == 1 );
	}
	void load(FILE * f, std::vector<uint8_t> & data) {
		data.resize( this->data.nbytes );
		load(f, data.data());
	}
	void load(FILE * f, bool backend = false) {
		assert( leaf );
		if (backend) {
			std::vector<uint8_t> buf(data.nbytes);
			load(f, buf);
			ggml_backend_tensor_set(tensor, buf.data(), 0, data.nbytes);
		} else {
			load(f, tensor->data);
		}
	}
	void check_alloc() {
		assert( tensor->type == data.type );

		size_t i = 0;
		size_t min = data.ne.size();
		if (min > GGML_MAX_DIMS) min = GGML_MAX_DIMS;
		for (; i < min; i++) {
			assert( tensor->ne[i] == data.ne[i] );
		}
		for (; i < GGML_MAX_DIMS; i++) {
			assert( tensor->ne[i] == 1 );
		}
		for (; i < data.ne.size(); i++) {
			assert( data.ne[i] == 1 );
		}
	}
	bool check_results(FILE * f, float epsilonBase = 1e-5, bool backend = false) {
		std::vector<uint8_t> expected_buf(data.nbytes);
		load(f, expected_buf);
		assert( data.type == GGML_TYPE_F32 || data.type == GGML_TYPE_I32 ); // TODO
		if (backend) {
			std::vector<uint8_t> result_buf(data.nbytes);
			ggml_backend_tensor_get( tensor, result_buf.data(), 0, data.nbytes );
			size_t nelements = ggml_nelements( tensor );
			if ( data.type == GGML_TYPE_F32 ) {
				float * expected = (float*)expected_buf.data();
				float * result = (float*)result_buf.data();
				return compareFloats( expected, result, nelements, epsilonBase );
			} else if (data.type == GGML_TYPE_I32) {
				int32_t * expected = (int32_t*)expected_buf.data();
				int32_t * result = (int32_t*)result_buf.data();
				for (size_t i = 0; i < nelements; i++) {
					if (expected[i] != result[i])
						return false;
				}
			}
		} else {
			size_t nelements = ggml_nelements( tensor );
			if ( data.type == GGML_TYPE_F32 ) {
				float * expected = (float*)expected_buf.data();
				float * result = (float*)tensor->data;
				return compareFloats( expected, result, nelements, epsilonBase );
			} else if (data.type == GGML_TYPE_I32) {
				int32_t * expected = (int32_t*)expected_buf.data();
				int32_t * result = (int32_t*)tensor->data;
				for (size_t i = 0; i < nelements; i++) {
					if (expected[i] != result[i])
						return false;
				}
			}
		}
		return true;
	}
};

class replay_op1 : public replay_op_base {
public:
	replay_op1(std::string id, std::string op_name)
		: replay_op_base(id, op_name) {}
	virtual int parse(const_str_t & json, int offset) {
		if (src.size() != 1)
			return replay_error("src size not 1");
		return json_skip_value(json, offset);
	}
};

#define CLASS_OP1(op) \
class replay_op_##op : public replay_op1 {\
	public:\
	replay_op_##op(std::string id, std::string op_name)\
		: replay_op1(id, op_name) { implemented = true; }\
	virtual ggml_tensor * alloc(ggml_context * ctx, ggml_tensor ** src) {\
		tensor = ggml_##op(ctx, src[0]);\
		return tensor;\
	}\
}

class replay_op2 : public replay_op_base {
public:
	replay_op2(std::string id, std::string op_name)
		: replay_op_base(id, op_name) {}
	virtual int parse(const_str_t & json, int offset) {
		if (src.size() != 2)
			return replay_error("src size not 2");
		return json_skip_value(json, offset);
	}
};

#define CLASS_OP2(op) \
class replay_op_##op : public replay_op2 {\
	public:\
	replay_op_##op(std::string id, std::string op_name)\
		: replay_op2(id, op_name) { implemented = true; }\
	virtual ggml_tensor * alloc(ggml_context * ctx, ggml_tensor ** src) {\
		tensor = ggml_##op(ctx, src[0], src[1]);\
		return tensor;\
	}\
}

class replay_op_inplace : public replay_op_base {
public:
	bool inplace;
	replay_op_inplace(std::string id, std::string op_name)
		: replay_op_base(id, op_name) {}
	virtual int parse(const_str_t & json, int offset) {
		if (src.size() != 2)
			return replay_error("src size not 2");
		return json_bool_parse(json, offset, inplace);
	}
};

#define CLASS_OP_INPLACE(op) \
class replay_op_##op : public replay_op_inplace {\
	public:\
	replay_op_##op(std::string id, std::string op_name)\
		: replay_op_inplace(id, op_name) { implemented = true; }\
	virtual ggml_tensor * alloc(ggml_context * ctx, ggml_tensor ** src) {\
		if (inplace)\
			tensor = ggml_##op##_inplace(ctx, src[0], src[1]);\
		else\
			tensor = ggml_##op(ctx, src[0], src[1]);\
		return tensor;\
	}\
}

CLASS_OP_INPLACE(add);
CLASS_OP_INPLACE(sub);
CLASS_OP_INPLACE(mul);
CLASS_OP_INPLACE(div);
CLASS_OP1(neg);
CLASS_OP1(sum);

class replay_op_repeat_4d : public replay_op_base {
public:
	std::vector<int64_t> ne;
	replay_op_repeat_4d(std::string id, std::string op_name)
		: replay_op_base(id, op_name) { implemented = true; }
	virtual int parse(const_str_t & json, int offset) {
		if (src.size() != 1)
			return replay_error("src size not 1");
		offset = json_int64_array_parse(json, offset, ne);
		if (ne.size() != 4)
			return replay_error("ne array size not 4");
		return offset;
	}
	virtual ggml_tensor * alloc(ggml_context * ctx, ggml_tensor ** src) {
		tensor = ggml_repeat_4d(ctx, src[0], ne[0], ne[1], ne[2], ne[3]);
		return tensor;
	}
};

class replay_op_concat : public replay_op_base {
public:
	int64_t dim;
	replay_op_concat(std::string id, std::string op_name)
		: replay_op_base(id, op_name) { implemented = true; }
	virtual int parse(const_str_t & json, int offset) {
		if (src.size() != 2)
			return replay_error("src size not 2");
		offset = json_int64_parse(json, offset, dim);
		if (dim < 0 || dim > 3)
			return replay_error("out of range dim value");
		return offset;
	}
	virtual ggml_tensor * alloc(ggml_context * ctx, ggml_tensor ** src) {
		tensor = ggml_concat(ctx, src[0], src[1], dim);
		return tensor;
	}
};

CLASS_OP1(elu);
CLASS_OP1(gelu);
CLASS_OP1(silu);

class replay_op_norm : public replay_op_base {
public:
	float eps;
	replay_op_norm(std::string id, std::string op_name)
		: replay_op_base(id, op_name) { implemented = true; }
	virtual int parse(const_str_t & json, int offset) {
		if (src.size() != 1)
			return replay_error("src size not 1");
		return json_float_parse(json, offset, eps);
	}
	virtual ggml_tensor * alloc(ggml_context * ctx, ggml_tensor ** src) {
		tensor = ggml_norm(ctx, src[0], eps);
		return tensor;
	}
};

class replay_op_rms_norm : public replay_op_base {
public:
	float eps;
	replay_op_rms_norm(std::string id, std::string op_name)
		: replay_op_base(id, op_name) { implemented = true; }
	virtual int parse(const_str_t & json, int offset) {
		if (src.size() != 1)
			return replay_error("src size not 1");
		return json_float_parse(json, offset, eps);
	}
	virtual ggml_tensor * alloc(ggml_context * ctx, ggml_tensor ** src) {
		tensor = ggml_rms_norm(ctx, src[0], eps);
		return tensor;
	}
};

CLASS_OP2(mul_mat);
CLASS_OP1(argmax);

class replay_op_scale : public replay_op_base {
public:
	float s;
	replay_op_scale(std::string id, std::string op_name)
		: replay_op_base(id, op_name) { implemented = true; }
	virtual int parse(const_str_t & json, int offset) {
		if (src.size() != 1)
			return replay_error("src size not 1");
		return json_float_parse(json, offset, s);
	}
	virtual ggml_tensor * alloc(ggml_context * ctx, ggml_tensor ** src) {
		tensor = ggml_scale(ctx, src[0], s);
		return tensor;
	}
};

CLASS_OP2(cpy);

class replay_op_cast : public replay_op_base {
public:
	ggml_type type;
	replay_op_cast(std::string id, std::string op_name)
		: replay_op_base(id, op_name) { implemented = true; }
	virtual int parse(const_str_t & json, int offset) {
		if (src.size() != 1)
			return replay_error("src size not 1");
		std::string type_name;
		offset = json_string_parse(json, offset, type_name);
		auto it = type_map.find(type_name);
		if (it == type_map.end())
			return replay_error("unknown type");
		type = it->second;
		return offset;
	}
	virtual ggml_tensor * alloc(ggml_context * ctx, ggml_tensor ** src) {
		tensor = ggml_cast(ctx, src[0], type);
		return tensor;
	}
};

CLASS_OP1(cont);

class replay_op_reshape : public replay_op_base {
public:
	std::vector<int64_t> ne;
	replay_op_reshape(std::string id, std::string op_name)
		: replay_op_base(id, op_name) { implemented = true; }
	virtual int parse(const_str_t & json, int offset) {
		if (src.size() != 1)
			return replay_error("src size not 1");
		offset = json_int64_array_parse(json, offset, ne);
		if (ne.size() < 2 || ne.size() > 4)
			return replay_error("reshape dimensions wrong");
		return offset;
	}
	virtual ggml_tensor * alloc(ggml_context * ctx, ggml_tensor ** src) {
		switch(ne.size()) {
		case 2: tensor = ggml_reshape_2d(ctx, src[0], ne[0], ne[1]); break;
		case 3: tensor = ggml_reshape_3d(ctx, src[0], ne[0], ne[1], ne[2]); break;
		case 4: tensor = ggml_reshape_4d(ctx, src[0], ne[0], ne[1], ne[2], ne[3]); break;
		}
		return tensor;
	}
};

class replay_op_view : public replay_op_base {
public:
	std::vector<int64_t> params;
	replay_op_view(std::string id, std::string op_name)
		: replay_op_base(id, op_name) { implemented = true; }
	virtual int parse(const_str_t & json, int offset) {
		if (src.size() != 1)
			return replay_error("src size not 1");
		offset = json_int64_array_parse(json, offset, params);
		if (params.size() < 2 || params.size() > 8 || params.size() & 1)
			return replay_error("view params wrong");
		return offset;
	}
	virtual ggml_tensor * alloc(ggml_context * ctx, ggml_tensor ** src) {
		switch(params.size()) {
		case 2: tensor = ggml_view_1d(ctx, src[0], params[0], params[1]); break;
		case 4: tensor = ggml_view_2d(ctx, src[0], params[0], params[1], params[2], params[3]); break;
		case 6: tensor = ggml_view_3d(ctx, src[0], params[0], params[1], params[2], params[3], params[4], params[5]); break;
		case 8: tensor = ggml_view_4d(ctx, src[0], params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7]); break;
		}
		return tensor;
	}
};

class replay_op_permute : public replay_op_base {
public:
	std::vector<int64_t> axis;
	replay_op_permute(std::string id, std::string op_name)
		: replay_op_base(id, op_name) { implemented = true; }
	virtual int parse(const_str_t & json, int offset) {
		if (src.size() != 1)
			return replay_error("src size not 1");
		offset = json_int64_array_parse(json, offset, axis);
		if (axis.size() != 4)
			return replay_error("axis array size not 4");
		return offset;
	}
	virtual ggml_tensor * alloc(ggml_context * ctx, ggml_tensor ** src) {
		tensor = ggml_permute(ctx, src[0], axis[0], axis[1], axis[2], axis[3]);
		return tensor;
	}
};

CLASS_OP1(transpose);
CLASS_OP1(soft_max);

class replay_op_soft_max_ext : public replay_op_base {
public:
	std::vector<float> params;
	replay_op_soft_max_ext(std::string id, std::string op_name)
		: replay_op_base(id, op_name) { implemented = true; }
	virtual int parse(const_str_t & json, int offset) {
		if (src.size() < 1 || src.size() > 2)
			return replay_error("src size not 1 or 2");
		offset = json_float_array_parse(json, offset, params);
		if (params.size() != 2)
			return replay_error("params array size not 2");
		return offset;
	}
	virtual ggml_tensor * alloc(ggml_context * ctx, ggml_tensor ** src) {
		if (this->src.size() == 1)
			tensor = ggml_soft_max_ext(ctx, src[0], NULL, params[0], params[1]);
		else
			tensor = ggml_soft_max_ext(ctx, src[0], src[1], params[0], params[1]);
		return tensor;
	}
};

CLASS_OP2(get_rows);

class replay_op_clamp : public replay_op_base {
public:
	std::vector<float> params;
	replay_op_clamp(std::string id, std::string op_name)
		: replay_op_base(id, op_name) { implemented = true; }
	virtual int parse(const_str_t & json, int offset) {
		if (src.size() != 1)
			return replay_error("src size not 1");
		offset = json_float_array_parse(json, offset, params);
		if (params.size() != 2)
			return replay_error("params array size not 2");
		return offset;
	}
	virtual ggml_tensor * alloc(ggml_context * ctx, ggml_tensor ** src) {
		tensor = ggml_clamp(ctx, src[0], params[0], params[1]);
		return tensor;
	}
};

class replay_op_conv_1d : public replay_op_base {
public:
	std::vector<int64_t> params;
	replay_op_conv_1d(std::string id, std::string op_name)
		: replay_op_base(id, op_name) { implemented = true; }
	virtual int parse(const_str_t & json, int offset) {
		if (src.size() != 2)
			return replay_error("src size not 2");
		offset = json_int64_array_parse(json, offset, params);
		if (params.size() != 3)
			return replay_error("params array size not 3");
		return offset;
	}
	virtual ggml_tensor * alloc(ggml_context * ctx, ggml_tensor ** src) {
		tensor = ggml_conv_1d(ctx, src[0], src[1], params[0], params[1], params[2]);
		return tensor;
	}
};

class replay_op_conv_transpose_1d : public replay_op_base {
public:
	std::vector<int64_t> params;
	replay_op_conv_transpose_1d(std::string id, std::string op_name)
		: replay_op_base(id, op_name) { implemented = true; }
	virtual int parse(const_str_t & json, int offset) {
		if (src.size() != 2)
			return replay_error("src size not 2");
		offset = json_int64_array_parse(json, offset, params);
		if (params.size() != 3)
			return replay_error("params array size not 3");
		return offset;
	}
	virtual ggml_tensor * alloc(ggml_context * ctx, ggml_tensor ** src) {
		tensor = ggml_conv_transpose_1d(ctx, src[0], src[1], params[0], params[1], params[2]);
		return tensor;
	}
};

class replay_op_arange : public replay_op_base {
public:
	std::vector<float> params;
	replay_op_arange(std::string id, std::string op_name)
		: replay_op_base(id, op_name) { implemented = true; }
	virtual int parse(const_str_t & json, int offset) {
		if (src.size() != 0)
			return replay_error("src size not 0");
		offset = json_float_array_parse(json, offset, params);
		if (params.size() != 3)
			return replay_error("params array size not 3");
		return offset;
	}
	virtual ggml_tensor * alloc(ggml_context * ctx, ggml_tensor ** src) {
		tensor = ggml_arange(ctx, params[0], params[1], params[2]);
		return tensor;
	}
};

class replay_op_top_k : public replay_op_base {
public:
	int64_t k;
	replay_op_top_k(std::string id, std::string op_name)
		: replay_op_base(id, op_name) { implemented = true; }
	virtual int parse(const_str_t & json, int offset) {
		if (src.size() != 1)
			return replay_error("src size not 1");
		return json_int64_parse(json, offset, k);
	}
	virtual ggml_tensor * alloc(ggml_context * ctx, ggml_tensor ** src) {
		tensor = ggml_top_k(ctx, src[0], k);
		return tensor;
	}
};

class replay_op_timestep_embedding : public replay_op_base {
public:
	std::vector<int64_t> params;
	replay_op_timestep_embedding(std::string id, std::string op_name)
		: replay_op_base(id, op_name) { implemented = true; }
	virtual int parse(const_str_t & json, int offset) {
		if (src.size() != 1)
			return replay_error("src size not 1");
		offset = json_int64_array_parse(json, offset, params);
		if (params.size() != 2)
			return replay_error("params array size not 2");
		return offset;
	}
	virtual ggml_tensor * alloc(ggml_context * ctx, ggml_tensor ** src) {
		tensor = ggml_timestep_embedding(ctx, src[0], params[0], params[1]);
		return tensor;
	}
};

replay_op_base * make_replay_op(std::string op, std::string id) {
	if (op == "add") return new replay_op_add(id, op);
	if (op == "sub") return new replay_op_sub(id, op);
	if (op == "mul") return new replay_op_mul(id, op);
	if (op == "div") return new replay_op_div(id, op);
	if (op == "neg") return new replay_op_neg(id, op);
	if (op == "sum") return new replay_op_sum(id, op);
	if (op == "repeat_4d") return new replay_op_repeat_4d(id, op);
	if (op == "concat") return new replay_op_concat(id, op);
	if (op == "elu") return new replay_op_elu(id, op);
	if (op == "gelu") return new replay_op_gelu(id, op);
	if (op == "silu") return new replay_op_silu(id, op);
	if (op == "norm") return new replay_op_norm(id, op);
	if (op == "rms_norm") return new replay_op_rms_norm(id, op);
	if (op == "mul_mat") return new replay_op_mul_mat(id, op);
	if (op == "argmax") return new replay_op_argmax(id, op);
	if (op == "scale") return new replay_op_scale(id, op);
	if (op == "cpy") return new replay_op_cpy(id, op);
	if (op == "cast") return new replay_op_cast(id, op);
	if (op == "cont") return new replay_op_cont(id, op);
	if (op == "reshape") return new replay_op_reshape(id, op);
	if (op == "view") return new replay_op_view(id, op);
	if (op == "permute") return new replay_op_permute(id, op);
	if (op == "transpose") return new replay_op_transpose(id, op);
	if (op == "soft_max") return new replay_op_soft_max(id, op);
	if (op == "soft_max_ext") return new replay_op_soft_max_ext(id, op);
	if (op == "get_rows") return new replay_op_get_rows(id, op);
	if (op == "clamp") return new replay_op_clamp(id, op);
	if (op == "conv_1d") return new replay_op_conv_1d(id, op);
	if (op == "conv_transpose_1d") return new replay_op_conv_transpose_1d(id, op);
	if (op == "arange") return new replay_op_arange(id, op);
	if (op == "top_k") return new replay_op_top_k(id, op);
	if (op == "timestep_embedding") return new replay_op_timestep_embedding(id, op);
	return new replay_op_base(id, op);
}

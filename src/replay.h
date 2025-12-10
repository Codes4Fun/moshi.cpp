#pragma once

#include "src/json.h"
#include <assert.h>
#include <iostream>
#include <string>
#include <map>

/*
struct 

class Captured {
	std::vector<
};
*/

int _replay_error(const char * message, const char * file, int line) {
	printf("%s", message);
	return -1;
}
#define replay_error(message) _replay_error(message, __FILE__, __LINE__)

#include "replay_ops.h"

ggml_tensor * alloc(
		ggml_context * ctx,
		std::map<std::string,replay_op_base*> & tensor_from_id,
		replay_op_base * op,
		std::vector<replay_op_base*> & loads) {
	if (op->tensor)
		return op->tensor;
	
	int nsrc = op->src.size();
	assert( nsrc <= 2 );
	ggml_tensor * src[2];
	for (int i = 0; i < nsrc; i++) {
		auto src_op = tensor_from_id[op->src[i]];
		src[i] = alloc( ctx, tensor_from_id, src_op, loads );
	}
	
	ggml_tensor * tensor;
	if (op->op_name == "new_tensor") {
		loads.push_back( op );
		tensor = op->alloc( ctx );
	} else {
		tensor = op->alloc( ctx, src );
	}
	ggml_set_name( tensor, op->id.c_str());
	return tensor;
}

int replay_test( std::string filename, ggml_backend * backend = NULL ) {
	auto f = fopen( (filename + ".json").c_str(), "rb" );
	assert( f );

	fseek( f, 0, SEEK_END );
	auto length = ftell( f );
	fseek( f, 0, SEEK_SET );
	assert( length > 0 );

	std::vector<char> raw( length );
	assert( fread(raw.data(), length, 1, f) == 1 );
	fclose( f );

	const_str_t json = {raw.data(), (int)length};

    int offset = str_skip_whitespaces(json, 0);
    if (offset >= length || json.s[offset] != '{') {
        return replay_error("did not find expected json object");
	}

	std::vector<replay_op_base*> tensors;
	std::vector<group_t*> groups;
	std::vector<std::string> forward_expand;

    offset = json_object_parse(json, offset, [
			&tensors,
			&groups,
			&forward_expand
		](
            const_str_t & json, int offset,
            int key_start, int key_length
        ) {
		const char * key = json.s + key_start;
		if (key_length == 6 && strncmp(key, "tensor", 6) == 0) {
			// tensors
			printf("found tensors\n");
			offset = json_object_parse(json, offset, [&tensors](
					const_str_t & json, int offset,
					int key_start, int key_length
				) {
				std::string id;
				id.assign(json.s + key_start, key_length);
				tensors.push_back(NULL);
				auto & tensor = tensors.back();
				offset = json_array_parse(json, offset, [&tensor,&id](
						const_str_t & json, int offset, int index ) {
					switch(index) {
					case 0: {
						std::string op;
						offset = json_string_parse(json, offset, op);
						if (offset == -1)
							return -1;
						if (offset >= json.length)
							return replay_error("unexpected end parsing tensor");
						tensor = make_replay_op(op, id);
						if (!tensor)
							return replay_error("unknown op");
						return offset;
					}
					case 1: return json_string_array_parse(json, offset, tensor->src);
					case 2: return tensor->parse(json, offset);
					case 3: return replay_parse_tensor_data(json, offset, tensor->data);
					case 4: return json_string_parse(json, offset, tensor->name);
					case 5: return json_string_parse(json, offset, tensor->group);
					case 6: return json_string_parse(json, offset, tensor->caller);
					}
					return replay_error("unexpected item in tensor");
				});
				if (offset == -1)
					return offset;
				if (offset >= json.length)
					return replay_error("unexpected end of tensors");

				std::cout << "  " << id << ' ' << tensor->op_name << ' ' << tensor->src.size() << "  ";
				if (tensor->data.nbytes > 1024*1024*1024)
					std::cout << (tensor->data.nbytes / 1024.f / 1024.f / 1024.f) << " GiB";
				else if (tensor->data.nbytes > 1024*1024)
					std::cout << (tensor->data.nbytes / 1024.f / 1024.f) << " MiB";
				else if (tensor->data.nbytes > 1024)
					std::cout << (tensor->data.nbytes / 1024.f) << " KiB";
				else
					std::cout << tensor->data.nbytes << " B";
				std::cout << std::endl;

				return offset;
			});
			if (offset >= json.length)
				return replay_error("unexpected end of object");
			return offset;
		} else if (key_length == 6 && strncmp(key, "groups", 6) == 0) {
			// groups
			offset = json_object_parse(json, offset, [&groups](
					const_str_t & json, int offset,
					int key_start, int key_length
				) {
				std::string id;
				id.assign(json.s + key_start, key_length);
				groups.push_back(new group_t{id});
				auto group = groups.back();
				offset = json_array_parse(json, offset, [group](
						const_str_t & json, int offset, int index ) {
					switch(index) {
					case 0: return json_string_parse(json, offset, group->name);
					case 1: return json_string_parse(json, offset, group->parent);
					case 2: return json_string_array_parse(json, offset, group->tensors);
					case 3: return json_string_array_parse(json, offset, group->children);
					}
					return replay_error("unexpected item in group");
				});
				if (offset == -1)
					return offset;
				if (offset >= json.length)
					return replay_error("unexpected end of groups");
				return offset;
			});
			if (offset >= json.length)
				return replay_error("unexpected end of object");

			printf("found groups %ld\n", groups.size());
			for (auto group : groups) {
				printf("  %s %s %s %ld %ld\n", group->id.c_str(), group->name.c_str(), group->parent.c_str(), group->tensors.size(), group->children.size());
			}

			return offset;
		} else if (key_length == 14 && strncmp(key, "forward_expand", 6) == 0) {
			// forward_expand
			offset = json_string_array_parse(json, offset, forward_expand);
			if (offset == -1)
				return offset;
			if (offset >= json.length)
				return replay_error("unexpected end of object");
			printf("found forward_expand %ld\n", forward_expand.size());
			for (auto item : forward_expand) {
				printf("  %s\n", item.c_str());
			}
			return offset;
		} else if (key_length == 6 && strncmp(key, "nbytes", 6) == 0) {
			// nbytes
			int64_t nbytes;
			offset = json_int64_parse(json, offset, nbytes);
			printf("nbytes: %ld\n", nbytes);
			return offset;
		}
		return replay_error("unknown key");
	});

	std::map<std::string,replay_op_base*> tensor_from_id;
	for (auto tensor : tensors) {
		tensor_from_id[tensor->id] = tensor;
	}
	std::map<std::string,group_t*> group_from_id;
	for (auto group : groups) {
		group_from_id[group->id] = group;
	}

	std::cout << "validating" << std::endl;
	
	int64_t nbytes = 0;
	for (auto tensor : tensors) {
		for (auto tid : tensor->src) {
			auto it = tensor_from_id.find(tid);
			if (it == tensor_from_id.end())
				return replay_error("tensor not found");
		}
		nbytes += tensor->data.nbytes;
		if (tensor->group == "0")
			continue;
		auto it = group_from_id.find(tensor->group);
		if (it == group_from_id.end())
			return replay_error("group not found");
		auto & group_tensors = it->second->tensors;
		bool found = false;
		for (auto id : group_tensors) {
			if ( id == tensor->id ) {
				found = true;
				break;
			}
		}
		if (!found)
			return replay_error("group to tensor reference error");
	}
	for (auto group : groups) {
		for (auto tid : group->tensors) {
			auto it = tensor_from_id.find(tid);
			if (it == tensor_from_id.end())
				return replay_error("tensor not found");
			if (it->second->group != group->id)
				return replay_error("tensor to group reference error");
		}
		for (auto gid : group->children) {
			auto it = group_from_id.find(gid);
			if (it == group_from_id.end())
				return replay_error("group not found");
			if (it->second->parent != group->id)
				return replay_error("group to group reference error");
		}
	}

	std::cout << "passed" << std::endl << "testing" << std::endl;

	// more than enough for individual op testing
	nbytes += tensors.size() * ggml_tensor_overhead();
	nbytes += ggml_graph_overhead();
	if (!backend) {
		nbytes += 512 * 1024 * 1024;
	}
	auto ctx = ggml_init({ (size_t)nbytes, 0, backend? true : false });
	assert( ctx );

	f = fopen( (filename + ".tensors").c_str(), "rb" );
	assert( f );

	int tested = 0;
	int skipped = 0;
	std::vector<std::string> failed;
    int idx = -1;
	for (auto tensor : tensors) {
        idx++;
		if (!tensor->implemented ||
            tensor->op_name == "view" ||
            tensor->op_name == "transpose" ||
            tensor->op_name == "permute" ||
            tensor->op_name == "cont"
        ) {
			if (tensor->op_name == "new_tensor")
				continue;
			printf("  skipping %s\n", tensor->op_name.c_str());
			skipped++;
			continue;
		}
		if (!tensor->can_load()) {
			printf("  skipping %s cannot load\n", tensor->op_name.c_str());
			skipped++;
			continue;
		}
		int nsrc = tensor->src.size();
		assert( nsrc <= 2 );
		// collect sources
		bool can_load = true;
		replay_op_base * src_op[2];
		for (int i = 0; i < nsrc; i++) {
			src_op[i] = tensor_from_id[tensor->src[i]];
			assert( src_op[i] );
			if (!src_op[i]->can_load()) {
				can_load = false;
				break;
			}
		}
		if (!can_load) {
			printf("  skipping %s child cannot load\n", tensor->op_name.c_str());
			skipped++;
			continue;
		}
		printf("  testing %s\n", tensor->op_name.c_str());
		// allocate
		ggml_tensor * src[2];
		for (int i = 0; i < nsrc; i++) {
			src[i] = src_op[i]->alloc( ctx );
		}
		auto op = tensor->alloc( ctx, src );
		tensor->check_alloc(); // check type and shape
		ggml_backend_buffer * buffer = NULL;
		if (backend)
			buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
		// load
		for (int i = 0; i < nsrc; i++) {
			src_op[i]->load( f, backend? true : false );
		}
		// compute
        /*if (idx == 189) {
            printf("break here!\n");
        }*/
		ggml_cgraph * gf = ggml_new_graph( ctx );
		ggml_build_forward_expand(gf, op);
		if (backend) {
			ggml_backend_graph_compute( backend, gf );
		} else {
			ggml_graph_compute_with_ctx( ctx, gf, 8 );
		}
		// check results
		if (backend) {
			if (!tensor->check_results( f, 1e-2, true )) {
				failed.push_back( tensor->op_name + " " + tensor->id + " " + std::to_string(idx) );
			}
		} else {
			if (!tensor->check_results( f, 1e-5, false )) {
				failed.push_back( tensor->op_name + " " + tensor->id + " " + std::to_string(idx) );
			}
		}
		tested++;

		if (buffer)
			ggml_backend_buffer_free(buffer);
		
		tensor->reset();
		for (int i = 0; i < nsrc; i++)
			src_op[i]->reset();
		ggml_reset( ctx );
	}
	
	for (auto fail : failed) {
		std::cout << fail << std::endl;
	}
	assert( failed.size() == 0 );

	printf("tested %d skipped %d\n", tested, skipped);

	// do full graph testing
	std::vector<replay_op_base*> loads;
	ggml_cgraph * gf = ggml_new_graph( ctx );
	for (auto tid : forward_expand) {
		auto op = tensor_from_id[tid];
		auto tensor = alloc( ctx, tensor_from_id, op, loads);
		ggml_build_forward_expand( gf, tensor );
	}
	ggml_backend_buffer * buffer = NULL;
	if (backend)
		buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
	for (auto op : loads) {
		op->load( f, backend? true : false );
	}
	if (backend) {
		ggml_backend_graph_compute( backend, gf );
	} else {
		ggml_graph_compute_with_ctx( ctx, gf, 8 );
	}
	for (auto tid : forward_expand) {
		auto op = tensor_from_id[tid];
		assert( op->check_results( f, 1e-2, backend? true : false ) );
	}
	// TODO: cleanup reset
	if (buffer)
		ggml_backend_buffer_free(buffer);

	fclose(f);

	ggml_free( ctx );

	return 0;
}

int replay_test() {
    //auto backend = ggml_backend_init_by_name("Vulkan0", NULL);
    //auto backend = ggml_backend_init_by_name("CUDA0", NULL);
    auto backend = ggml_backend_init_by_name("CPU", NULL);

	//std::string filename = "capture/voice";
	//std::string filename = "capture/text";
	//std::string filename = "capture/text_32";
	//std::string filename = "capture/depformer_32";
	//std::string filename = "capture/decode_4";

	//std::string filename = "h:/capture/voice";
	//std::string filename = "h:/capture/text";
	std::string filename = "h:/capture/text_32";
	//std::string filename = "h:/capture/depformer_32";
	//std::string filename = "h:/capture/decode_4";

	int result = replay_test( filename, backend );

    if (backend)
		ggml_backend_free(backend);

	return result;
}





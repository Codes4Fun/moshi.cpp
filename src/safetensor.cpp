
#include <stdint.h>

#include <vector>
#include <string>

#include <moshi/json.h>
#include <moshi/safetensor.h>

int safetensor_parse(const_str_t & json, safetensors_t & tensors) {
    // skip white spaces
    int offset = str_skip_whitespaces(json, 0);
    if (offset >= json.length || json.s[offset] != '{')
        return json_error("expected object");

    offset = json_object_parse(json, offset, [&tensors](
            const_str_t & json, int offset,
            int key_start, int key_length
        ) {
        safetensor_t tensor;
        tensor.key.assign(json.s + key_start, key_length);
        tensor.flags = 0;
        offset = json_object_parse(json, offset, [&tensor](
                const_str_t & json, int offset,
                int key_start, int key_length
            ) {
            const char * key = json.s + key_start;
            if (key_length == 5) {
                if (strncmp(key, "dtype", 5) == 0) {
                    // find string
                    offset = str_skip_whitespaces(json, offset);
                    if (json.s[offset] == '"') {
                        int str_start = ++offset;
                        offset = str_find_unescaped(json, offset, '"');
                        int str_length = offset - str_start;
                        tensor.flags |= SAFETENSOR_DTYPE;
                        tensor.dtype.assign(json.s + str_start, str_length);
                        return ++offset;
                    }
                } else if (strncmp(key, "shape", 5) == 0) {
                    // find int64_t array
                    offset = json_maybe_get_int64_array(json, offset, [&tensor](std::vector<int64_t> &array) {
                        tensor.flags |= SAFETENSOR_SHAPE;
                        tensor.shape.swap(array);
                    });
                    return offset;
                }
            } if (key_length == 12 && strncmp(key, "data_offsets", 12) == 0) {
                // find int64_t array
                offset = json_maybe_get_int64_array(json, offset, [&tensor](std::vector<int64_t> &array) {
                    if (array.size() == 2) {
                        tensor.flags |= SAFETENSOR_DATA_OFFSETS;
                        tensor.data_offsets.swap(array);
                    } else {
                        printf("warning: irregular data_offsets array\n");
                    }
                });
                return offset;
            }
            offset = json_skip_value(json, offset);
            return offset;
        });
        if (tensor.flags == (SAFETENSOR_DTYPE|SAFETENSOR_SHAPE|SAFETENSOR_DATA_OFFSETS))
            tensors[tensor.key] = tensor;
        return offset;
    });
    if (offset == -1)
        return 0;
    return (int) tensors.size();
}


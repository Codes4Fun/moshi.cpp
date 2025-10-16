#pragma once

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <vector>
#include <string>
#include <map>
#include <set>

#include "json.h"

#define SAFETENSOR_DTYPE 1
#define SAFETENSOR_SHAPE 2
#define SAFETENSOR_DATA_OFFSETS 4

struct safetensor_t {
    int flags;
    std::string key;
    std::string dtype;
    std::vector<int64_t> shape;
    std::vector<int64_t> data_offsets;
};

typedef std::map<std::string, safetensor_t> safetensors_t;


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
    return tensors.size();
}

#ifdef LOGGER

void usage(const char * msg = NULL) {
    printf("usage: ./safetensor file.safetensor\n");
    if (msg) printf("%s\n", msg);
    exit(1);
}

int main(int argc, char **argv) {
    if (argc < 2)
        usage();
    FILE * f = fopen(argv[1], "rb");
    if (!f)
        usage("invalid file");
    int64_t length;
    size_t r;
    r = fread(&length, sizeof(length), 1, f);
    if (r != 1 || length == 0)
        usage("error reading file");
    char * data = new char[length+1];
    if (!data)
        usage("unable to allocate buffer");
    r = fread(data, length, 1, f);
    if (r != 1)
        usage("error reading file");
    data[length] = 0;

    //printf("%s\n", data);

    const_str_t json = {data, (int)length};

    // skip white spaces
    int offset = str_skip_whitespaces(json, 0);
    if (offset >= length || data[offset] != '{')
        usage("did not find expected json object");

    //json_object_parse(json, offset, json_object_key_log);

    safetensors_t tensors;
    safetensor_parse(json, tensors);
    std::set<std::string> dtypes;
    for (auto it : tensors) {
        int64_t nelements = 1;
        for (auto dim : it.second.shape)
            nelements *= dim;
        printf("%s %s %ld { %ld", it.first.c_str(), it.second.dtype.c_str(), nelements, it.second.shape[0]);
        for (int i = 1; i < it.second.shape.size(); i++)
            printf(", %ld", it.second.shape[i]);
        printf(" }\n");
        dtypes.insert(it.second.dtype);
    }
    for (auto it: dtypes) {
        printf("%s\n", it.c_str());
    }

    /*do {
        // skip white spaces
        offset = str_skip_whitespaces(json, ++offset);
        if (offset >= length || data[offset] != '"')
            usage("did not find start of key string");
        int start = offset+1;
        offset = str_find_unescaped(json, start, '"');
        if (offset >= length)
            usage("did not find end of string");

        printf("%.*s\n", offset - start, data + start);

        // skip white spaces
        offset = str_skip_whitespaces(json, ++offset);
        if (offset >= length || data[offset] != ':')
            usage("did not find key value seperator");

        offset = json_skip_value(json, ++offset);
        if (offset == -1 || offset >= length)
            usage("error skipping value");

        // skip white spaces
        offset = str_skip_whitespaces(json, offset);
    } while (offset < length && data[offset] == ',');
    if (offset >= length || data[offset] != '}')
        usage("did not find object tail '}'");*/
}

#endif//LOGGER

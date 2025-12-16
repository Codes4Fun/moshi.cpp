#pragma once

#include <string.h>

struct const_str_t {
    const char * s;
    const int length;
};

const char white_spaces[] = " \t\n\r";

#define const_str(str) const_str_t{str, strlen(str)}

bool chr_of(char c, const_str_t & cs);
bool chr_of(char c, const char * cs);
int str_find(const_str_t & s, int offset, char c);
int str_find(const_str_t & s, int offset, const char * t, int length);
int str_find_unescaped(const_str_t & s, int offset, char c);
int str_find_of(const_str_t & s, int offset, const char * cs);
int str_find_not_of(const_str_t & s, int offset, const char * cs);
int str_find_whitespaces(const_str_t & s, int offset);
int str_skip_whitespaces(const_str_t & s, int offset);
int json_error(const char * err);
int json_skip_value(const_str_t & json, int offset);

template<class T>
int json_array_parse(const_str_t & json, int offset, T on_item) {
    offset = str_find_not_of(json, offset, white_spaces);
	if (offset >= json.length || json.s[offset] != '[')
		return json_error("expected array");
    offset = str_find_not_of(json, ++offset, white_spaces);
	if (offset >= json.length)
		return json_error("unexpected end of array");
	if (json.s[offset] == ']')
		return ++offset;
	int index = 0;
	do {
		offset = on_item(json, offset, index++);
		if (offset < 0)
			return offset;
		offset = str_find_not_of(json, offset, white_spaces);
		if (offset >= json.length)
			return json_error("unexpected end of array");
		if (json.s[offset] != ',')
			break;
		offset = str_find_not_of(json, ++offset, white_spaces);
		if (offset >= json.length)
			return offset;
	} while(true);
	if (json.s[offset] != ']')
        return json_error("expected tail of array ']'");
	return ++offset;
}

template<class T>
int json_object_parse(const_str_t & json, int offset, T on_item) {
    offset = str_find_not_of(json, offset, white_spaces);
	if (offset >= json.length || json.s[offset] != '{')
		return json_error("expected object");
    offset = str_find_not_of(json, ++offset, white_spaces);
	if (offset >= json.length)
		return json_error("unexpected end of object");
	if (json.s[offset] == '}')
		return ++offset;
    do {
        if (json.s[offset] != '"')
            return json_error("expected key");
        int key_start = ++offset;
        offset = str_find_unescaped(json, offset, '"');
        if (offset >= json.length)
            return json_error("expected end of key");
        int key_length = offset - key_start;

        offset = str_find_not_of(json, ++offset, white_spaces);
        if (offset >= json.length || json.s[offset] != ':')
            return json_error("expected seperator");

        offset = str_skip_whitespaces( json, ++offset );
        if (offset >= json.length)
            return json_error("unexpected end of object");

        offset = on_item(json, offset, key_start, key_length);
		if (offset == -1)
			return -1;

        offset = str_find_not_of(json, offset, white_spaces);
		if (offset >= json.length)
			return json_error("unexpected end of object");
		if (json.s[offset] != ',')
			break;
        offset = str_find_not_of(json, ++offset, white_spaces);
		if (offset >= json.length)
			return offset;
    } while (true);
    if (json.s[offset] != '}')
        return json_error("expected tail of object '}'");
    return ++offset;
}

int json_object_key_log(const_str_t & json, int offset, int key_start, int key_length);
int json_int64_parse(const_str_t & json, int offset, int64_t & value);

template<class T>
int json_maybe_get_int64_array(const_str_t & json, int offset, T on_array) {
    offset = str_find_not_of(json, offset, white_spaces);
    if (offset >= json.length)
        return json_error("unexpected end before value");

    if (json.s[offset] != '[')
        return json_skip_value(json, offset);

    bool skipping = false;
    std::vector<int64_t> array;
    do {
        offset = str_find_not_of(json, ++offset, white_spaces);
        if (offset >= json.length)
            return json_error("unexpected end of array");

        char c = json.s[offset];
        if (!skipping && chr_of(c, "0123456789-")) {
            const char *start = json.s + offset;
            char *end;
            double d = strtod(start, &end);
            int64_t i = (int64_t)d;
            if (d == (double)i) {
                array.push_back(i);
            } else {
                skipping = true;
            }
            offset = (int) (end - start) + offset;
        } else {
            skipping = true;

            offset = json_skip_value(json, ++offset);
            if (offset == -1)
                return -1;
        }
        offset = str_find_not_of(json, offset, white_spaces);
    } while (offset < json.length && json.s[offset] == ',');
    if (json.s[offset] != ']')
        return json_error("expected tail of array ']'");

    if (!skipping) on_array(array);

    return ++offset;
}

int json_double_parse(const_str_t & json, int offset, double & value);
int json_float_parse(const_str_t & json, int offset, float & value);
int json_bool_parse(const_str_t & json, int offset, bool & value);
int json_string_parse(const_str_t & json, int offset, std::string & item);
int json_string_array_parse(const_str_t & json, int offset, std::vector<std::string> & items);
int json_int64_array_parse(const_str_t & json, int offset, std::vector<int64_t> & items);
int json_float_array_parse(const_str_t & json, int offset, std::vector<float> & items);

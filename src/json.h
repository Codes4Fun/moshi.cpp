#pragma once

#include <string.h>

struct const_str_t {
    const char * s;
    const int length;
};

const char white_spaces[] = " \t\n\r";

#define const_str(str) const_str_t{str, strlen(str)}

bool chr_of(char c, const_str_t & cs) {
    int ci = 0;
    while (ci < cs.length && c != cs.s[ci])
        ci++;
    return (ci != cs.length);
}

bool chr_of(char c, const char * cs) {
    for (; *cs; cs++) {
        if (*cs == c)
            return true;
    }
    return false;
}

int str_find(const_str_t & s, int offset, char c) {
    while (offset < s.length && s.s[offset] != c)
        offset++;
    return offset;
}

int str_find_unescaped(const_str_t & s, int offset, char c) {
    while (offset < s.length) {
        if (s.s[offset] == '\\') {
            // ignore the next character
            offset += 2;
            continue;
        }
        if (s.s[offset] == c)
            break;
        offset++;
    }
    return offset;
}

int str_find_of(const_str_t & s, int offset, const char * cs) {
    for (;offset < s.length; offset++) {
        char c = s.s[offset];
        if (chr_of(c, cs))
            break;
    }
    return offset;
}

int str_find_not_of(const_str_t & s, int offset, const char * cs) {
    for (;offset < s.length; offset++) {
        char c = s.s[offset];
        if (!chr_of(c, cs))
            break;
    }
    return offset;
}

int str_skip_whitespaces(const_str_t & s, int offset) {
	return str_find_not_of(s, offset, white_spaces);
}

int json_error(const char * err) {
    printf("json_error: %s\n", err);
    return -1;
}

int json_skip_value(const_str_t & json, int offset) {
    offset = str_find_not_of(json, offset, white_spaces);
    if (offset >= json.length)
        return json_error("expected value");
    char c = json.s[offset];
    if (chr_of(c, "0123456789-")) { // skip number
        offset = str_find_not_of(json, ++offset, "0123456789.-eE");
        return offset;
    }
    if (c == '"') { // skip string
        offset = str_find_unescaped(json, ++offset, '"');
        if (offset >= json.length)
            return json_error("did not find end of string value");
        return ++offset;
    }
    if (c == '{') { // skip object
        offset = str_find_not_of(json, ++offset, white_spaces);
		if (offset >= json.length)
			return json_error("unexpected end of object");
        if (json.s[offset] == '}')
			return ++offset;
        do {
            // skip key
            offset = str_find_not_of(json, offset, white_spaces);
            if (offset >= json.length || json.s[offset] != '"')
                return json_error("expected key");
            offset = str_find_unescaped(json, ++offset, '"');
            if (offset >= json.length)
                return json_error("expected end of key");

            offset = str_find_not_of(json, ++offset, white_spaces);
            if (offset >= json.length || json.s[offset] != ':')
                return json_error("expected seperator");

            offset = json_skip_value(json, ++offset);
            if (offset == -1)
                return -1;
            offset = str_find_not_of(json, offset, white_spaces);
			if (offset >= json.length)
				return offset;
			if (json.s[offset] != ',')
				break;
        } while (++offset < json.length);
        if (json.s[offset] != '}')
            return json_error("expected tail of object '}'");
        return ++offset;
    }
    if (c == '[') { // skip array
		offset = str_find_not_of(json, ++offset, white_spaces);
		if (offset >= json.length)
			return json_error("unexpected end of array");
		if (json.s[offset] == ']')
			return ++offset;
		do {
			offset = json_skip_value(json, offset);
			if (offset == -1)
				return -1;
			offset = str_find_not_of(json, offset, white_spaces);
			if (offset >= json.length)
				return offset;
			if (json.s[offset] != ',')
				break;
		} while (++offset < json.length);
		if (json.s[offset] != ']')
			return json_error("expected tail of array ']'");
        return ++offset;
    }
	if (c == 't') {
		int remaining = json.length - offset;
		if (remaining >= 4 && strncmp(json.s + offset, "true", 4) == 0) {
			offset += 4;
			return offset;
		}
	}
	if (c == 'f') {
		int remaining = json.length - offset;
		if (remaining >= 5 && strncmp(json.s + offset, "false", 5) == 0) {
			offset += 5;
			return offset;
		}
	}
	if (c == 'n') {
		int remaining = json.length - offset;
		if (remaining >= 4 && strncmp(json.s + offset, "null", 4) == 0) {
			offset += 4;
			return offset;
		}
	}
    return json_error("unknown value type");
}

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

/*template<class T>
int json_object_parse(const_str_t & json, int offset, T on_item) {
    offset = str_find_not_of(json, offset, white_spaces);
	if (offset >= json.length || json.s[offset] != '{')
		return json_error("expected object");
    do {
        offset = str_find_not_of(json, ++offset, white_spaces);
        if (offset >= json.length || json.s[offset] != '"')
            return json_error("expected key");
        int key_start = ++offset;
        offset = str_find_unescaped(json, offset, '"');
        if (offset >= json.length)
            return json_error("expected end of key");
        int key_length = offset - key_start;

        offset = str_find_not_of(json, ++offset, white_spaces);
        if (offset >= json.length || json.s[offset] != ':')
            return json_error("expected seperator");

        offset = on_item(json, ++offset, key_start, key_length);
		if (offset == -1)
			return -1;

        offset = str_find_not_of(json, offset, white_spaces);
    } while (offset < json.length && json.s[offset] == ',');
    if (json.s[offset] != '}')
        return json_error("expected tail of object '}'");
    return ++offset;
}*/

int json_object_key_log(const_str_t & json, int offset, int key_start, int key_length) {
    printf("%.*s\n", key_length, json.s + key_start);

    offset = str_find_not_of(json, offset, white_spaces);
    if (offset >= json.length)
        return json_error("expected value");

    if (json.s[offset] == '{') {
        offset = json_object_parse(json, offset, json_object_key_log);
        return offset;
    }

    offset = json_skip_value(json, offset);
    return offset;
}

int json_int64_parse(const_str_t & json, int offset, int64_t & value) {
    offset = str_find_not_of(json, offset, white_spaces);
    if (offset >= json.length)
        return json_error("unexpected end before value");
    char c = json.s[offset];
    if (!chr_of(c, "0123456789-"))
		return json_error("did not find number");
	const char *start = json.s + offset;
	char *end;
	double d = strtod(start, &end);
	value = (int64_t)d;
	if (d != (double)value)
		return json_error("number not an integer");
	offset = (end - start) + offset;
	return offset;
}

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
            offset = (end - start) + offset;
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

int json_double_parse(const_str_t & json, int offset, double & value) {
    char c = json.s[offset];
    if (!chr_of(c, "0123456789-"))
		return json_error("did not find number");
	const char *start = json.s + offset;
	char *end;
	value = strtod(start, &end);
	offset = (end - start) + offset;
	return offset;
}

int json_float_parse(const_str_t & json, int offset, float & value) {
	double dvalue;
	offset = json_double_parse(json, offset, dvalue);
	value = dvalue;
	return offset;
}

int json_bool_parse(const_str_t & json, int offset, bool & value) {
	int remaining = json.length - offset;
	if (remaining < 4)
		return json_error("unexpected end parsing bool");
	char c = json.s[offset];
	if (c == 't' && strncmp(json.s + offset, "true", 4) == 0) {
		value = true;
		return offset + 4;
	}
	if (remaining < 5)
		return json_error("unexpected end parsing bool");
	if (c == 'f' && strncmp(json.s + offset, "false", 5) == 0) {
		value = false;
		return offset + 5;
	}
	return json_error("expected bool");
}

int json_string_parse(const_str_t & json, int offset, std::string & item) {
	if (json.s[offset] != '"')
		return json_error("expected string");
	if (++offset >= json.length)
		return json_error("unexpected end of string");
	int start = offset;
	offset = str_find_unescaped(json, offset, '"');
	if (offset >= json.length || json.s[offset] != '"')
		return json_error("unexpected end of string");
	item.assign(json.s + start, offset - start);
	return ++offset;
}

int json_string_array_parse(const_str_t & json, int offset, std::vector<std::string> & items) {
	offset = json_array_parse(json, offset, [&items]( const_str_t & json, int offset, int index ) {
		items.push_back("");
		return json_string_parse(json, offset, items.back());
	});
	return offset;
}

int json_int64_array_parse(const_str_t & json, int offset, std::vector<int64_t> & items) {
	offset = json_array_parse(json, offset, [&items]( const_str_t & json, int offset, int index ) {
		items.push_back(0);
		return json_int64_parse(json, offset, items.back());
	});
	return offset;
}

int json_float_array_parse(const_str_t & json, int offset, std::vector<float> & items) {
	offset = json_array_parse(json, offset, [&items]( const_str_t & json, int offset, int index ) {
		items.push_back(0);
		return json_float_parse(json, offset, items.back());
	});
	return offset;
}

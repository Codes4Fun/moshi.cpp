#pragma once

#include <sys/stat.h>

static void list_devices() {
    ggml_backend_load_all();
    auto dev_count = ggml_backend_dev_count();
    fprintf( stderr, "available devices:\n" );
    for (size_t i = 0; i < dev_count; i++) {
        auto dev = ggml_backend_dev_get( i );
        auto name = ggml_backend_dev_name( dev );
        fprintf( stderr, "  \"%s\"\n", name );
    }
    exit(1);
}

int find_last( const char * s, char c ) {
    int index = -1;
    for ( int i = 0; s[i]; ++i ) {
        if ( s[i] == c )
            index = i;
    }
    return index;
}

int find_last( const char * s, int size, char c ) {
    for ( int i = size - 1; i >= 0; --i ) {
        if ( s[i] == c )
            return i;
    }
    return -1;
}

int find_last( std::string s, char c ) {
    return find_last( s.c_str(), s.size(), c );
}

const char * get_ext( const char * filename ) {
    int index = find_last( filename, '.' );
    if ( index < 0 )
        return NULL;
    return filename + index;
}

bool is_abs_or_rel( std::string & path ) {
    auto size = path.size();
    if ( size < 1 )
        return false;
    if ( path[0] == '/' )
        return true; // absolute
    if ( path[0] != '.' )
        return false;
    if ( size < 2 ) // "."
        return true;
    if ( path[1] == '/' ) // "./"
        return true;
    if ( path[1] != '.' )
        return false;
    if ( size < 3 ) // ".."
        return true;
    return path[2] == '/'; // "../"
}

std::string get_program_path( const char * argv0 ) {
    std::string path;
    int index = find_last( argv0, '/' );
    if ( index >= 0 ) {
        path.assign( argv0, index+1 );
        return path;
    }
    // TODO: add support for windows?
    /*char filepath[4096];
    auto size = readlink( "/proc/self/exe", filepath, sizeof(filepath) - 1 );
    assert ( size != -1 && size != sizeof(filepath) - 1 );
    index = find_last( filepath, size, '/' );
    assert( index >= 0 );
    path.assign( filepath, index+1 );
    return path;*/
    return "./";
}

void unref( FILE * f ) {
    fclose( f );
}

int find_file(
        const char *filename,
        const std::vector<std::string> & search_dirs,
        std::string & result
) {
    // try filename directly first
    result = filename;
    if ( access( result.c_str(), F_OK | R_OK ) == 0 )
        return 0;

    if ( filename[0] == '/' ) // absolute path
        return -1;

    // try search paths next
    for ( auto & dir : search_dirs ) {
        result = dir + "/" + filename;
        if ( access( result.c_str(), F_OK | R_OK ) == 0 )
            return 0;
    }

    return -1;
}


void check_arg_path( std::string & path, bool & found_file, bool & found_dir ) {
    found_file = false;
    found_dir = false;

    if ( access( path.c_str(), F_OK | R_OK ) != 0 ) {
        return;
    }

    if ( path.ends_with("/") | path.ends_with("\\") ) {
        found_dir = true;
        return;
    }

    struct stat stats;
    if ( stat( path.c_str(), &stats ) != 0 ) {
        fprintf( stderr, "error: failed to stat %s\n", path.c_str() );
        exit(1);
    }

    found_dir = S_ISDIR(stats.st_mode);
    if ( ! found_dir ) {
        if ( is_abs_or_rel( path ) ) {
            fprintf( stderr, "error: failed to find file path: \"%s\"\n", path.c_str() );
            exit(1);
        }
        found_file = true;
    }
}

void ensure_path( std::string & path ) {
    auto path_size = path.size();
    if ( path_size > 1 && path[path_size - 1] != '/' ) {
        path += "/";
    }
}



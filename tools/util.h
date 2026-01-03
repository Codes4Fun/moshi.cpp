#pragma once

#include <sys/stat.h>

#ifndef S_ISDIR
#define S_ISDIR(m) (((m) & _S_IFMT) == _S_IFDIR)
#endif

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
    return find_last( s.c_str(), (int) s.size(), c );
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
#if _WIN32
    if ( size < 2 || path[1] == ':' )
        return true; // absolute
#endif
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
    return "./";
}

void unref( FILE * f ) {
    fclose( f );
}

bool file_exists( const char * filepath ) {
#if _WIN32
    struct __stat64 stats;
    return _stat64( filepath, &stats ) == 0;
#else
    struct stat stats;
    return stat( filepath, &stats ) == 0;
#endif
}

void check_arg_path( std::string & path, bool & found_file, bool & found_dir ) {
    found_file = false;
    found_dir = false;

#if _WIN32
    struct __stat64 stats;
    if ( _stat64( path.c_str(), &stats ) != 0 ) {
        return;
    }
#else
    struct stat stats;
    if ( stat( path.c_str(), &stats ) != 0 ) {
        return;
    }
#endif

    found_dir = S_ISDIR(stats.st_mode);
    if ( ! found_dir ) {
        found_file = true;
    }
}

void ensure_path( std::string & path ) {
    auto path_size = path.size();
    if ( path_size > 1 && path[path_size - 1] != '/' ) {
        path += "/";
    }
}



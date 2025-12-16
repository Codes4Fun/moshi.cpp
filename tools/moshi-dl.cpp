#include <stdio.h>
#include <assert.h>
#include <stdint.h>

#include <vector>
#include <string>
#include <sstream>
#include <iomanip>

#include <unistd.h>

#include <curl/curl.h>
#include <ggml.h>
#include <ggml-backend.h>
#include <moshi/json.h>
#include <moshi/ptrs.h>
#include "util.h"
#include "validate.h"

struct model_file_t {
    std::string filepath;
    int64_t size;
    std::string oid;
    std::string hash;
};

struct model_t {
    std::string host, path, rev;
    bool hidden; // not part of default or extras
    std::vector<model_file_t> defaults; // tools use by default
    std::vector<model_file_t> extras;
    std::vector<model_file_t> other; // unused or duplicate files
};

static void print_usage(const char * program) {
    fprintf( stderr, "usage: %s [option(s)] [org/model | org/model/filepath]\n", program );
    fprintf( stderr, "\nBy Default uses envar MODEL_CACHE or current directory,\n" );
    fprintf( stderr, "unless the model path option is used, and then if present\n" );
    fprintf( stderr, "validates existing model files or downloads missing default model files.\n" );
    fprintf( stderr, "\noption(s):\n" );
    fprintf( stderr, "  -h,       --help         show this help message\n" );
    fprintf( stderr, "  -m PATH,  --models PATH  root of model cache/storage.\n" );
    fprintf( stderr, "  -e,       --extras       include extra models.\n" );
    fprintf( stderr, "  -c,       --complete     include nonessential or duplicate files.\n" );
    fprintf( stderr, "  -v,       --validate     validate only.\n" );
    fprintf( stderr, "  -l,       --list         list selected models and files.\n");
    fprintf( stderr, "  -lm,      --list-models  list models only.\n");
    fprintf( stderr, "  -la,      --list-all     list include hidden models.\n");
    exit(1);
}

int parse_model_files( const_str_t & json, int offset, std::vector<model_file_t> & files ) {
    return json_array_parse(json, offset,
    [&files]( const_str_t & json, int offset, int index ) {
        files.push_back({});
        auto & file = files[index];
        return json_array_parse(json, offset,
        [&file]( const_str_t & json, int offset, int index ) {
            switch( index ) {
            case 0: return json_string_parse(json, offset, file.filepath);
            case 1: return json_int64_parse(json, offset, file.size);
            case 2: return json_string_parse(json, offset, file.oid);
            case 3: return json_string_parse(json, offset, file.hash);
            }
            fprintf( stderr, "error: unexpected value in file entry\n");
            return -1;
        });
    });
}

int get_models( std::vector<model_t> & models, const char * filename ) {
	unref_ptr<FILE> f = fopen( filename, "rb" );
    if ( ! f ) {
        fprintf( stderr, "error: failed to open %s\n", filename );
        return -1;
    }
    // get file length
	fseek( f, 0, SEEK_END );
	auto length = ftell( f );
	fseek( f, 0, SEEK_SET );
    if ( length < 1 ) {
        fprintf( stderr, "error: file was empty %s\n", filename );
        return -1;
    }
    // read file
	std::vector<char> raw( length );
    if ( fread(raw.data(), length, 1, f) != 1 ) {
        fprintf( stderr, "error: failed to read file %s\n", filename );
        return -1;
    }

	const_str_t json = {raw.data(), (int)length};

    int offset = str_skip_whitespaces(json, 0);
    if (offset >= length || json.s[offset] != '[') {
        fprintf( stderr, "error: did not find expected json array in %s\n", filename );
        return -1;
	}

    offset = json_array_parse(json, offset,
    [&models]( const_str_t & json, int offset, int index )
    {
        if ( index == 0 ) { // skip header
            return json_skip_value(json, offset);
        }
        models.push_back({});
        auto & model = models[index - 1];
        offset = json_array_parse(json, offset,
        [&model]( const_str_t & json, int offset, int index ) {
            switch ( index ) {
			case 0: return json_string_parse(json, offset, model.host);
			case 1: return json_string_parse(json, offset, model.path);
			case 2: return json_string_parse(json, offset, model.rev);
            case 3: return json_bool_parse(json, offset, model.hidden);
            case 4: return parse_model_files(json, offset, model.defaults);
            case 5: return parse_model_files(json, offset, model.extras);
            case 6: return parse_model_files(json, offset, model.other);
            }
            fprintf( stderr, "error: unexpected value in model entry\n" );
            return -1;
        });
        return offset;
    });

    return 0;
}

std::string build_hex_string( int bytes, const unsigned char* buffer ) {
    std::ostringstream oss;
    for ( int i = 0; i < bytes; ++i ) {
        oss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(buffer[i]);
    }
    return oss.str();
}

int get_hash( const char * filepath, std::string & hash, bool sha256 ) {
    unref_ptr<FILE> f = fopen( filepath, "rb" );
    if ( ! f )
        return -1;

	fseek( f, 0, SEEK_END );
	auto length = ftell( f );
	fseek( f, 0, SEEK_SET );

    if ( sha256 ) {
        validate_sha256<1024*1024> cs;
        cs.init();

        while( true ) {
            cs.bytes = fread( cs.buffer, 1, sizeof(cs.buffer), f );
            if ( ! cs.bytes )
                break;
            cs.update();
        }

        cs.final();
        hash = build_hex_string( cs.bytes, cs.buffer );
        printf("%s\n", hash.c_str());
    } else {
        validate_sha1<1024*1024> cs;
        cs.init();

        cs.bytes = snprintf( (char*)cs.buffer, sizeof(cs.buffer), "blob %ld", length ) + 1;
        cs.update();

        while( true ) {
            cs.bytes = fread( cs.buffer, 1, sizeof(cs.buffer), f );
            if ( ! cs.bytes )
                break;
            cs.update();
        }

        cs.final();
        hash = build_hex_string( cs.bytes, cs.buffer );
        printf("%s\n", hash.c_str());
    }

    return 0;
}

int main(int argc, char *argv[]) {
    const char * model_cache = getenv("MODEL_CACHE");
    std::string model_root = model_cache? model_cache : "";
    bool extras = false;
    bool other = false;
    bool validate_only = false;
    bool list = false;
    bool list_model_only = false;
    bool list_all = false;
    const char * target;

    //////////////////////
    // MARK: Parse Args
    //////////////////////

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
        }
        if (arg == "-m" || arg == "--model-root") {
            if (i + 1 >= argc) {
                fprintf( stderr, "error: \"%s\" requires path to models\n", argv[i] );
                exit(1);
            }
            model_root = argv[++i];
            continue;
        }
        if (arg == "-e" || arg == "--extras") {
            extras = true;
            continue;
        }
        if (arg == "-c" || arg == "--complete") {
            other = true;
            continue;
        }
        if (arg == "-v" || arg == "--validate") {
            validate_only = true;
            continue;
        }
        if (arg == "-l" || arg == "--list") {
            list = true;
            continue;
        }
        if (arg == "-lm" || arg == "--list-models") {
            list_model_only = true;
            continue;
        }
        if (arg == "-la" || arg == "--list-all") {
            list_all = true;
            continue;
        }
        if (arg[0] == '-') {
            fprintf( stderr, "error: unrecognized option \"%s\"\n", argv[i] );
            exit(1);
        }
        if ( ! target ) {
            target = argv[i];
            continue;
        }
        fprintf( stderr, "error: unexpected extra argument \"%s\"\n", argv[i] );
        exit(1);
    }

    ensure_path( model_root );

    std::string program_path = get_program_path(argv[0]);
    ensure_path( program_path );
    std::string models_json = program_path + "moshi-dl.json";

    std::vector<model_t> models;
    if ( get_models( models, models_json.c_str() ) < 0 ) {
        return -1;
    }

    if ( list_all || list ) {
        for ( auto & model : models ) {
            bool extra_only = ! model.defaults.size();
            if ( ! target && ! list_all && ! extras && extra_only )
                continue;
            if ( ! target && ! list_all && model.hidden )
                continue;
            if ( target && strcmp( target, model.path.c_str() ) != 0 )
                continue;
            printf("%s (%s)\n",
                model.path.c_str(),
                model.hidden? "hidden" :
                model.defaults.size()? "default" :
                model.extras.size()? "extra" : "other"
            );
            if ( list_model_only )
                continue;
            for ( auto & file : model.defaults ) {
                printf( "%s/%s %ld\n", model.path.c_str(), file.filepath.c_str(), file.size );
            }
            if ( ( target && extra_only ) || extras ) {
                for ( auto & file : model.extras ) {
                    printf( "%s/%s %ld\n", model.path.c_str(), file.filepath.c_str(), file.size );
                }
            }
            if ( other ) {
                for ( auto & file : model.other ) {
                    printf( "%s/%s %ld\n", model.path.c_str(), file.filepath.c_str(), file.size );
                }
            }
        }

        return 0;
    }

    if ( validate_only ) {
        for ( auto & model : models ) {
            bool extra_only = ! model.defaults.size();
            if ( ! target && ! list_all && ! extras && extra_only )
                continue;
            if ( ! target && ! list_all && model.hidden )
                continue;
            if ( target && strcmp( target, model.path.c_str() ) != 0 )
                continue;
            printf("(%s)  %s\n",
                model.hidden? "hidden" :
                model.defaults.size()? "default" :
                model.extras.size()? "extra" : "other",
                model.path.c_str()
            );
            if ( list_model_only )
                continue;
            std::string model_path = model_root + model.path + "/";
            for ( auto & file : model.defaults ) {
                printf( "%s/%s %ld\n", model.path.c_str(), file.filepath.c_str(), file.size );
                printf( "%s\n", file.oid.c_str() );
                std::string filepath = model_path + file.filepath;
                std::string hash;
                get_hash(filepath.c_str(), hash, file.oid.size() != 40);
                if ( file.oid != hash ) {
                    fprintf( stderr, "error: %s failed hash\n", filepath.c_str() );
                    exit(-1);
                }
            }
            if ( ( target && extra_only ) || extras ) {
                for ( auto & file : model.extras ) {
                    printf( "%s/%s %ld\n", model.path.c_str(), file.filepath.c_str(), file.size );
                    printf( "%s\n", file.oid.c_str() );
                    std::string filepath = model_path + file.filepath;
                    std::string hash;
                    get_hash(filepath.c_str(), hash, file.oid.size() != 40);
                    if ( file.oid != hash ) {
                        fprintf( stderr, "error: %s failed hash\n", filepath.c_str() );
                        exit(-1);
                    }
                }
            }
            if ( other ) {
                for ( auto & file : model.other ) {
                    printf( "%s/%s %ld\n", model.path.c_str(), file.filepath.c_str(), file.size );
                    printf( "%s\n", file.oid.c_str() );
                    std::string filepath = model_path + file.filepath;
                    std::string hash;
                    get_hash(filepath.c_str(), hash, file.oid.size() != 40);
                    if ( file.oid != hash ) {
                        fprintf( stderr, "error: %s failed hash\n", filepath.c_str() );
                        exit(-1);
                    }
                }
            }
        }

        return 0;
    }

    /////////////////////////
    // MARK: Validate Args
    /////////////////////////

    /*bool model_root_exists = access( model_root.c_str(), F_OK ) != 0;
    bool model_root_dir = false;
    if ( model_root_exists ) {
        if ( lstat( model_root.c_str(), &stats ) != 0 || ! S_ISDIR(stats.st_mode) ) {
        }
    }

    if ( validate_only && model_root.size() ) {
        if ( access( model_root.c_str() ) != 0 ) {
            fprintf( stderr, "error: model root does not exist %s\n", model_root.c_str() );
            exit(1);
        }
        struct stat stats;
        if ( lstat( model_root.c_str(), &stats ) != 0 || ! S_ISDIR(stats.st_mode) ) {
            fprintf( stderr, "error: model root is not a path %s\n", model_root.c_str() );
            exit(1);
        }
    }*/





}



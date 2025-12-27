
find_package(SDL2 REQUIRED)
find_package(FFmpeg REQUIRED COMPONENTS AVCODEC AVFILTER AVFORMAT AVUTIL SWRESAMPLE)

add_executable(mimi-encode mimi-encode.cpp)
target_link_libraries(mimi-encode PRIVATE moshi ${GGML_LIBRARIES} ${FFMPEG_LIBRARIES})
target_include_directories(mimi-encode PRIVATE ${GGML_INCLUDE_DIRS} ${SentencePiece_INCLUDE_DIRS} ${FFMPEG_INCLUDE_DIRS})
setup_rpath(mimi-encode)

add_executable(mimi-decode mimi-decode.cpp)
target_link_libraries(mimi-decode PRIVATE moshi ${GGML_LIBRARIES} ${FFMPEG_LIBRARIES})
target_include_directories(mimi-decode PRIVATE ${GGML_INCLUDE_DIRS} ${SentencePiece_INCLUDE_DIRS} ${FFMPEG_INCLUDE_DIRS})
setup_rpath(mimi-decode)

add_executable(mimi-play mimi-play.cpp)
target_link_libraries(mimi-play PRIVATE moshi ${GGML_LIBRARIES} SDL2::SDL2)
target_include_directories(mimi-play PRIVATE ${GGML_INCLUDE_DIRS} ${SentencePiece_INCLUDE_DIRS})
setup_rpath(mimi-play)

add_executable(mimi-echo mimi-echo.cpp)
target_link_libraries(mimi-echo PRIVATE moshi ${GGML_LIBRARIES} SDL2::SDL2)
target_include_directories(mimi-echo PRIVATE ${GGML_INCLUDE_DIRS} ${SentencePiece_INCLUDE_DIRS})
setup_rpath(mimi-echo)

add_executable(moshi-tts moshi-tts.cpp)
target_link_libraries(moshi-tts PRIVATE moshi ${GGML_LIBRARIES} ${FFMPEG_LIBRARIES} SDL2::SDL2)
target_include_directories(moshi-tts PRIVATE ${GGML_INCLUDE_DIRS} ${SentencePiece_INCLUDE_DIRS} ${FFMPEG_INCLUDE_DIRS})
setup_rpath(moshi-tts)

add_executable(moshi-stt moshi-stt.cpp)
target_link_libraries(moshi-stt PRIVATE moshi ${GGML_LIBRARIES} ${FFMPEG_LIBRARIES} SDL2::SDL2)
target_include_directories(moshi-stt PRIVATE ${GGML_INCLUDE_DIRS} ${SentencePiece_INCLUDE_DIRS} ${FFMPEG_INCLUDE_DIRS})
setup_rpath(moshi-stt)

add_executable(moshi-sts moshi-sts.cpp)
target_link_libraries(moshi-sts PRIVATE moshi ${GGML_LIBRARIES} ${FFMPEG_LIBRARIES} SDL2::SDL2)
target_include_directories(moshi-sts PRIVATE ${GGML_INCLUDE_DIRS} ${SentencePiece_INCLUDE_DIRS} ${FFMPEG_INCLUDE_DIRS})
set(CONFIG_FILE_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/moshi-config.json)
set(CONFIG_FILE_DEST $<TARGET_FILE_DIR:moshi-sts>)
add_custom_command(
    TARGET moshi-sts
    POST_BUILD        # This runs after the target is built
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CONFIG_FILE_SOURCE}
            ${CONFIG_FILE_DEST}
    COMMENT "Copying configuration file to build directory"
)
setup_rpath(moshi-sts)

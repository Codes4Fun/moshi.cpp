
find_package(SDL2 REQUIRED)
find_package(FFmpeg REQUIRED COMPONENTS AVCODEC AVFILTER AVFORMAT AVUTIL SWRESAMPLE)

add_executable(mimi-encode mimi-encode.cpp)
target_link_libraries(mimi-encode PRIVATE moshi ${GGML_LIBRARIES} ${FFMPEG_LIBRARIES} ${SentencePiece_LIBRARIES})
target_include_directories(mimi-encode PRIVATE ${GGML_INCLUDE_DIRS} ${SentencePiece_INCLUDE_DIRS} ${FFMPEG_INCLUDE_DIRS})
setup_rpath(mimi-encode)

add_executable(mimi-decode mimi-decode.cpp)
target_link_libraries(mimi-decode PRIVATE moshi ${GGML_LIBRARIES} ${FFMPEG_LIBRARIES} ${SentencePiece_LIBRARIES})
target_include_directories(mimi-decode PRIVATE ${GGML_INCLUDE_DIRS} ${SentencePiece_INCLUDE_DIRS} ${FFMPEG_INCLUDE_DIRS})
setup_rpath(mimi-decode)

add_executable(mimi-play mimi-play.cpp)
target_link_libraries(mimi-play PRIVATE moshi ${GGML_LIBRARIES} SDL2::SDL2 ${SentencePiece_LIBRARIES})
target_include_directories(mimi-play PRIVATE ${GGML_INCLUDE_DIRS} ${SentencePiece_INCLUDE_DIRS})
setup_rpath(mimi-play)

add_executable(mimi-echo mimi-echo.cpp)
target_link_libraries(mimi-echo PRIVATE moshi ${GGML_LIBRARIES} SDL2::SDL2 ${SentencePiece_LIBRARIES})
target_include_directories(mimi-echo PRIVATE ${GGML_INCLUDE_DIRS} ${SentencePiece_INCLUDE_DIRS})
setup_rpath(mimi-echo)

add_executable(moshi-tts moshi-tts.cpp)
target_link_libraries(moshi-tts PRIVATE moshi ${GGML_LIBRARIES} ${FFMPEG_LIBRARIES} SDL2::SDL2 ${SentencePiece_LIBRARIES})
target_include_directories(moshi-tts PRIVATE ${GGML_INCLUDE_DIRS} ${SentencePiece_INCLUDE_DIRS} ${FFMPEG_INCLUDE_DIRS})
setup_rpath(moshi-tts)

add_executable(moshi-stt moshi-stt.cpp)
target_link_libraries(moshi-stt PRIVATE moshi ${GGML_LIBRARIES} ${FFMPEG_LIBRARIES} SDL2::SDL2 ${SentencePiece_LIBRARIES})
target_include_directories(moshi-stt PRIVATE ${GGML_INCLUDE_DIRS} ${SentencePiece_INCLUDE_DIRS} ${FFMPEG_INCLUDE_DIRS})
setup_rpath(moshi-stt)

add_executable(moshi-sts moshi-sts.cpp)
target_link_libraries(moshi-sts PRIVATE moshi ${GGML_LIBRARIES} ${FFMPEG_LIBRARIES} SDL2::SDL2 ${SentencePiece_LIBRARIES})
target_include_directories(moshi-sts PRIVATE ${GGML_INCLUDE_DIRS} ${SentencePiece_INCLUDE_DIRS} ${FFMPEG_INCLUDE_DIRS})
set(CONFIG_FILES_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
file(GLOB CONFIG_FILES_TO_COPY
    RELATIVE ${CONFIG_FILES_SOURCE_DIR}
    "${CONFIG_FILES_SOURCE_DIR}/*-config.json"
)
set(CONFIG_FILES_DEST_DIR $<TARGET_FILE_DIR:moshi-sts>)
foreach(file ${CONFIG_FILES_TO_COPY})
    add_custom_command(
        TARGET moshi-sts
        POST_BUILD # Run this command after the target's main command
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CONFIG_FILES_SOURCE_DIR}/${file}"
                "${CONFIG_FILES_DEST_DIR}/${file}"
        COMMENT "Copying ${file} to ${CONFIG_FILES_DEST_DIR}"
    )
endforeach()
setup_rpath(moshi-sts)

set(ARIA2_FILES_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(ARIA2_FILES_DEST_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
file(GLOB ARIA2_FILES_TO_COPY
    RELATIVE ${ARIA2_FILES_SOURCE_DIR}
    "${ARIA2_FILES_SOURCE_DIR}/kyutai_*.txt"
    "${ARIA2_FILES_SOURCE_DIR}/Codes4Fun_*.txt"
    "moshi-defaults.txt"
)
add_custom_target(copy_aria2_files ALL
    COMMENT "Creating destination directory for text files"
)
foreach(file ${ARIA2_FILES_TO_COPY})
    add_custom_command(
        TARGET copy_aria2_files
        POST_BUILD # Run this command after the target's main command
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${ARIA2_FILES_SOURCE_DIR}/${file}"
                "${ARIA2_FILES_DEST_DIR}/${file}"
        COMMENT "Copying ${file} to ${ARIA2_FILES_DEST_DIR}"
    )
endforeach()

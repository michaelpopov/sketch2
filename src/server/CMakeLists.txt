add_executable(server main.cpp)
target_link_libraries(server PRIVATE cntrl stor comp membuf sketch_compile_options Threads::Threads)

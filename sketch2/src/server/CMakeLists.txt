add_executable(server main.cpp)
target_link_libraries(server PRIVATE cntrl stor comp membuf Threads::Threads)

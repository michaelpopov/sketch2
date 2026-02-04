# src

Various executables.
The number of exectuables can grow if "writer" and "indexer" will be defined as
standalone executables running in dedicated processes.
Control of distributed processing on multiple hosts might be added.

## core

Core libraries immplementing base functionality.

## tester

Driver for integration tests. Basically it's a "server" with built-in "client".

## server

Database daemon with all related functionality. Network support.

## client

CLI client to connect to the server and send commands.

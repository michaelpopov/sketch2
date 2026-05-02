CXX      = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra
INCLUDES = -Illama-include
LIBS     = -ldl

TARGET = emgen
SRCS   = main.cpp llama-link.cpp

$(TARGET): $(SRCS) llama-link.h .includes.stamp .libs.stamp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $(SRCS) $(LIBS)

# Populate llama-include from the llama-cpp source tree.
# Re-runs if the script itself changes.
.includes.stamp: get_includes.sh
	./get_includes.sh
	@touch $@

# Populate llama-libs from the llama-cpp build output.
# Re-runs if the script itself changes.
.libs.stamp: get_libs.sh
	./get_libs.sh
	@touch $@

clean:
	rm -f $(TARGET) .includes.stamp .libs.stamp

distclean: clean
	rm -rf llama-include llama-libs

.PHONY: clean distclean

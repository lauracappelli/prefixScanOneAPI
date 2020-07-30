CXX = dpcpp
CXXFLAGS = -O2 -g -std=c++17

USM_EXE_NAME = prefixScan_t
USM_SOURCES = src/prefixScan_t.cpp

all: build_usm
	
build_usm:
	$(CXX) $(CXXFLAGS) -o $(USM_EXE_NAME) $(USM_SOURCES)

run:
	./$(USM_EXE_NAME)

clean:
	rm -rf $(USM_EXE_NAME)
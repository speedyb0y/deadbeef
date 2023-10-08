CXX?=clang++
CXXFLAGS?=$(CFLAGS) -msse3 -DNDEBUG
OUT=ddb_dsp_libretro

all:
	$(CXX) $(CXXFLAGS) -shared -O2 -o $(OUT).so libretro.cpp -fPIC -Wall -march=native
debug: CXXFLAGS += -g
debug: all

clean:
	rm -f $(OUT).so

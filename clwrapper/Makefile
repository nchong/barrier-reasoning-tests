LIB = libclwrapper.so
default: $(LIB)

include ../Makefile.common
LOG_LEVEL=LOG_INFO
CXXFLAGS += -fPIC
CXXFLAGS += -DLOG_LEVEL=$(LOG_LEVEL)

OBJS = log.o clerror.o clwrapper.o

$(LIB): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OPENCL_LIB) $(OPENCL_INC) $(SHARED) -o $@ $^

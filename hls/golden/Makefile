# Makefile for GFE_Net

TARGET = GFENet_FPGA

KERNEL_DIR = kernel
TESTBENCH_DIR = testbench

INC_PATH = -I.

LIB_PATH = -L.

LIBS = 

SRCS = $(TESTBENCH_DIR)/main.cpp \
	   $(TESTBENCH_DIR)/preProcess.cpp \
	   $(TESTBENCH_DIR)/netConfig.cpp \
	   $(KERNEL_DIR)/DFENet.cpp \
	   $(TESTBENCH_DIR)/postProcess.cpp \

OBJ_DIR = build

OBJS = $(addprefix $(OBJ_DIR)/, $(subst .cpp,.o, $(SRCS)))
DEPS = $(addprefix $(OBJ_DIR)/, $(subst .cpp,.d, $(SRCS)))

UNAME := $(shell uname)

ifeq ($(UNAME), Linux)
	CXX = g++
endif

# for debug
CXXFLAGS = -Wall -g -std=c++11 -MMD $(INC_PATH) -DDEBUG
# for release
# CXXFLAGS = -w -O3 -std=c++11 -MMD $(INC_PATH) -DNDEBUG

LDXXFLAGS = $(LIB_PATH) $(LIBS)

RM = rm -f

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) -o $@ $^ $(LDXXFLAGS)

$(OBJ_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -o $@ -c $<

.PHONY: clean distclean

clean:
	$(RM) $(OBJS)

distclean:
	$(RM) $(OBJS) $(TARGET) $(DEPS)

-include $(DEPS)

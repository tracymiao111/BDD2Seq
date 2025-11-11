# Makefile for compiling dagDumpSingle.c

CUDD_PATH = /root/cudd
CUDD_INCLUDE = $(CUDD_PATH) $(CUDD_PATH)/cudd $(CUDD_PATH)/nanotrav $(CUDD_PATH)/st $(CUDD_PATH)/mtr $(CUDD_PATH)/epd $(CUDD_PATH)/util
CUDD_LIB = $(CUDD_PATH)/cudd/.libs

# Compilation command
CC = gcc
CFLAGS = $(foreach dir, $(CUDD_INCLUDE), -I$(dir)) -L$(CUDD_LIB) -lcudd -lm
TARGET = optimalAlgorithm
SRCS = optimalAlgorithm.c $(CUDD_PATH)/nanotrav/bnet.c

$(TARGET): $(SRCS)
	$(CC) -o $(TARGET) $(SRCS) $(CFLAGS)

.PHONY: clean
clean:
	rm -f $(TARGET)
	rm -f *.dot

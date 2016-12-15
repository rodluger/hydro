# -*- makefile -*-

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
GCC_FLAGS1 = -fPIC -Wl,-Bsymbolic-functions -c -O3
GCC_FLAGS2 = -shared -O3 -Wl,-Bsymbolic-functions,-soname,chydro.so
else ifeq ($(UNAME_S),Darwin)
GCC_FLAGS1 = -fPIC -c
GCC_FLAGS2 = -shared -Wl,-install_name,chydro.so
else
$(error Unsupported opperating system. You will have to manually compile the chydro library)
endif

GCC = gcc

.PHONY: all
.SILENT: all

all:
	echo "[hydro] Compiling C source code..."
	${GCC} ${GCC_FLAGS1} laxfried/chydro.c
	${GCC} ${GCC_FLAGS1} laxfried/integrate.c
	${GCC} ${GCC_FLAGS1} laxfried/functions.c
	echo "[hydro] Generating shared library..."
	gcc ${GCC_FLAGS2} -o chydro.so chydro.o integrate.o functions.o -lc
	rm chydro.o
	rm integrate.o
	rm functions.o
	mv chydro.so laxfried/chydro.so
	echo "[hydro] Install successful."
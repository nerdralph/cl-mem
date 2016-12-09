# Change this path if the SDK was installed in a non-standard location
OPENCL_HEADERS = "/opt/AMDAPPSDK-3.0/include"
# By default libOpenCL.so is searched in default system locations, this path
# lets you adds one more directory to the search path.
LIBOPENCL = "/opt/amdgpu-pro/lib/x86_64-linux-gnu"

CC = gcc -O2 -flto
CPPFLAGS = -std=gnu99 -pedantic -Wextra -Wall -ggdb \
    -Wno-deprecated-declarations \
    -Wno-overlength-strings \
    -I${OPENCL_HEADERS}
LDFLAGS = -L${LIBOPENCL}
LDLIBS = -lOpenCL
OBJ = main.o
INCLUDES = config.h _kernel.h
EXE = cl-mem

all : ${EXE}

${EXE} : ${OBJ}
	${CC} -o $@ ${OBJ} ${LDFLAGS} ${LDLIBS}

${OBJ} : ${INCLUDES}

_kernel.h : input.cl config.h  
	echo 'const char *ocl_code = R"_mrb_(' >$@
	cpp $< >>$@
	echo ')_mrb_";' >>$@

clean :
	rm -f ${EXE} _kernel.h *.o _temp_*

re : clean all

.cpp.o :
	${CC} ${CPPFLAGS} -o $@ -c $<

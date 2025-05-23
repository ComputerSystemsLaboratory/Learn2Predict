DPU_DIR := dpu
HOST_DIR := host
BUILDDIR ?= bin
NR_DPUS ?= 1
NR_TASKLETS ?= 16
ENERGY ?= 0
PASSES ?= -O2

define conf_filename
	${BUILDDIR}/.NR_DPUS_$(1)_NR_TASKLETS_$(2).conf
endef
CONF := $(call conf_filename,${NR_DPUS},${NR_TASKLETS})

HOST_TARGET := ${BUILDDIR}/host_code
DPU_TARGET := ${BUILDDIR}/dpu_code

COMMON_INCLUDES := support
HOST_SOURCES := $(wildcard ${HOST_DIR}/*.c)
DPU_SOURCES := $(wildcard ${DPU_DIR}/*.c)

.PHONY: all clean test

__dirs := $(shell mkdir -p ${BUILDDIR})

COMMON_FLAGS := -w -I${COMMON_INCLUDES}
HOST_FLAGS := ${COMMON_FLAGS} -std=c11 -O3 `dpu-pkg-config --cflags --libs dpu` -DNR_TASKLETS=${NR_TASKLETS} -DNR_DPUS=${NR_DPUS} -DENERGY=${ENERGY} 
DPU_FLAGS := ${COMMON_FLAGS} -DNR_TASKLETS=${NR_TASKLETS}
DPU_FLAGS1 := ${COMMON_FLAGS} -DNR_TASKLETS=${NR_TASKLETS} -S -Xclang -disable-O0-optnone -emit-llvm
DPU_FLAGS2 := -DNR_TASKLETS=${NR_TASKLETS}

all: ${HOST_TARGET} ${DPU_TARGET}

${CONF}:
	@$(RM) $(call conf_filename,*,*)
	@touch ${CONF}

${HOST_TARGET}: ${HOST_SOURCES} ${COMMON_INCLUDES} ${CONF}
	@$(CC) -o $@ ${HOST_SOURCES} ${HOST_FLAGS}

${DPU_TARGET}: ${DPU_SOURCES} ${COMMON_INCLUDES} ${CONF}
	@dpu-upmem-dpurte-clang -E ${DPU_FLAGS} -o $@_p.c ${DPU_SOURCES}
	@dpu-upmem-dpurte-clang ${DPU_FLAGS1} -o $@.ll ${DPU_SOURCES}
	@opt -S ${PASSES} -o $@_opt.ll $@.ll
	@dpu-upmem-dpurte-clang ${DPU_FLAGS2} -o $@ $@_opt.ll

clean:
	@$(RM) -r $(BUILDDIR)

test: all
	./${HOST_TARGET}

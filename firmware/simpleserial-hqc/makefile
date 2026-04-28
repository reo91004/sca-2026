# Hey Emacs, this is a -*- makefile -*-
# ----------------------------------------------------------------------------
# SimpleSerial-HQC firmware for ChipWhisperer-Lite + CW308T-STM32F4 (STM32F415)
#
# Build (default custom RM-only):
#   make PLATFORM=CW308_STM32F4
#
# Build PQClean code_encode (RS+RM) variant:
#   make PLATFORM=CW308_STM32F4 HQC_IMPL=pqclean
#
# HQC sources are taken from $(PROJ_ROOT)/lib/crypto_kem/hqc128/$(HQC_DIR)/
# HQC headers from        $(PROJ_ROOT)/include/crypto_kem/hqc128/$(HQC_DIR)/
# ----------------------------------------------------------------------------

# Output base name -> simpleserial-hqc-CW308_STM32F4.{hex,bin,elf,...}
TARGET = simpleserial-hqc

# sca-2026/ project root and the ChipWhisperer firmware tree
PROJ_ROOT  ?= $(abspath $(CURDIR)/../..)
CW_FW_PATH ?= /home/reo/Documents/Repository/chipwhisperer/firmware/mcu

# HQC implementation: custom (default) or pqclean
HQC_IMPL ?= custom

ifeq ($(HQC_IMPL),custom)
  HQC_DIR  := hqc
  SRC      += simpleserial-hqc.c
  HQC_SRCS := reed_muller.c reed_solomon.c fft.c gf.c crypto_memset.c
else ifeq ($(HQC_IMPL),pqclean)
  HQC_DIR  := pqclean
  SRC      += simpleserial-hqc-pqclean.c
  HQC_SRCS := code.c reed_muller.c reed_solomon.c fft.c gf.c
else
  $(error Unknown HQC_IMPL=$(HQC_IMPL). Use custom or pqclean)
endif

# Pull HQC sources from project lib/ via VPATH (SRC stays as bare filenames so
# ChipWhisperer's OBJ = $(SRC:%.c=$(OBJDIR)/%.o) rule keeps producing flat objs).
HQC_SRC_DIR := $(PROJ_ROOT)/lib/crypto_kem/hqc128/$(HQC_DIR)
VPATH       += $(HQC_SRC_DIR)
SRC         += $(HQC_SRCS)

# Headers
EXTRAINCDIRS += $(PROJ_ROOT)/include
EXTRAINCDIRS += $(PROJ_ROOT)/include/crypto_kem/hqc128
EXTRAINCDIRS += $(PROJ_ROOT)/include/crypto_kem/hqc128/$(HQC_DIR)

# No ChipWhisperer-bundled crypto needed; HQC ships its own SHAKE/AES/RNG glue.
CRYPTO_TARGET  = NONE
CRYPTO_OPTIONS =

# SimpleSerial protocol version (matches reference HQC firmware default).
# Override with SS_VER=SS_VER_2_1 for v2 framing if needed.
SS_VER ?= SS_VER_1_1

# Optimization
OPT = 3

include $(CW_FW_PATH)/simpleserial/Makefile.simpleserial

FIRMWAREPATH = $(CW_FW_PATH)
include $(FIRMWAREPATH)/Makefile.inc

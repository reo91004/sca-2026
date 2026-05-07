# Hey Emacs, this is a -*- makefile -*-
# ----------------------------------------------------------------------------
# SimpleSerial-NTRU+ firmware for ChipWhisperer-Lite + CW308T-STM32F4 (STM32F415)
#
# Build (default level 576) :
#   make PLATFORM=CW308_STM32F4
#
# Build other security levels :
#   make PLATFORM=CW308_STM32F4 NTRUPLUS_LEVEL=768
#   make PLATFORM=CW308_STM32F4 NTRUPLUS_LEVEL=864
#   make PLATFORM=CW308_STM32F4 NTRUPLUS_LEVEL=1152
# ----------------------------------------------------------------------------

# Output base name -> simpleserial-ntruplus-CW308_STM32F4.{hex,bin,elf,...}
TARGET = simpleserial-ntruplus

# sca-2026/ project root and the ChipWhisperer firmware tree
PROJ_ROOT  ?= $(abspath $(CURDIR)/../..)
CW_FW_PATH ?= /home/pacl/Documents/Repository/chipwhisperer/firmware/mcu

# NTRU+ security level : 576 | 768 | 864 | 1152
NTRUPLUS_LEVEL  ?= 576

# Whitelist - reject anything else early so a typo doesn't silently link a
# nonexistent archive.
ifeq ($(filter $(NTRUPLUS_LEVEL),576 768 864 1152),)
$(error NTRUPLUS_LEVEL must be one of 576 768 864 1152, got '$(NTRUPLUS_LEVEL)')
endif

NTRUPLUS_ARCHIVE := $(PROJ_ROOT)/lib/crypto_kem/ntruplus$(NTRUPLUS_LEVEL).a

# Firmware glue (NTRU+ core comes from the static archive)
SRC += simpleserial-ntruplus.c
SRC += ntruplus_rng.c

# Per-level NTRU+ headers + the project-wide include/
EXTRAINCDIRS += $(PROJ_ROOT)/include
EXTRAINCDIRS += $(PROJ_ROOT)/include/crypto_kem/ntruplus$(NTRUPLUS_LEVEL)

# NTRU+ archive is built with -mfpu=fpv4-sp-d16 -mfloat-abi=hard (VFPv4-D16 +
# VFP-args, same tags as smaug archives). Match it on our side or the linker
# will refuse to combine the objects.
FPUUSE = 1

# Skip ChipWhisperer's bundled crypto - NTRU+ ships its own AES/SHA/SHAKE/Keccak.
CRYPTO_TARGET  = NONE
CRYPTO_OPTIONS =

# SimpleSerial protocol version.
#   SS_VER_1_1 (default) : ASCII hex framing, callback (uint8_t*, uint8_t),
#                          host pairs with cw.targets.SimpleSerial.
# Pinned to v1_1 to match HQC and SMAUG-T reference firmware so behavior can be
# diagnosed apples-to-apples between schemes.
SS_VER ?= SS_VER_1_1

# Hand the NTRU+ archive to the linker. GNU ld scans archives only for currently
# undefined symbols, so ntruplus_rng.o (defining rng_get_random_blocking) must
# already be on the link line - which it is, since it lives in $(SRC) above.
LDFLAGS += $(NTRUPLUS_ARCHIVE)

include $(CW_FW_PATH)/simpleserial/Makefile.simpleserial

FIRMWAREPATH = $(CW_FW_PATH)
include $(FIRMWAREPATH)/Makefile.inc

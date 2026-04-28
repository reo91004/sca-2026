# Hey Emacs, this is a -*- makefile -*-
# ----------------------------------------------------------------------------
# SimpleSerial-SMAUG firmware for ChipWhisperer-Lite + CW308T-STM32F4 (STM32F415)
#
# Build (default level 1):
#   make PLATFORM=CW308_STM32F4
#
# Build other security levels:
#   make PLATFORM=CW308_STM32F4 SMAUG_LEVEL=3
#   make PLATFORM=CW308_STM32F4 SMAUG_LEVEL=5
# ----------------------------------------------------------------------------

# Output base name -> simpleserial-smaug-CW308_STM32F4.{hex,bin,elf,...}
TARGET = simpleserial-smaug

# sca-2026/ project root and the ChipWhisperer firmware tree
PROJ_ROOT  ?= $(abspath $(CURDIR)/../..)
CW_FW_PATH ?= /home/pacl/Documents/Repository/chipwhisperer/firmware/mcu

# SMAUG-T security level: 1, 3, or 5
SMAUG_LEVEL   ?= 1
SMAUG_ARCHIVE := $(PROJ_ROOT)/lib/crypto_kem/smaug$(SMAUG_LEVEL).a

# Firmware glue (SMAUG-T core comes from the static archive)
SRC += simpleserial-smaug.c
SRC += smaug_rng.c

# Per-level SMAUG-T headers + the project-wide include/
EXTRAINCDIRS += $(PROJ_ROOT)/include
EXTRAINCDIRS += $(PROJ_ROOT)/include/crypto_kem/smaug$(SMAUG_LEVEL)

# SMAUG-T archive is built with -mfpu=fpv4-sp-d16 -mfloat-abi=hard.
# Match it on our side or the linker will refuse to combine the objects.
FPUUSE = 1

# Skip ChipWhisperer's bundled crypto - SMAUG-T ships its own AES/SHA/SHAKE/Keccak.
CRYPTO_TARGET  = NONE
CRYPTO_OPTIONS =

# SimpleSerial protocol version.
#   SS_VER_1_1 (default): ASCII hex framing, callback (uint8_t*, uint8_t),
#                          host pairs with cw.targets.SimpleSerial.
#   SS_VER_2_1          : binary framing, callback (cmd,scmd,len,buf),
#                          host pairs with cw.targets.SimpleSerial2.
# Pinned to v1_1 to match the HQC reference firmware so 0x44 ack behavior
# can be diagnosed apples-to-apples between SMAUG-T and HQC.
SS_VER ?= SS_VER_1_1

# Hand the SMAUG-T archive to the linker. GNU ld scans archives only for currently
# undefined symbols, so smaug_rng.o (defining rng_get_random_blocking) must already
# be on the link line - which it is, since it lives in $(SRC) above.
LDFLAGS += $(SMAUG_ARCHIVE)

include $(CW_FW_PATH)/simpleserial/Makefile.simpleserial

FIRMWAREPATH = $(CW_FW_PATH)
include $(FIRMWAREPATH)/Makefile.inc

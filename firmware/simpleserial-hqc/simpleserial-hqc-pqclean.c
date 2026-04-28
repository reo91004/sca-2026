/*
    HQC PQClean code_encode for side-channel analysis
    Uses PQCLEAN_HQC128_CLEAN implementation directly.

    Commands:
      'm' + offset + data: Set message chunk (46-byte RM input)
      'b' + 1 byte: Set single byte message
      'e' + 1 byte: Encode single byte RM (returns 16 bytes)
      't' + 1 byte: Encode full message RM with per-byte triggers (46 pulses)
      'i' + 16 bytes: Set 16-byte original message (RS input)
      'c' + 1 byte: PQCLEAN_HQC128_CLEAN_code_encode (single trigger)
      'd' + 1 byte: code_encode with per-byte RM triggers (46 pulses)
      'p' + data: Encode full message RM (single trigger)
      'g' + chunk: Get codeword chunk
      'x' + 1 byte: Reset
*/

#include "hal.h"
#include "parameters.h"
#include "reed_muller.h"
#include "reed_solomon.h"
#include "code.h"
#include "simpleserial.h"
#include <stdint.h>
#include <string.h>

#define MULTIPLICITY 3  /* CEIL_DIVIDE(384, 128) */

/* Storage */
static uint8_t msg[PARAM_N1];                      /* 46-byte message (RM input) */
static uint64_t codeword[PARAM_N1 * MULTIPLICITY * 2]; /* 2208 bytes */

/* Single byte encode */
static uint8_t single_msg;
static uint64_t single_codeword[2];  /* 128 bits = 16 bytes */

/* code_encode: 16-byte original message (RS input) */
static uint8_t original_msg[PARAM_K];  /* 16 bytes */

/*
 * Set message chunk (46-byte RM input)
 * Command 'm' + [offset] + [data bytes]
 */
uint8_t set_msg(uint8_t *data, uint8_t len) {
    if (len < 2)
        return 0x01;
    uint8_t offset = data[0];
    uint8_t copy_len = len - 1;
    if (offset + copy_len > PARAM_N1)
        copy_len = PARAM_N1 - offset;
    memcpy(&msg[offset], &data[1], copy_len);
    return 0x00;
}

/*
 * Set single byte message
 * Command 'b' + [1 byte]
 */
uint8_t set_single_byte(uint8_t *data, uint8_t len) {
    single_msg = data[0];
    return 0x00;
}

/*
 * Single byte RM encode with trigger
 * Command 'e' + [1 dummy byte]
 */
uint8_t encode_single(uint8_t *data, uint8_t len) {
    trigger_high();
    encode(single_codeword, single_msg);
    trigger_low();
    simpleserial_put('r', 16, (uint8_t *)single_codeword);
    return 0x00;
}

/*
 * Full 46-byte RM encode with single trigger
 * Command 'p' + [optional first byte]
 */
uint8_t encode_full(uint8_t *data, uint8_t len) {
    if (len > 0)
        msg[0] = data[0];
    trigger_high();
    PQCLEAN_HQC128_CLEAN_reed_muller_encode(codeword, msg);
    trigger_low();
    simpleserial_put('r', 16, (uint8_t *)codeword);
    return 0x00;
}

/*
 * Full 46-byte RM encode with per-byte triggers (46 pulses)
 * Command 't' + [1 dummy byte]
 */
uint8_t encode_full_triggered(uint8_t *data, uint8_t len) {
    for (size_t i = 0; i < PARAM_N1; i++) {
        trigger_high();
        encode(&codeword[2 * i * MULTIPLICITY], msg[i]);
        trigger_low();
        for (size_t copy = 1; copy < MULTIPLICITY; copy++) {
            memcpy(&codeword[2 * i * MULTIPLICITY + 2 * copy],
                   &codeword[2 * i * MULTIPLICITY], 16);
        }
    }
    simpleserial_put('r', 16, (uint8_t *)codeword);
    return 0x00;
}

/*
 * Set 16-byte original message for code_encode (RS input)
 * Command 'i' + [16 bytes]
 */
uint8_t set_original_msg(uint8_t *data, uint8_t len) {
    memcpy(original_msg, data, PARAM_K);
    return 0x00;
}

/*
 * PQCLEAN_HQC128_CLEAN_code_encode: RS+RM with single trigger
 * Command 'c' + [1 dummy byte]
 * This calls the exact PQClean code_encode() inside the trigger.
 */
uint8_t encode_code(uint8_t *data, uint8_t len) {
    trigger_high();
    PQCLEAN_HQC128_CLEAN_code_encode(codeword, original_msg);
    trigger_low();
    simpleserial_put('r', 16, (uint8_t *)codeword);
    return 0x00;
}

/*
 * code_encode with per-byte RM triggers (46 pulses)
 * Command 'd' + [1 dummy byte]
 * RS encode outside trigger, RM encode with per-byte triggers.
 */
uint8_t encode_code_triggered(uint8_t *data, uint8_t len) {
    /* RS encode: 16B -> 46B (outside trigger) */
    memset(msg, 0, PARAM_N1);
    PQCLEAN_HQC128_CLEAN_reed_solomon_encode(msg, original_msg);

    /* RM encode with per-byte triggers (46 pulses) */
    for (size_t i = 0; i < PARAM_N1; i++) {
        trigger_high();
        encode(&codeword[2 * i * MULTIPLICITY], msg[i]);
        trigger_low();
        for (size_t copy = 1; copy < MULTIPLICITY; copy++) {
            memcpy(&codeword[2 * i * MULTIPLICITY + 2 * copy],
                   &codeword[2 * i * MULTIPLICITY], 16);
        }
    }
    simpleserial_put('r', 16, (uint8_t *)codeword);
    return 0x00;
}

/*
 * Get codeword chunk
 * Command 'g' + [chunk_index]
 */
uint8_t get_codeword(uint8_t *data, uint8_t len) {
    uint8_t chunk = data[0];
    uint16_t offset = chunk * 16;
    if (offset + 16 > sizeof(codeword))
        return 0x01;
    simpleserial_put('r', 16, ((uint8_t *)codeword) + offset);
    return 0x00;
}

/*
 * Reset all buffers
 * Command 'x' + [1 dummy byte]
 */
uint8_t reset(uint8_t *data, uint8_t len) {
    memset(msg, 0, PARAM_N1);
    memset(codeword, 0, sizeof(codeword));
    single_msg = 0;
    memset(single_codeword, 0, sizeof(single_codeword));
    memset(original_msg, 0, PARAM_K);
    return 0x00;
}

int main(void) {
    platform_init();
    init_uart();
    trigger_setup();

    memset(msg, 0, PARAM_N1);
    memset(codeword, 0, sizeof(codeword));

    simpleserial_init();

    /* RM encode commands (compatible with custom firmware) */
    simpleserial_addcmd('m', 16, set_msg);
    simpleserial_addcmd('b', 1, set_single_byte);
    simpleserial_addcmd('e', 1, encode_single);
    simpleserial_addcmd('p', 16, encode_full);
    simpleserial_addcmd('t', 1, encode_full_triggered);
    simpleserial_addcmd('g', 1, get_codeword);
    simpleserial_addcmd('x', 1, reset);

    /* code_encode commands (PQClean-specific) */
    simpleserial_addcmd('i', 16, set_original_msg);
    simpleserial_addcmd('c', 1, encode_code);
    simpleserial_addcmd('d', 1, encode_code_triggered);

    while (1)
        simpleserial_get();
}

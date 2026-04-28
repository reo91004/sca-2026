/*
    HQC Reed-Muller Encode implementation for side-channel analysis

    Commands:
      'm' + offset + data: Set message chunk
      'b' + 1 byte: Set single byte message
      'k' + 1 byte: Set mask for masked encode
      'e' + 1 byte: Encode single byte (returns 16 bytes)
      'a' + 1 byte: Masked encode (returns 32 bytes: encode(m^r) | encode(r))
      'p' + data: Encode full message (single trigger)
      't' + 1 byte: Encode full message with per-byte triggers (46 trigger pulses)
      'g' + chunk: Get codeword chunk
      'x' + 1 byte: Reset
*/

#include "hal.h"
#include "hqc/data_structures.h"
#include "hqc/parameters.h"
#include "hqc/reed_muller.h"
#include "simpleserial.h"
#include <stdint.h>
#include <string.h>

// MULTIPLICITY = CEIL_DIVIDE(PARAM_N2, 128) = CEIL_DIVIDE(384, 128) = 3
#define MULTIPLICITY CEIL_DIVIDE(PARAM_N2, 128)

// Storage for message and codeword
// VEC_N1_SIZE_BYTES = 46 bytes for the message
// Output: 46 * 3 * 2 = 276 uint64_t values (2208 bytes)
static uint8_t msg[VEC_N1_SIZE_BYTES];
static uint64_t codeword[VEC_N1_SIZE_BYTES * MULTIPLICITY * 2];

// Single byte encode for quick test
static uint8_t single_msg;
static uint8_t single_mask;
static rm_codeword_t single_codeword;    // 128 bits = rm_codeword_t
static uint64_t single_codeword_mask[2]; // encode(r) for masked mode

/**
 * @brief Set message data (up to 16 bytes at a time due to simpleserial limit)
 * Command 'm' with offset in first byte, then data
 * Example: 'm' + [offset] + [data bytes]
 */
uint8_t set_msg(uint8_t *data, uint8_t len) {
    if (len < 2)
        return 0x01; // Need at least offset + 1 byte

    uint8_t offset = data[0];
    uint8_t copy_len = len - 1;

    if (offset + copy_len > VEC_N1_SIZE_BYTES) {
        copy_len = VEC_N1_SIZE_BYTES - offset;
    }

    memcpy(&msg[offset], &data[1], copy_len);
    return 0x00;
}

/**
 * @brief Set single byte message for quick encode test
 * Command 'b' + [1 byte message]
 */
uint8_t set_single_byte(uint8_t *data, uint8_t len) {
    single_msg = data[0];
    return 0x00;
}

/**
 * @brief Perform single byte Reed-Muller encode with trigger
 * Command 'e' (no data needed)
 * This encodes just one byte - good for detailed trace analysis
 */
uint8_t encode_single(uint8_t *data, uint8_t len) {
    trigger_high();
    encode(&single_codeword, single_msg);
    trigger_low();

    // Return 16 bytes of codeword (128 bits)
    simpleserial_put('r', 16, single_codeword.u8);
    return 0x00;
}

/**
 * @brief Perform full Reed-Muller encode with trigger
 * Command 'p' + [message byte to set before encode, optional]
 * Encodes full 46-byte message
 */
uint8_t encode_full(uint8_t *data, uint8_t len) {
    // Optionally set first byte of message
    if (len > 0) {
        msg[0] = data[0];
    }

    trigger_high();
    reed_muller_encode(codeword, (const uint64_t *)msg);
    trigger_low();

    // Return first 16 bytes of encoded codeword
    simpleserial_put('r', 16, (uint8_t *)codeword);
    return 0x00;
}

/**
 * @brief Perform full Reed-Muller encode with per-byte triggers
 * Command 't' + [dummy byte]
 * Encodes full 46-byte message with trigger pulse around each byte's encode()
 * This creates 46 separate trigger pulses for side-channel capture
 */
uint8_t encode_full_triggered(uint8_t *data, uint8_t len) {
    uint8_t *message_array = msg;
    rm_codeword_t *codeArray = (rm_codeword_t *)codeword;

    for (size_t i = 0; i < VEC_N1_SIZE_BYTES; i++) {
        int32_t pos = i * MULTIPLICITY;

        // Trigger around each encode() call
        trigger_high();
        encode(&codeArray[pos], message_array[i]);
        trigger_low();

        // Copy to other identical codewords (no trigger needed)
        for (size_t copy = 1; copy < MULTIPLICITY; copy++) {
            memcpy(&codeArray[pos + copy], &codeArray[pos], sizeof(rm_codeword_t));
        }
    }

    // Return first 16 bytes of encoded codeword
    simpleserial_put('r', 16, (uint8_t *)codeword);
    return 0x00;
}

/**
 * @brief Get encoded codeword chunk
 * Command 'g' + [chunk_index]
 * Returns 16 bytes starting at chunk_index * 16
 */
uint8_t get_codeword(uint8_t *data, uint8_t len) {
    uint8_t chunk = data[0];
    uint16_t offset = chunk * 16;

    // Total codeword size: 276 * 8 = 2208 bytes
    if (offset + 16 > sizeof(codeword)) {
        return 0x01; // Invalid chunk
    }

    simpleserial_put('r', 16, ((uint8_t *)codeword) + offset);
    return 0x00;
}

/**
 * @brief Set mask for masked encoding
 * Command 'r' + [1 byte mask]
 */
uint8_t set_mask(uint8_t *data, uint8_t len) {
    single_mask = data[0];
    return 0x00;
}

/**
 * @brief Perform masked Reed-Muller encode (2-share)
 * Command 'a' (masked encode)
 * Returns 32 bytes: encode(m^r) | encode(r)
 */
uint8_t encode_masked(uint8_t *data, uint8_t len) {
    uint8_t r = single_mask;
    uint8_t m_masked = single_msg ^ r;

    rm_codeword_t cw_masked, cw_mask;

    trigger_high();

    // Compute both shares
    encode(&cw_masked, (int32_t)m_masked); // encode(m^r)
    encode(&cw_mask, (int32_t)r);          // encode(r)

    trigger_low();

    // Store both shares
    single_codeword.u32[0] = cw_masked.u32[0];
    single_codeword.u32[1] = cw_masked.u32[1];
    single_codeword.u32[2] = cw_masked.u32[2];
    single_codeword.u32[3] = cw_masked.u32[3];

    single_codeword_mask[0] = cw_mask.u32[0] | ((uint64_t)cw_mask.u32[1] << 32);
    single_codeword_mask[1] = cw_mask.u32[2] | ((uint64_t)cw_mask.u32[3] << 32);

    // Send both shares: encode(m^r) then encode(r)
    uint8_t both[32];
    memcpy(both, single_codeword.u8, 16);
    memcpy(both + 16, single_codeword_mask, 16);
    simpleserial_put('r', 32, both);

    return 0x00;
}

/**
 * @brief Reset message to zeros
 * Command 'x'
 */
uint8_t reset(uint8_t *data, uint8_t len) {
    memset(msg, 0, VEC_N1_SIZE_BYTES);
    memset(codeword, 0, sizeof(codeword));
    single_msg = 0;
    single_mask = 0;
    memset(&single_codeword, 0, sizeof(single_codeword));
    memset(single_codeword_mask, 0, sizeof(single_codeword_mask));
    return 0x00;
}

int main(void) {
    platform_init();
    init_uart();
    trigger_setup();

    // Initialize message to zeros
    memset(msg, 0, VEC_N1_SIZE_BYTES);
    memset(codeword, 0, sizeof(codeword));

    simpleserial_init();

    // Register commands
    simpleserial_addcmd('m', 16, set_msg);              // Set message chunk (offset + data)
    simpleserial_addcmd('b', 1, set_single_byte);       // Set single byte for quick test
    simpleserial_addcmd('k', 1, set_mask);              // Set mask for masked encode
    simpleserial_addcmd('e', 1, encode_single);         // Encode single byte (1 dummy byte input)
    simpleserial_addcmd('a', 1, encode_masked);         // Masked encode (returns 32 bytes)
    simpleserial_addcmd('p', 16, encode_full);          // Encode full message
    simpleserial_addcmd('t', 1, encode_full_triggered); // Encode full with per-byte triggers
    simpleserial_addcmd('g', 1, get_codeword);          // Get codeword chunk
    simpleserial_addcmd('x', 1, reset);                 // Reset (1 dummy byte input)

    while (1)
        simpleserial_get();
}

/**
 * @file reed_muller.h
 * @brief Header file of reed_muller.c
 */

#ifndef HQC_REED_MULLER_H
#define HQC_REED_MULLER_H

#include <stddef.h>
#include <stdint.h>
#include "parameters.h"
#include "data_structures.h"

/**
 * @brief Encode a single byte into a 128-bit RM(1,7) codeword
 * @param[out] word The resulting codeword
 * @param[in] message The byte to encode
 */
void encode(rm_codeword_t *word, int32_t message);

/**
 * @brief Encode a full message using Reed-Muller RM(1,7)
 * @param[out] cdw Array receiving the encoded message
 * @param[in] msg Array storing the message
 */
void reed_muller_encode(uint64_t* cdw, const uint64_t* msg);

/**
 * @brief Decode a received word using Reed-Muller RM(1,7)
 * @param[out] msg Array receiving the decoded message
 * @param[in] cdw Array storing the received word
 */
void reed_muller_decode(uint64_t* msg, const uint64_t* cdw);

#endif  // HQC_REED_MULLER_H

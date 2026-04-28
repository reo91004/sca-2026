// SimpleSerial-SMAUG: ChipWhisperer SCA target wrapper around SMAUG-T KEM.
//
// Commands (SimpleSerial v1.1; matches HQC reference firmware so 0x44 ack
// behavior can be compared apples-to-apples between the two KEMs):
//   k  - generate fresh (pk, sk) on-chip; reply 'r' carries first 16 B of pk
//   e  - encapsulate against the resident pk (no trigger); reply 'r' = first 16 B of ct
//   d  - decapsulate the resident ct with sk; TRIGGER HIGH around the call
//        reply 'r' = 1-byte mismatch flag (0 = ss_dec == ss_enc)
//   p  - full pipeline: keypair -> enc -> (trigger high) dec (trigger low)
//        single-shot capture helper; reply 'r' = 1-byte mismatch flag
//
// All key/ct buffers stay resident on the MCU; the host never sees the full
// 672 B pk/ct (would exceed SimpleSerial's 64 B-per-frame limit).

#include <stdint.h>

#include "hal.h"
#include "simpleserial.h"

#include "api.h"          // CRYPTO_{PUBLICKEY,SECRETKEY,CIPHERTEXT}BYTES
#include "parameters.h"   // CRYPTO_BYTES (= 32, shared-secret size)

extern int crypto_kem_keypair(uint8_t *pk, uint8_t *sk);
extern int crypto_kem_enc(uint8_t *ct, uint8_t *ss, const uint8_t *pk);
extern int crypto_kem_dec(uint8_t *ss, const uint8_t *ct, const uint8_t *sk);

static uint8_t pk[CRYPTO_PUBLICKEYBYTES];
static uint8_t sk[CRYPTO_SECRETKEYBYTES];
static uint8_t ct[CRYPTO_CIPHERTEXTBYTES];
static uint8_t ss_enc[CRYPTO_BYTES];
static uint8_t ss_dec[CRYPTO_BYTES];

static uint8_t cmd_keypair(uint8_t *buf, uint8_t len)
{
    (void)len; (void)buf;
    crypto_kem_keypair(pk, sk);
    simpleserial_put('r', 16, pk);
    return 0x00;
}

static uint8_t cmd_encaps(uint8_t *buf, uint8_t len)
{
    (void)len; (void)buf;
    crypto_kem_enc(ct, ss_enc, pk);
    simpleserial_put('r', 16, ct);
    return 0x00;
}

static uint8_t cmd_decaps(uint8_t *buf, uint8_t len)
{
    (void)len; (void)buf;

    trigger_high();
    crypto_kem_dec(ss_dec, ct, sk);
    trigger_low();

    uint8_t mismatch = 0;
    for (unsigned i = 0; i < CRYPTO_BYTES; i++) {
        mismatch |= (uint8_t)(ss_dec[i] ^ ss_enc[i]);
    }
    uint8_t status = (mismatch != 0);
    simpleserial_put('r', 1, &status);
    return 0x00;
}

static uint8_t cmd_pipeline(uint8_t *buf, uint8_t len)
{
    (void)len; (void)buf;

    crypto_kem_keypair(pk, sk);
    crypto_kem_enc(ct, ss_enc, pk);

    trigger_high();
    crypto_kem_dec(ss_dec, ct, sk);
    trigger_low();

    uint8_t mismatch = 0;
    for (unsigned i = 0; i < CRYPTO_BYTES; i++) {
        mismatch |= (uint8_t)(ss_dec[i] ^ ss_enc[i]);
    }
    uint8_t status = (mismatch != 0);
    simpleserial_put('r', 1, &status);
    return 0x00;
}

int main(void)
{
    platform_init();
    init_uart();
    trigger_setup();

    simpleserial_init();
    simpleserial_addcmd('k', 0, cmd_keypair);
    simpleserial_addcmd('e', 0, cmd_encaps);
    simpleserial_addcmd('d', 0, cmd_decaps);
    simpleserial_addcmd('p', 0, cmd_pipeline);

    while (1) {
        simpleserial_get();
    }
}

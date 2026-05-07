// SimpleSerial-NTRU+: ChipWhisperer SCA target wrapper around NTRU+ KEM.
//
// firmware/simpleserial-smaug 와 같은 형식. 차이점 :
//   - NTRU+ archive 는 namespace prefix 가 없어 bare crypto_kem_*, sha3_256
//     을 직접 호출.
//   - NTRU+ 는 indcpa_enc/dec 를 export 하지 않아 SMAUG 의 'M'/'Z' 직역 wrapper
//     를 만들 수 없음. 명령 셋은 baseline + chosen-CT plumbing + 'X' (sk dump)
//     에 한정. NTRU+ 결정 구조에 맞는 diagnostic helpers (poly_basemul /
//     poly_invntt / poly_sotp_inv 기반) 는 v2 에서 별도 설계.
//   - ntruplus864 는 POLYBYTES=1296 이 32 의 배수가 아니라 ct/pk chunk 만
//     16B 로 (1296=81×16). 다른 level (576/768/1152) 는 32B chunk — smaug 와 동일.
//   - sk 사이즈 (2·POLYBYTES + 32) 는 모든 level 에서 32 의 배수라 sk chunk
//     는 항상 32B.
//
// Command groups and threat-model status :
//
//  (A) baseline / smoke                                  payload  trigger  ack
//      'k'  crypto_kem_keypair(pk, sk)                    0       OFF       16 B  pk[0:16] (지문)
//      'e'  crypto_kem_enc(ct, ss_enc, pk)                0       OFF       16 B  ct[0:16]
//      'd'  crypto_kem_dec(ss_dec, ct, sk)                0       ON        1  B  mismatch (0=OK)
//      'p'  k -> e -> (trigger high) d (trigger low)      0       d 구간     1  B  mismatch
//
//  (B) chosen-ciphertext/session plumbing
//      'F'  fresh keypair, sk 영속, ct_inj/ss_enc 0-init  0       OFF       16 B  sha3_256(pk)[0:16]
//      'B'  dump public pk bytes [idx*PKCB, idx*PKCB+PKCB) 1      OFF       PKCB   public pk chunk
//      'I'  ct chunk inject (idx 1B + data CTCB)          1+CTCB   OFF       1  B  status (0=OK,
//                                                                                  1=idx OOR,
//                                                                                  2=len OOR)
//      'L'  load done; ct_inj 무결성 지문                  0       OFF       16 B  sha3_256(ct_inj)[0:16]
//      'D'  crypto_kem_dec(ss_dec, ct_inj, sk_resident)   0       ON        1  B  mismatch (vs ss_enc)
//
//  (C) calibration helpers
//      'X'  raw sk bytes [idx*32, idx*32+32)                1      OFF      32  B  calibration labels only
//
//  CTCB / PKCB :
//      ntruplus864 :  16 B / 16 B  (POLYBYTES=1296 = 81×16)
//      others      :  32 B / 32 B  (smaug 와 동일)
//
//  Attack-valid claims must be trace-only. 'X' 응답은 ground-truth 라벨 정량화
//  목적이며 oracle 로 사용 금지.
//
//  ct 페이로드는 보드에 상주 (SimpleSerial v1.1 페이로드 한도 = 64 B 라
//  POLYBYTES 단위 ct 를 한 번에 못 올린다). 'I' 를 CHUNK_COUNT 회 호출하면
//  ct 가 채워지고, 'L' 로 host 가 무결성 검증, 'D' 로 분류 trace.
//
//  trigger 윈도우 : 'd', 'p', 'D' 모두 crypto_kem_dec 전체를 감싼다.
//  sub-trigger 는 v2 (NTRU+ 결정 구조에 맞춘 isolated multiplication / decoder
//  window) 에서 추가.

#include <stdint.h>
#include <string.h>

#include "hal.h"
#include "simpleserial.h"

#include "api.h"      // CRYPTO_{PUBLICKEY,SECRETKEY,CIPHERTEXT,BYTES}
#include "params.h"   // NTRUPLUS_{N,POLYBYTES,SYMBYTES,SSBYTES,...}

extern int crypto_kem_keypair(unsigned char *pk, unsigned char *sk);
extern int crypto_kem_enc(unsigned char *ct, unsigned char *ss,
                          const unsigned char *pk);
extern int crypto_kem_dec(unsigned char *ss, const unsigned char *ct,
                          const unsigned char *sk);

// NTRU+ 도 fips202 의 sha3_256 을 archive 에 그대로 노출 — host 용 무결성
// 지문 (pk/ct fingerprint) 산출에 활용. 심볼은 prefix 없이 bare.
extern void sha3_256(uint8_t *output, const uint8_t *input, size_t inputByteLen);

// ---- chunk size : level 별 분기 ------------------------------------------
//   - ntruplus864 (POLYBYTES=1296) 만 16B chunk
//   - 576/768/1152 는 32B chunk (smaug 와 동일 protocol)
#if NTRUPLUS_N == 864
  #define CTPK_CHUNK_BYTES 16u
#else
  #define CTPK_CHUNK_BYTES 32u
#endif
#define SK_CHUNK_BYTES   32u

#define CT_CHUNK_COUNT  (CRYPTO_CIPHERTEXTBYTES / CTPK_CHUNK_BYTES)
#define PK_CHUNK_COUNT  (CRYPTO_PUBLICKEYBYTES  / CTPK_CHUNK_BYTES)
#define SK_CHUNK_COUNT  (CRYPTO_SECRETKEYBYTES  / SK_CHUNK_BYTES)

#if CRYPTO_CIPHERTEXTBYTES % CTPK_CHUNK_BYTES
#error "CRYPTO_CIPHERTEXTBYTES must be divisible by CTPK_CHUNK_BYTES"
#endif
#if CRYPTO_PUBLICKEYBYTES % CTPK_CHUNK_BYTES
#error "CRYPTO_PUBLICKEYBYTES must be divisible by CTPK_CHUNK_BYTES"
#endif
#if CRYPTO_SECRETKEYBYTES % SK_CHUNK_BYTES
#error "CRYPTO_SECRETKEYBYTES must be divisible by SK_CHUNK_BYTES"
#endif

// SS_VER_1_1 페이로드 한도는 64 미만 (simpleserial_addcmd 가 len>=MAX_SS_LEN=64
// 거부). idx + chunk 가 가장 큰 'I' 페이로드 = 1+32=33B 또는 1+16=17B 모두
// 안전.
#define INJECT_PAYLOAD_LEN  (1u + CTPK_CHUNK_BYTES)
#define PKDUMP_PAYLOAD_LEN  1u
#define DUMPSK_PAYLOAD_LEN  1u

// ---- 정적 버퍼 ------------------------------------------------------------
// pk/sk/ct 는 보드에 상주. host 는 SimpleSerial 64 B/프레임 한도 때문에
// 전체를 받아오지 않고, 명령마다 16/32 B 지문/응답만 받는다.
static uint8_t pk[CRYPTO_PUBLICKEYBYTES];
static uint8_t sk[CRYPTO_SECRETKEYBYTES];
static uint8_t ct[CRYPTO_CIPHERTEXTBYTES];          // 'e'/'p' 가 채우는 정상 ct
static uint8_t ct_inj[CRYPTO_CIPHERTEXTBYTES];      // 'I' 가 채우는 chosen ct
static uint8_t ss_enc[CRYPTO_BYTES];                // 'e'/'F' 가 만든 reference ss
static uint8_t ss_dec[CRYPTO_BYTES];                // 'd'/'D' 가 채우는 ss
static uint8_t fp_buf[32];                          // sha3_256 응답용 임시 버퍼

// -- (A) baseline 명령 -------------------------------------------------------

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

// -- (B) chosen-ciphertext 명령 ----------------------------------------------

// 'F' : 새 sk 영속화. ct_inj/ss_enc 도 0-init 해서 이전 세션 잔여를 제거.
//        응답으로 sha3_256(pk)[0:16] 을 보낸다 — host 가 keypair 후
//        같은 보드와 통신 중인지를 짧게 검증할 수 있도록.
static uint8_t cmd_keygen_persistent(uint8_t *buf, uint8_t len)
{
    (void)len; (void)buf;
    crypto_kem_keypair(pk, sk);
    memset(ct_inj, 0, sizeof(ct_inj));
    memset(ss_enc, 0, sizeof(ss_enc));
    sha3_256(fp_buf, pk, sizeof(pk));
    simpleserial_put('r', 16, fp_buf);
    return 0x00;
}

// 'B' : public key chunk dump. pk is public and needed by host-side labels for
//       FO downstream diagnostics, e.g. H(pk) in G(mu', H(pk)).
static uint8_t cmd_dump_pk_chunk(uint8_t *buf, uint8_t len)
{
    if (len != PKDUMP_PAYLOAD_LEN) {
        uint8_t status = 2;
        simpleserial_put('r', 1, &status);
        return 0x00;
    }
    uint8_t idx = buf[0];
    if (idx >= PK_CHUNK_COUNT) {
        uint8_t status = 1;
        simpleserial_put('r', 1, &status);
        return 0x00;
    }
    simpleserial_put('r', CTPK_CHUNK_BYTES,
                     &pk[(unsigned)idx * CTPK_CHUNK_BYTES]);
    return 0x00;
}

// 'I' : ct_inj 의 [idx*CTCB, idx*CTCB+CTCB) 에 chunk 복사.
//        idx 는 페이로드 첫 바이트. 0..CT_CHUNK_COUNT-1 만 허용.
//        응답: 1B 상태 (0=OK, 1=idx OOR, 2=len mismatch).
static uint8_t cmd_inject(uint8_t *buf, uint8_t len)
{
    uint8_t status = 0;
    if (len != INJECT_PAYLOAD_LEN) {
        status = 2;
    } else {
        uint8_t idx = buf[0];
        if (idx >= CT_CHUNK_COUNT) {
            status = 1;
        } else {
            memcpy(&ct_inj[(unsigned)idx * CTPK_CHUNK_BYTES],
                   &buf[1], CTPK_CHUNK_BYTES);
        }
    }
    simpleserial_put('r', 1, &status);
    return 0x00;
}

// 'L' : ct_inj 가 CT_CHUNK_COUNT chunk 다 채워졌다고 host 가 선언 — 무결성
//        지문 응답. sha3_256(ct_inj)[0:16] 을 보내 host 가 자기 기대값과 비교.
static uint8_t cmd_load_done(uint8_t *buf, uint8_t len)
{
    (void)len; (void)buf;
    sha3_256(fp_buf, ct_inj, sizeof(ct_inj));
    simpleserial_put('r', 16, fp_buf);
    return 0x00;
}

// 'D' : crypto_kem_dec(ss_dec, ct_inj, sk). trigger high → dec → low.
//        chosen-CT 분석에서는 ss_enc 가 의미 없을 수 있지만 ('F' 가
//        ss_enc 를 0-init 했음), 응답 형식을 'd'/'p' 와 통일하기 위해
//        mismatch flag 1B 를 그대로 응답한다.
static uint8_t cmd_decap_inject(uint8_t *buf, uint8_t len)
{
    (void)len; (void)buf;

    trigger_high();
    crypto_kem_dec(ss_dec, ct_inj, sk);
    trigger_low();

    uint8_t mismatch = 0;
    for (unsigned i = 0; i < CRYPTO_BYTES; i++) {
        mismatch |= (uint8_t)(ss_dec[i] ^ ss_enc[i]);
    }
    uint8_t status = (mismatch != 0);
    simpleserial_put('r', 1, &status);
    return 0x00;
}

// -- (C) calibration helper --------------------------------------------------

// 'X' : sk 전체 (2·POLYBYTES + 32) 를 32B chunk 로 dump.
//        payload = 1B chunk index (0..SK_CHUNK_COUNT-1).
//        응답 = 32B raw sk bytes. NTRU+ sk layout (PQClean form) :
//          [0 .. POLYBYTES)             : f bytes (NTT 도메인 packed)
//          [POLYBYTES .. 2·POLYBYTES)   : f^{-1} bytes (NTT 도메인 packed)
//          [2·POLYBYTES .. +32)          : H(pk) 또는 implicit-rejection seed
//        host 측 디코드는 host/ntruplus/codec.py (별도 작성) 에서 poly_frombytes
//        역변환으로 ground-truth 라벨 추출. 본 응답은 calibration 용 — oracle
//        목적 사용 금지 (IND-CCA 위반).
//
//        SK_CHUNK_COUNT = CRYPTO_SECRETKEYBYTES / 32
//          ntruplus576  (sk=1760B) :  55 chunks
//          ntruplus768  (sk=2336B) :  73 chunks
//          ntruplus864  (sk=2624B) :  82 chunks
//          ntruplus1152 (sk=3488B) : 109 chunks
//        모든 level 에서 sk %32 == 0.
static uint8_t cmd_dump_sk_chunk(uint8_t *buf, uint8_t len)
{
    if (len != DUMPSK_PAYLOAD_LEN) {
        uint8_t status = 2;
        simpleserial_put('r', 1, &status);
        return 0x00;
    }
    uint8_t idx = buf[0];
    if (idx >= SK_CHUNK_COUNT) {
        uint8_t status = 1;
        simpleserial_put('r', 1, &status);
        return 0x00;
    }
    simpleserial_put('r', SK_CHUNK_BYTES,
                     &sk[(unsigned)idx * SK_CHUNK_BYTES]);
    return 0x00;
}

int main(void)
{
    platform_init();
    init_uart();
    trigger_setup();

    simpleserial_init();
    // (A) baseline
    simpleserial_addcmd('k', 0, cmd_keypair);
    simpleserial_addcmd('e', 0, cmd_encaps);
    simpleserial_addcmd('d', 0, cmd_decaps);
    simpleserial_addcmd('p', 0, cmd_pipeline);
    // (B) chosen-CT/session plumbing
    simpleserial_addcmd('F', 0,                  cmd_keygen_persistent);
    simpleserial_addcmd('B', PKDUMP_PAYLOAD_LEN, cmd_dump_pk_chunk);
    simpleserial_addcmd('I', INJECT_PAYLOAD_LEN, cmd_inject);
    simpleserial_addcmd('L', 0,                  cmd_load_done);
    simpleserial_addcmd('D', 0,                  cmd_decap_inject);
    // (C) calibration : raw sk dump (ground-truth 라벨용; oracle 금지)
    simpleserial_addcmd('X', DUMPSK_PAYLOAD_LEN, cmd_dump_sk_chunk);

    while (1) {
        simpleserial_get();
    }
}

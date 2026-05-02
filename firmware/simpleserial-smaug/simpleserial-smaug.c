// SimpleSerial-SMAUG: ChipWhisperer SCA target wrapper around SMAUG-T KEM.
//
// Two command groups:
//
//  (A) baseline / smoke                                  payload  trigger  ack
//      'k'  crypto_kem_keypair(pk, sk)                    0       OFF       16 B  pk[0:16] (지문)
//      'e'  crypto_kem_enc(ct, ss_enc, pk)                0       OFF       16 B  ct[0:16]
//      'd'  crypto_kem_dec(ss_dec, ct, sk)                0       ON        1  B  mismatch (0=OK)
//      'p'  k -> e -> (trigger high) d (trigger low)      0       d 구간     1  B  mismatch
//
//  (B) chosen-ciphertext (W1: docs/HANDOFF.md)
//      'F'  fresh keypair, sk 영속, ct_inj/ss_enc 0-init  0       OFF       16 B  sha3_256(pk)[0:16]
//      'I'  ct chunk inject (idx 1B + data 32B = 33B)     33      OFF       1  B  status (0=OK,
//                                                                                  1=idx OOR,
//                                                                                  2=len OOR)
//      'L'  load done; ct_inj 무결성 지문                  0       OFF       16 B  sha3_256(ct_inj)[0:16]
//      'M'  indcpa_enc(ct_inj, pk, μ, seed=0×32)          32      OFF       16 B  sha3_256(ct_inj)[0:16]
//           — payload = μ 32B (seed 는 펌웨어에서 zero-fixed)                      (host μ → valid PKE ct)
//           — SS_VER_1_1 의 addcmd len < MAX_SS_LEN=64 라 64B 페이로드 거부 →
//             μ 만 host 가 통제, seed 는 보드 zero. 추후 seed 변경 필요하면
//             별도 'N' 명령 (seed 32B inject) 추가.
//      'D'  crypto_kem_dec(ss_dec, ct_inj, sk_resident)   0       ON        1  B  mismatch (vs ss_enc)
//      'Z'  indcpa_dec(mu', sk, ct_inj) — verify/cmov skip 0       ON       32  B  µ'[0..31] (256 raw bits)
//      'X'  raw sk bytes [idx*32, idx*32+32)                1      OFF      32  B  sk chunk (host unpacks)
//
//  ct 페이로드는 보드에 상주 (SimpleSerial v1.1 페이로드 한도 = 64 B 라
//  672 B 의 ct 를 한 번에 못 올린다). 'I' 를 21 회 호출 (idx 0..20, 32 B/chunk)
//  하면 ct 가 채워지고, 'L' 로 host 가 무결성 검증, 'D' 로 분류 trace.
//
//  trigger 윈도우: 'd', 'p', 'D' 모두 crypto_kem_dec 전체를 감싼다.
//  sub-trigger (예: indcpa_dec 진입 ↔ µ' round 종료) 는 추후 W2 에서
//  보드 핀맵 정한 뒤 별도 GPIO 에 추가. 본 펌웨어는 hook (cmd_sub_trigger_)
//  자리만 마련했고, 기본은 dec 전체 trigger.

#include <stdint.h>
#include <string.h>

#include "hal.h"
#include "simpleserial.h"

#include "api.h"          // CRYPTO_{PUBLICKEY,SECRETKEY,CIPHERTEXT}BYTES
#include "parameters.h"   // CRYPTO_BYTES (= 32, shared-secret size)

extern int crypto_kem_keypair(uint8_t *pk, uint8_t *sk);
extern int crypto_kem_enc(uint8_t *ct, uint8_t *ss, const uint8_t *pk);
extern int crypto_kem_dec(uint8_t *ss, const uint8_t *ct, const uint8_t *sk);

// SMAUG-T 의 IND-CPA 레이어. 우리가 'M' 명령에서 직접 부른다 — KEM 레이어
// 와 달리 μ 와 seed 둘 다 host 가 통제할 수 있게 노출된 시그니처.
// indcpa.h 는 fips202.h (archive 내부) 를 끌어와 우리 include path 에서
// 못 컴파일하므로, 심볼만 forward-declare 한다. 매크로 SMAUG_NAMESPACE 가
// parameters.h 에 정의되어 있어 level 별 prefix 가 자동으로 풀린다 — smaug1
// 빌드면 cryptolab_smaug1_indcpa_enc 로 expand.
#define indcpa_enc_namespaced SMAUG_NAMESPACE(indcpa_enc)
extern void indcpa_enc_namespaced(uint8_t ctxt[CRYPTO_CIPHERTEXTBYTES],
                                  const uint8_t pk[CRYPTO_PUBLICKEYBYTES],
                                  const uint8_t mu[32],
                                  const uint8_t seed[32]);

// indcpa_dec — KEM 의 verify/cmov 단계를 건너뛰고 IND-CPA 복호화 (= µ' 추출)
// 만 부른다. crypto_kem_dec ≈ Decompress_pk + indcpa_dec + re-enc + cmov 인데
// chosen-CT µ' SCA 는 indcpa_dec 만 보면 충분 + 시간 단축 + 다른 단계가 만드는
// 측면 누설 (re-enc 의 µ 사용, cmov 의 K_z mask) 제거.
// signature: (delta[32], sk[PKE_SECRETKEY_BYTES=128], ct[CIPHERTEXT_BYTES])
#define indcpa_dec_namespaced SMAUG_NAMESPACE(indcpa_dec)
extern void indcpa_dec_namespaced(uint8_t delta[32],
                                  const uint8_t sk[128],
                                  const uint8_t ctxt[CRYPTO_CIPHERTEXTBYTES]);

// SMAUG-T 가 내부에서 쓰는 SHA3-256. KEM 자체가 그걸 export 하지 않아
// host 용 무결성 지문 산출은 같은 archive 내 fips202 의 sha3_256 을 직접
// 호출한다. 심볼 이름은 lib/crypto_kem/smaug{1,3,5}.a 의 fips202.c.o 에서
// `T sha3_256` 으로 잡힘.
extern void sha3_256(uint8_t *output, const uint8_t *input, size_t inputByteLen);

// ---- 정적 버퍼 -------------------------------------------------------------
// pk/sk/ct 는 보드에 상주. host 는 SimpleSerial 64 B/프레임 한도 때문에
// 전체를 받아오지 않고, 명령마다 16 B 지문/응답만 받는다.
static uint8_t pk[CRYPTO_PUBLICKEYBYTES];
static uint8_t sk[CRYPTO_SECRETKEYBYTES];
static uint8_t ct[CRYPTO_CIPHERTEXTBYTES];          // 'e'/'p' 가 채우는 정상 ct
static uint8_t ct_inj[CRYPTO_CIPHERTEXTBYTES];      // 'I' 가 채우는 chosen ct
static uint8_t ss_enc[CRYPTO_BYTES];                // 'e'/'F' 가 만든 reference ss
static uint8_t ss_dec[CRYPTO_BYTES];                // 'd'/'D' 가 채우는 ss
static uint8_t fp_buf[32];                          // sha3_256 응답용 임시 버퍼

#define CHUNK_BYTES   32u
#define CHUNK_COUNT   (CRYPTO_CIPHERTEXTBYTES / CHUNK_BYTES) /* smaug1: 21 */
#if CRYPTO_CIPHERTEXTBYTES % CHUNK_BYTES
#error "CRYPTO_CIPHERTEXTBYTES must be divisible by CHUNK_BYTES (32)"
#endif

// SS_MAX_LEN (chipwhisperer simpleserial-v1.1) = 64. 33B (idx + 32B data)
// 페이로드는 그 안. SS_VER_1_1 콜백 시그니처는 (uint8_t*, uint8_t) 이며
// len 은 simpleserial_addcmd 에 등록된 고정값.
#define INJECT_PAYLOAD_LEN  (1u + CHUNK_BYTES)  /* idx + data */
// 'M' 명령 페이로드 = μ (32B) 만. seed 는 펌웨어에서 zero-fixed.
// SS_VER_1_1 의 simpleserial_addcmd 는 `if (len >= MAX_SS_LEN=64) reject`
// 라 정확히 64B 페이로드도 거부 (실측). 따라서 μ + seed = 64 안 됨.
#define MU_BYTES            (DELTA_BYTES)        /* 32 (smaug1) */
#define MENC_PAYLOAD_LEN    (MU_BYTES)           /* 32 */
#if MENC_PAYLOAD_LEN >= 64
#error "MENC payload >= SS_VER_1_1 MAX_SS_LEN=64"
#endif

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

// 'I' : ct_inj 의 [idx*32, idx*32+32) 에 33B 페이로드의 1..32 를 복사.
//        idx 는 페이로드 첫 바이트. 0..CHUNK_COUNT-1 만 허용.
//        응답: 1B 상태 (0=OK, 1=idx OOR, 2=len mismatch).
static uint8_t cmd_inject(uint8_t *buf, uint8_t len)
{
    uint8_t status = 0;
    if (len != INJECT_PAYLOAD_LEN) {
        status = 2;
    } else {
        uint8_t idx = buf[0];
        if (idx >= CHUNK_COUNT) {
            status = 1;
        } else {
            memcpy(&ct_inj[(unsigned)idx * CHUNK_BYTES], &buf[1], CHUNK_BYTES);
        }
    }
    simpleserial_put('r', 1, &status);
    return 0x00;
}

// 'L' : ct_inj 가 21 chunk 다 채워졌다고 host 가 선언 — 무결성 지문 응답.
//        sha3_256(ct_inj)[0:16] 을 보내 host 가 자기 기대값과 비교.
static uint8_t cmd_load_done(uint8_t *buf, uint8_t len)
{
    (void)len; (void)buf;
    sha3_256(fp_buf, ct_inj, sizeof(ct_inj));
    simpleserial_put('r', 16, fp_buf);
    return 0x00;
}

// 'M' : indcpa_enc(ct_inj, pk, μ, seed=0×32). host 가 μ 만 통제 (seed 는
//        보드에서 zero-fixed — SS_VER_1_1 페이로드 한도 64 미만 제약 때문).
//        spec v4.0 Theorem 2 (1-δ correctness) 에 의해 다음 'D' 의
//        decryption 결과 µ′ = μ (LWR error 무시 가능). 같은 μ → 같은 ct
//        보장 (seed 가 fixed 라 indcpa_enc 가 deterministic).
//        응답 = sha3_256(ct_inj)[0:16] — host 가 결정성 검증.
static const uint8_t k_seed_zero[DELTA_BYTES] = {0};

static uint8_t cmd_pke_enc_labeled(uint8_t *buf, uint8_t len)
{
    if (len != MENC_PAYLOAD_LEN) {
        uint8_t status = 2;
        simpleserial_put('r', 1, &status);
        return 0x00;
    }
    indcpa_enc_namespaced(ct_inj, pk, /* mu */ buf, k_seed_zero);
    sha3_256(fp_buf, ct_inj, sizeof(ct_inj));
    simpleserial_put('r', 16, fp_buf);
    return 0x00;
}

// 'D' : crypto_kem_dec(ss_dec, ct_inj, sk). trigger high → dec → low.
//        chosen-CT 분석에서는 ss_enc 가 의미 없을 수 있지만 ('F' 가
//        ss_enc 를 0-init 했음), 응답 형식을 'd'/'p' 와 통일하기 위해
//        mismatch flag 1B 를 그대로 응답한다 (분류기 입력으로 ss_dec[0]
//        도 노출하고 싶으면 여기 확장).
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

// 'Z' : indcpa_dec 만 호출 (verify/re-enc/cmov 스킵 → 시간/누설 단축).
//        trigger 윈도우 = indcpa_dec 만. 응답 = µ'[0..15] 16B.
//        host 는 µ'[bit_idx/8] >> (bit_idx%8) & 1 로 µ'_i 를 직접 확인 →
//        chosen-CT 분류 ground truth 로 활용 (mismatch flag 보다 훨씬 풍부).
//        SS_VER_1_1 응답 한도 64B 안. PKE sk 는 KEM sk 의 첫 PKE_SECRETKEY_BYTES
//        (smaug1: 128B) 영역.
static uint8_t cmd_indcpa_dec_inject(uint8_t *buf, uint8_t len)
{
    (void)len; (void)buf;
    uint8_t mu_prime[DELTA_BYTES];

    trigger_high();
    indcpa_dec_namespaced(mu_prime, sk, ct_inj);
    trigger_low();

    // 32B 응답 = 256 µ'_i 비트 전체. SS_VER_1_1 응답 한도 64B 안 (실측 OK).
    // µ'[0..15] = 첫 128 bits, µ'[16..31] = 후반 128 bits.
    simpleserial_put('r', 32, mu_prime);
    return 0x00;
}

// 'X' : sk 의 첫 PKE_SECRETKEY_BYTES (= 128B for smaug1) 를 32B chunk 로 dump.
//        payload = 1B chunk index (0..3 for smaug1).
//        응답 = 32B raw sk bytes (host 가 SMAUG-T Sx unpacking 으로 ternary 디코드).
//        Sx_to_bytes 는 4 ternary coeff/byte (2 bit each). 호스트 측 디코드는
//        host/smaug/codec.py 에 추가.
//        ground truth — E3c 분류기 정확도 정량화에 필수.
#define DUMPSK_PAYLOAD_LEN 1u
#define DUMPSK_CHUNK_BYTES 32u
#define DUMPSK_TOTAL_BYTES 128u   /* PKE_SECRETKEY_BYTES = SKPOLYVEC_BYTES = SKPOLY(64)*2 */
#define DUMPSK_CHUNK_COUNT (DUMPSK_TOTAL_BYTES / DUMPSK_CHUNK_BYTES)  /* = 4 */

static uint8_t cmd_dump_sk_chunk(uint8_t *buf, uint8_t len)
{
    if (len != DUMPSK_PAYLOAD_LEN) {
        uint8_t status = 2;
        simpleserial_put('r', 1, &status);
        return 0x00;
    }
    uint8_t idx = buf[0];
    if (idx >= DUMPSK_CHUNK_COUNT) {
        uint8_t status = 1;
        simpleserial_put('r', 1, &status);
        return 0x00;
    }
    simpleserial_put('r', DUMPSK_CHUNK_BYTES,
                     &sk[(unsigned)idx * DUMPSK_CHUNK_BYTES]);
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
    // (B) chosen-CT
    simpleserial_addcmd('F', 0,                   cmd_keygen_persistent);
    simpleserial_addcmd('I', INJECT_PAYLOAD_LEN,  cmd_inject);
    simpleserial_addcmd('L', 0,                   cmd_load_done);
    simpleserial_addcmd('M', MENC_PAYLOAD_LEN,    cmd_pke_enc_labeled);
    simpleserial_addcmd('D', 0,                   cmd_decap_inject);
    simpleserial_addcmd('Z', 0,                   cmd_indcpa_dec_inject);
    simpleserial_addcmd('X', DUMPSK_PAYLOAD_LEN,  cmd_dump_sk_chunk);

    while (1) {
        simpleserial_get();
    }
}

// Bridge: NTRU+ archive (PQClean-style randombytes.c) calls
// rng_get_random_blocking() once per 32-bit word it needs. In the upstream
// PQM4 setup that name lives in libopencm3 (drivers/rng.c). We're not linking
// libopencm3, so route the call to ChipWhisperer's STM32F4 HAL helper instead -
// it wraps HAL_RNG_GenerateRandomNumber.
//
// Same shape as firmware/simpleserial-smaug/smaug_rng.c — both archives leave
// rng_get_random_blocking undefined and consume the same HAL.

#include <stdint.h>

extern uint32_t get_rand(void);

uint32_t rng_get_random_blocking(uint32_t rng)
{
    (void)rng;
    return get_rand();
}

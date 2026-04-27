#ifndef SMAUG_CBD_H
#define SMAUG_CBD_H

#include "poly.h"

#include <stdint.h>
#include <stdio.h>

#define CBDSEED_BYTES ((4 * LWE_N) / 8)

#define poly_cbd SMAUG_NAMESPACE(poly_cbd)
void poly_cbd(poly *r, const uint8_t buf[CBDSEED_BYTES]);

#endif // SMAUG_CBD_H

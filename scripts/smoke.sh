#!/usr/bin/env bash
# sca-2026 smoke test
#   preflight -> build -> flash -> capture -> validate -> visualize
#
# 위치: scripts/smoke.sh (REPO_ROOT 한 단계 위가 레포 루트)
#
# Usage:
#   scripts/smoke.sh                                    # SMAUG-T level 1 (기본)
#   TARGET=hqc-custom scripts/smoke.sh                  # HQC custom RM (encode_single)
#   TARGET=hqc-pqclean scripts/smoke.sh                 # HQC PQClean code_encode (RS+RM)
#   LEVEL=3 scripts/smoke.sh                            # SMAUG level 3
#   NUM_TRACES=200 SAMPLES=8000 scripts/smoke.sh
#   SKIP_BUILD=1 SKIP_FLASH=1 scripts/smoke.sh          # capture + viz only
#   SKIP_CAPTURE=1 NPZ=traces/foo.npz scripts/smoke.sh  # re-visualize a saved file
#
# Env overrides:
#   TARGET         smaug | hqc-custom | hqc-pqclean (기본 smaug)
#   LEVEL          SMAUG-T 보안 레벨 (1|3|5, 기본 1; TARGET=smaug 일 때만 사용)
#   NUM_TRACES     캡처 트레이스 수 (기본 50)
#   SAMPLES        트레이스당 ADC 샘플 (기본 24400)
#   GAIN_DB        LNA 게인 dB (기본 25.0)
#   SERIAL         CW1173 시리얼 명시 (기본 자동 선택)
#   CW_FW_PATH     ChipWhisperer 펌웨어 트리 (기본 makefile 값)
#   NPZ            캡처 출력 .npz (기본 traces/smoke_<target>.npz)
#   PNG            시각화 출력 .png (기본 NPZ 와 같은 stem + .png)

set -euo pipefail

TARGET="${TARGET:-smaug}"
LEVEL="${LEVEL:-1}"
NUM_TRACES="${NUM_TRACES:-50}"
SAMPLES="${SAMPLES:-24400}"
GAIN_DB="${GAIN_DB:-25.0}"
SERIAL="${SERIAL:-}"
SKIP_BUILD="${SKIP_BUILD:-0}"
SKIP_FLASH="${SKIP_FLASH:-0}"
SKIP_CAPTURE="${SKIP_CAPTURE:-0}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST_DIR="$REPO_ROOT/host"
SCRIPTS_DIR="$REPO_ROOT/scripts"

# ---- TARGET dispatch ---------------------------------------------------------
# Each TARGET sets:
#   FW_DIR, HEX_PATH, ELF_PATH, MAKE_ARGS  : build/flash inputs
#   CAP_TARGET, CAP_CMD, CAP_SEND, CAP_RESP : capture.py args
#   DO_MATCH_CHECK                          : 1 → SMAUG mismatch flag check
#   NPZ_DEFAULT                             : default output path
#   LIB_ARCHIVE                             : optional preflight check
case "$TARGET" in
  smaug)
    FW_DIR="$REPO_ROOT/firmware/simpleserial-smaug"
    HEX_PATH="$FW_DIR/simpleserial-smaug-CW308_STM32F4.hex"
    ELF_PATH="$FW_DIR/simpleserial-smaug-CW308_STM32F4.elf"
    MAKE_ARGS=(PLATFORM=CW308_STM32F4 "SMAUG_LEVEL=$LEVEL")
    LIB_ARCHIVE="$REPO_ROOT/lib/crypto_kem/smaug${LEVEL}.a"
    CAP_TARGET="smaug"
    CAP_CMD="p"        # full pipeline (keypair + enc + dec)
    CAP_SEND=0
    CAP_RESP=1         # 1-byte mismatch flag
    DO_MATCH_CHECK=1
    NPZ_DEFAULT="$REPO_ROOT/traces/smoke_smaug${LEVEL}.npz"
    ;;
  hqc-custom)
    FW_DIR="$REPO_ROOT/firmware/simpleserial-hqc"
    HEX_PATH="$FW_DIR/simpleserial-hqc-CW308_STM32F4.hex"
    ELF_PATH="$FW_DIR/simpleserial-hqc-CW308_STM32F4.elf"
    MAKE_ARGS=(PLATFORM=CW308_STM32F4 HQC_IMPL=custom)
    LIB_ARCHIVE=""     # HQC has no .a archive; sources compiled in firmware build
    CAP_TARGET="hqc"
    CAP_CMD="e"        # encode_single (single-byte RM encode)
    CAP_SEND=1         # 1-byte dummy
    CAP_RESP=16        # 16-byte codeword
    DO_MATCH_CHECK=0
    NPZ_DEFAULT="$REPO_ROOT/traces/smoke_hqc_custom.npz"
    ;;
  hqc-pqclean)
    FW_DIR="$REPO_ROOT/firmware/simpleserial-hqc"
    HEX_PATH="$FW_DIR/simpleserial-hqc-CW308_STM32F4.hex"
    ELF_PATH="$FW_DIR/simpleserial-hqc-CW308_STM32F4.elf"
    MAKE_ARGS=(PLATFORM=CW308_STM32F4 HQC_IMPL=pqclean)
    LIB_ARCHIVE=""
    CAP_TARGET="hqc"
    CAP_CMD="e"        # encode_single — same shape as custom for clean comparison
    CAP_SEND=1
    CAP_RESP=16
    DO_MATCH_CHECK=0
    NPZ_DEFAULT="$REPO_ROOT/traces/smoke_hqc_pqclean.npz"
    ;;
  *)
    echo "[FAIL] Unknown TARGET=$TARGET (smaug | hqc-custom | hqc-pqclean)" >&2
    exit 1
    ;;
esac

NPZ="${NPZ:-$NPZ_DEFAULT}"
PNG="${PNG:-${NPZ%.npz}.png}"

c_red()  { printf '\033[31m%s\033[0m' "$*"; }
c_grn()  { printf '\033[32m%s\033[0m' "$*"; }
c_yel()  { printf '\033[33m%s\033[0m' "$*"; }
c_cyn()  { printf '\033[36m%s\033[0m' "$*"; }
hdr()    { printf '\n%s %s\n' "$(c_cyn "==>")" "$*"; }
info()   { printf '%s %s\n'   "$(c_cyn "[INFO]")" "$*"; }
ok()     { printf '%s %s\n'   "$(c_grn "[OK]")"   "$*"; }
warn()   { printf '%s %s\n'   "$(c_yel "[WARN]")" "$*"; }
fail()   { printf '%s %s\n'   "$(c_red "[FAIL]")" "$*"; exit 1; }

# ---------------------------------------------------------------- A. preflight
hdr "[A] preflight  (TARGET=$TARGET  LEVEL=$LEVEL  N=$NUM_TRACES  S=$SAMPLES)"

if [[ -n "$LIB_ARCHIVE" ]]; then
    [[ -f "$LIB_ARCHIVE" ]] || fail "정적 아카이브 없음: $LIB_ARCHIVE  (LEVEL=1|3|5 인지 확인)"
fi
[[ -f "$FW_DIR/makefile" ]] || fail "firmware makefile 없음: $FW_DIR/makefile"
[[ -d "$HOST_DIR" ]] || fail "host 디렉토리 없음: $HOST_DIR"

need() { command -v "$1" >/dev/null 2>&1 || fail "필수 도구 없음: $1  ($2)"; }
if [[ "$SKIP_BUILD" != "1" ]]; then
    need arm-none-eabi-gcc "apt install gcc-arm-none-eabi"
    need arm-none-eabi-size "binutils-arm-none-eabi 포함"
    need make             "apt install make"
fi
need python3 "Python 3.10+ 필요"

# CW_FW_PATH: makefile 디폴트는 /home/pacl/.../chipwhisperer/firmware/mcu.
# 사용자가 env 로 덮어썼으면 그것을, 아니면 makefile 디폴트를 검사.
EFFECTIVE_CW_FW_PATH="${CW_FW_PATH:-/home/pacl/Documents/Repository/chipwhisperer/firmware/mcu}"
if [[ "$SKIP_BUILD" != "1" && ! -d "$EFFECTIVE_CW_FW_PATH" ]]; then
    fail "CW_FW_PATH 가 디렉토리가 아님: $EFFECTIVE_CW_FW_PATH  (CW_FW_PATH 로 덮어쓰라)"
fi

# Python 의존성 (chipwhisperer/numpy 는 캡처/플래시에서, matplotlib 는 시각화에서 필요)
py_check() {
    python3 - "$@" <<'PY' || return 1
import importlib.util, sys
missing = [m for m in sys.argv[1:] if importlib.util.find_spec(m) is None]
if missing:
    print(",".join(missing)); sys.exit(1)
PY
}
need_py=()
[[ "$SKIP_FLASH" != "1" || "$SKIP_CAPTURE" != "1" ]] && need_py+=(chipwhisperer numpy)
need_py+=(numpy matplotlib)
if missing="$(py_check "${need_py[@]}" 2>&1 || true)" && [[ -n "$missing" ]]; then
    # py_check 가 stderr 로 import 에러를 흘릴 수 있으므로, 마지막 줄만 메시지로 사용
    last="$(printf '%s\n' "$missing" | tail -n1)"
    fail "python 패키지 누락: $last  (pip install $(echo "$last" | tr ',' ' '))"
fi

# CW USB 존재 여부는 약한 신호 — 없으면 경고만.
if command -v lsusb >/dev/null 2>&1; then
    if ! lsusb 2>/dev/null | grep -qi '2b3e:ace2'; then
        warn "lsusb 에 NewAE 2b3e:ace2 미감지. 보드/케이블 확인 필요할 수 있음."
    fi
fi
ok "preflight 통과"

# ------------------------------------------------------------------- B. build
if [[ "$SKIP_BUILD" == "1" ]]; then
    info "SKIP_BUILD=1 → 빌드 건너뜀"
    [[ -f "$HEX_PATH" ]] || fail "빌드 산출물 없음: $HEX_PATH  (SKIP_BUILD 해제 권장)"
else
    hdr "[B] build firmware  (TARGET=$TARGET  args=${MAKE_ARGS[*]})"
    make_args=("${MAKE_ARGS[@]}")
    [[ -n "${CW_FW_PATH:-}" ]] && make_args+=("CW_FW_PATH=$CW_FW_PATH")
    ( cd "$FW_DIR" && make "${make_args[@]}" clean >/dev/null && make "${make_args[@]}" )
    [[ -f "$HEX_PATH" && -f "$ELF_PATH" ]] || fail "빌드 산출물 없음: $HEX_PATH"
    arm-none-eabi-size "$ELF_PATH"
    ok "build 완료: $(basename "$HEX_PATH")"
fi

# ------------------------------------------------------------------- C. flash
if [[ "$SKIP_FLASH" == "1" ]]; then
    info "SKIP_FLASH=1 → 플래시 건너뜀"
else
    hdr "[C] flash via CW1173"
    # .hex 직접 사용 — chipwhisperer 의 IntelHex.write_hex_file 가 Py3 에서
    # 깨져 있어 .bin → 임시 .hex 변환 경로는 피한다. 빌드가 만든 .hex 가
    # 0x08000000 베이스로 이미 정렬되어 있다.
    flash_args=("$HEX_PATH")
    [[ -n "$SERIAL" ]] && flash_args+=(--serial "$SERIAL")
    python3 "$HOST_DIR/upload.py" "${flash_args[@]}"
    ok "flash 완료"
fi

# ----------------------------------------------------------------- D. capture
if [[ "$SKIP_CAPTURE" == "1" ]]; then
    info "SKIP_CAPTURE=1 → 캡처 건너뜀 (검증/시각화는 기존 $NPZ 사용)"
    [[ -f "$NPZ" ]] || fail "기존 .npz 없음: $NPZ"
else
    hdr "[D] capture traces  (target=$CAP_TARGET cmd='$CAP_CMD' send=$CAP_SEND resp=$CAP_RESP -> $NPZ)"
    mkdir -p "$(dirname "$NPZ")"
    cap_args=(
        --target "$CAP_TARGET"
        -n "$NUM_TRACES" -s "$SAMPLES" -g "$GAIN_DB"
        -c "$CAP_CMD" --send-len "$CAP_SEND" --resp-len "$CAP_RESP"
        -o "$NPZ"
    )
    [[ -n "$SERIAL" ]] && cap_args+=(--serial "$SERIAL")
    # capture.py 는 timeouts > 0 이면 exit 2, 정상이면 0. 명시적으로 분기.
    set +e
    python3 "$HOST_DIR/capture.py" "${cap_args[@]}"
    rc=$?
    set -e
    case "$rc" in
        0) ok  "capture 완료 (timeouts=0)";;
        2) fail "capture 중 타임아웃 발생 — 보드/케이블/펌웨어 상태 점검";;
        *) fail "capture.py 비정상 종료 (rc=$rc)";;
    esac
fi

# --------------------------------------------------------- E. validate + F. viz
# host/analysis/ 패키지가 SSOT — scripts/plot_overview.py 만 호출한다.
# 캡처를 건너뛴 경우(SKIP_CAPTURE=1) N/S 검증은 생략 (기존 .npz 그대로 시각화).
hdr "[E/F] validate + visualize  ($NPZ -> $PNG)"
plot_args=("$NPZ" --png "$PNG")
if [[ "$SKIP_CAPTURE" != "1" ]]; then
    plot_args+=(--expected-n "$NUM_TRACES" --expected-samples "$SAMPLES")
fi
[[ "$DO_MATCH_CHECK" == "1" ]] && plot_args+=(--kem-match-check)

python3 "$SCRIPTS_DIR/plot_overview.py" "${plot_args[@]}"
ok "smoke test 통과 — $NPZ / $PNG"

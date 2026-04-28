#!/usr/bin/env bash
# sca-2026 smoke test
#   preflight -> build -> flash -> capture -> validate -> visualize
#
# Usage:
#   ./smoke.sh                                    # SMAUG-T level 1 (기본)
#   TARGET=hqc-custom ./smoke.sh                  # HQC custom RM (encode_single)
#   TARGET=hqc-pqclean ./smoke.sh                 # HQC PQClean code_encode (RS+RM)
#   LEVEL=3 ./smoke.sh                            # SMAUG level 3
#   NUM_TRACES=200 SAMPLES=8000 ./smoke.sh
#   SKIP_BUILD=1 SKIP_FLASH=1 ./smoke.sh          # capture + viz only
#   SKIP_CAPTURE=1 NPZ=traces/foo.npz ./smoke.sh  # re-visualize a saved file
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

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST_DIR="$REPO_ROOT/host"

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
hdr "[E/F] validate + visualize  ($NPZ -> $PNG)"
NPZ="$NPZ" PNG="$PNG" EXPECTED_N="$NUM_TRACES" EXPECTED_S="$SAMPLES" \
SKIPPED_CAPTURE="$SKIP_CAPTURE" DO_MATCH_CHECK="$DO_MATCH_CHECK" \
TARGET="$TARGET" python3 - <<'PY'
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

npz_path = os.environ["NPZ"]
png_path = os.environ["PNG"]
expected_n = int(os.environ["EXPECTED_N"])
expected_s = int(os.environ["EXPECTED_S"])
skipped_capture = os.environ["SKIPPED_CAPTURE"] == "1"
do_match_check = os.environ["DO_MATCH_CHECK"] == "1"
target_name = os.environ["TARGET"]

d = np.load(npz_path, allow_pickle=True)
traces = d["traces"]
responses = d["responses"]
meta = d["meta"].item()

print(f"[VAL] target     : {target_name}")
print(f"[VAL] file       : {npz_path}")
print(f"[VAL] traces     : shape={traces.shape} dtype={traces.dtype}")
print(f"[VAL] responses  : shape={responses.shape} dtype={responses.dtype}")
print(f"[VAL] meta       : {meta}")

errs = []
if traces.ndim != 2:
    errs.append(f"traces.ndim={traces.ndim} (expected 2)")
if not skipped_capture:
    if traces.shape[0] != expected_n:
        errs.append(f"traces.shape[0]={traces.shape[0]} != N={expected_n}")
    if traces.shape[1] != expected_s:
        errs.append(f"traces.shape[1]={traces.shape[1]} != samples={expected_s}")
if traces.dtype != np.float32:
    errs.append(f"traces.dtype={traces.dtype} (expected float32)")
if responses.dtype != np.uint8:
    errs.append(f"responses.dtype={responses.dtype} (expected uint8)")

n_timeouts = int(meta.get("n_timeouts", 0))
if n_timeouts != 0:
    errs.append(f"meta.n_timeouts={n_timeouts} (expected 0)")
elif do_match_check:
    # SMAUG only: 1바이트 mismatch flag → 0 이 정상. HQC 응답은 codeword 데이터라 무의미.
    bad = int((responses != 0).sum())
    if bad:
        errs.append(f"mismatch != 0 인 trace {bad}개 (KEM dec 결과가 enc 와 불일치)")
else:
    # HQC: 응답이 codeword 페이로드. 응답 분포만 간단 요약.
    nonzero_rows = int((responses.any(axis=1) if responses.ndim == 2 else responses != 0).sum())
    print(f"[VAL] HQC ack    : nonzero rows = {nonzero_rows}/{responses.shape[0]} "
          f"(round-trip 확인용; mismatch flag 검사는 SMAUG 전용)")

# 트레이스 자체의 sanity: NaN/Inf, 전부 0, 분산 거의 0 등.
finite = np.isfinite(traces).all()
if not finite:
    errs.append("traces 에 NaN/Inf 포함")
elif not skipped_capture and (np.abs(traces).max() < 1e-9):
    errs.append("traces 가 사실상 0 — 게인/트리거/HAL 클록 의심")

if errs:
    print("[VAL] 실패 원인:")
    for e in errs:
        print(f"       - {e}")
    sys.exit(1)
print("[VAL] OK — KEM 동작 + 트레이스 형식 모두 정상")

# 시각화: 평균 ± 1σ + 무작위 5개 오버레이.
mean = traces.mean(axis=0)
std  = traces.std(axis=0)
x = np.arange(traces.shape[1])
rng = np.random.default_rng(0)
sample_idx = rng.choice(traces.shape[0], size=min(5, traces.shape[0]), replace=False)

fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                         gridspec_kw={"height_ratios": [2, 1]})
ax1, ax2 = axes
ax1.fill_between(x, mean - std, mean + std, color="C0", alpha=0.25, label=r"$\mu \pm \sigma$")
ax1.plot(x, mean, color="C0", lw=0.8, label=r"$\mu$")
for i in sample_idx:
    ax1.plot(x, traces[i], lw=0.4, alpha=0.6, label=f"trace {i}")
ax1.set_ylabel("ADC (norm.)")
ax1.set_title(f"{target_name} smoke trace  N={traces.shape[0]}  S={traces.shape[1]}  "
              f"cmd={meta.get('cmd','?')}  gain={meta.get('gain_db','?')}dB")
ax1.legend(loc="upper right", fontsize=8, ncols=2)
ax1.grid(alpha=0.3)

ax2.plot(x, std, color="C3", lw=0.6)
ax2.set_ylabel(r"$\sigma$")
ax2.set_xlabel("sample")
ax2.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(png_path, dpi=120)
print(f"[VIZ] saved {png_path}")
PY

ok "smoke test 통과 — $NPZ / $PNG"

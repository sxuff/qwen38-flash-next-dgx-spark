#!/usr/bin/env bash
# run.sh - one-command setup and launch for this recipe on one NVIDIA GB10.
#
#   git clone https://github.com/sxuff/qwen38-flash-next-dgx-spark && cd qwen38-flash-next-dgx-spark && ./run.sh
#
# Every pin and flag below is copied from this repo's README, manifests or
# receipts; the comment next to each value names its source. Nothing here
# tunes, "improves" or changes a measured setting.
set -euo pipefail
cd "$(dirname "$0")"
REPO_DIR="$(pwd)"

# ===================== recipe config (per repo) =============================
RECIPE_NAME="qwen38-flash-next-dgx-spark"
PORT="${PORT:-8001}"                             # README.md:107 "bound to 127.0.0.1:8001"; scripts/serve.py:33
MODELS_DIR="${MODELS_DIR:-$HOME/models}"         # README.md:60-61 put both model roots under $HOME/models
SRC_DIR="${SRC_DIR:-$HOME/src}"                  # unused: the pinned engine builds inside Docker (docker/Dockerfile:11)
STATE_DIR="${STATE_DIR:-$HOME/.local/state/qwen38-flash-next-dgx-spark}"
NEEDS_TOOLS="git curl sha256sum docker python3"  # README.md:43 "working Docker GPU access, Python 3 and Git"
NEED_MEM_BYTES=96700886352                       # README.md:121; scripts/supervise.py:18-19,51: 96220600952 + 480285400 (prompt kernels on by default)
ENGINE_DISK_BYTES=21474836480                    # 20 GiB floor: scripts/build_tensorfold.sh:5; same reserve as manifests/exl3-405.json:7 (the downloader enforces its own reserve on top)
ENGINE_KIND=process                              # the pid file tracks scripts/supervise.py, which owns the container
CONTAINER_NAME=""                                # container qwen38-flash-next-tensorfold (scripts/supervise.py:111) is stopped via stop_override below
READY_URL="http://127.0.0.1:${PORT}/health"      # scripts/supervise.py:160; scripts/smoke.py checks the same endpoint
READY_TIMEOUT=1200                               # scripts/supervise.py:144 startup readiness deadline
EXPECTED_LINE="1,568 tok/s native prefill at 64K code, up from 668 tok/s before the prompt kernels (a prefill and TTFT claim, not a decode-speed claim; README.md:13)"
RECEIPT_PATH="results/prefill-repeat.json"

# The repo's own env names and default locations (README.md:60-71).
MODEL_ROOT="${MODEL_ROOT:-$MODELS_DIR/Qwen3.8-Flash-Next-EXL3-4.05}"             # README.md:60
VISION_ROOT="${VISION_ROOT:-$MODELS_DIR/Qwen3.8-Flash-Next-BF16-vision}"         # README.md:61
VISION_WEIGHTS="${VISION_WEIGHTS:-$VISION_ROOT/vision_tower_bf16.safetensors}"   # README.md:71

# Disk: the downloader needs the remaining weight bytes plus its 20 GiB reserve
# (scripts/download_model.py:313). Count them here too, so --check reports a shortfall up front.
_present() { if [ -e "$1" ]; then du -sb "$1" 2>/dev/null | cut -f1; else echo 0; fi; }
WEIGHT_BYTES=$(( 107463600896 + 897899504 ))     # manifests/exl3-405.json:6 + manifests/vision-bf16.json:5
_have=$(( $(_present "$MODEL_ROOT") + $(_present "$VISION_ROOT") ))
ENGINE_DISK_BYTES=$(( ENGINE_DISK_BYTES + (WEIGHT_BYTES > _have ? WEIGHT_BYTES - _have : 0) ))

# Engine pins (verified by scripts, repeated here only for the label check):
IMAGE="qwen38-flash-next-tensorfold:609ca419"    # scripts/build_tensorfold.sh:6; scripts/supervise.py:16
RUNTIME_REV="609ca419abecebdc5a059498a613680bd3aa847f"                             # docker/Dockerfile:15; scripts/supervise.py:56
PROMPT_PATCH_SHA="bf884015181a1241b4038b25fdafff0f7d4887ce5725f112d99ba2f460f309fa" # docker/Dockerfile:16; scripts/supervise.py:58
# Weights pins live in manifests/: turboderp/Qwen3.8-Flash-Next-exl3
# @ 55a732e0c4c3d4614bc42b68493bb930d9b02c0a, 107,463,600,896 bytes (manifests/exl3-405.json:3-6)
# and the BF16 vision tower @ b5a234522364b10cdc4869844597bf77eeb942a7, 897,899,504 bytes,
# sha256 cdd69998e52e34badced49ef8ec09824b8b9a52e488094f0cadeadc7ebfc641e (manifests/vision-bf16.json:3-9).
# Both are public; no Hub token is required (README.md:74).

# Empty: weights_step below calls the repo's own pinned downloader instead.
WEIGHTS="
"

weights_step() {  # scripts/download_model.py resumes, verifies SHA-256/git-blob-SHA-1 and keeps a 20 GiB reserve (README.md:74)
  local pair manifest destination
  for pair in "manifests/exl3-405.json|$MODEL_ROOT" "manifests/vision-bf16.json|$VISION_ROOT"; do
    manifest="${pair%%|*}"; destination="${pair##*|}"
    if [ "$CHECK" = 1 ]; then
      if python3 "$REPO_DIR/scripts/download_model.py" --manifest "$REPO_DIR/$manifest" \
                 --destination "$destination" --verify-only >/dev/null 2>&1; then
        ok "already verified: $manifest -> $destination"
      else
        plan "python3 scripts/download_model.py --manifest $manifest --destination $destination"
      fi
    else
      python3 "$REPO_DIR/scripts/download_model.py" --manifest "$REPO_DIR/$manifest" --destination "$destination"
    fi
  done
}

engine_step() {   # build the pinned image locally; skip if it already carries the pinned labels
  local rev patch
  rev="$(docker image inspect --format '{{index .Config.Labels "qwen.runtime.revision"}}' "$IMAGE" 2>/dev/null || true)"
  patch="$(docker image inspect --format '{{index .Config.Labels "qwen.prompt-expert.patch.sha256"}}' "$IMAGE" 2>/dev/null || true)"
  if [ "$rev" = "$RUNTIME_REV" ] && [ "$patch" = "$PROMPT_PATCH_SHA" ]; then  # same check as scripts/supervise.py:54-59
    ok "image present with pinned labels: $IMAGE"
  else
    doit bash "$REPO_DIR/scripts/build_tensorfold.sh"
  fi
}

serve_step() {    # README.md:94 foreground command, backgrounded; supervise.py runs the guards and owns the container
  if [ "$CHECK" = 1 ]; then
    plan "nohup bash scripts/run_server.sh \"$MODEL_ROOT\" \"$VISION_WEIGHTS\" --port $PORT >\"$LOGFILE\" 2>&1 &  (pid file: $PIDFILE)"
    return
  fi
  nohup bash "$REPO_DIR/scripts/run_server.sh" "$MODEL_ROOT" "$VISION_WEIGHTS" --port "$PORT" >"$LOGFILE" 2>&1 &
  echo $! >"$PIDFILE"
  ok "supervisor started (pid $(cat "$PIDFILE"))"
}

stop_override() { # stop the supervisor; on SIGTERM it halts and stops its own container (scripts/supervise.py:135-188)
  local name=qwen38-flash-next-tensorfold      # scripts/supervise.py:111
  local pid
  if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    pid="$(cat "$PIDFILE")"
    doit kill "$pid"
    if [ "$CHECK" = 0 ]; then
      for _ in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
      kill -0 "$pid" 2>/dev/null && kill -9 "$pid"
      rm -f "$PIDFILE"
    fi
  else
    ok "supervisor not running (pid file: $PIDFILE)"
  fi
  if [ -n "$(docker ps -q --filter "name=^${name}\$" 2>/dev/null)" ]; then
    doit docker stop "$name"                   # only reached if the supervisor could not clean up
  fi
  ok "stopped"
}

curl_test() {     # the repo's own example request (README.md:151-153)
  cat <<EOF
curl http://127.0.0.1:${PORT}/v1/chat/completions \\
  -H 'Content-Type: application/json' \\
  -d '{"model":"qwen38-flash-next-tf405","messages":[{"role":"user","content":"Write a Python function that validates a UUID."}],"max_tokens":2048,"temperature":0,"stream":false}'
EOF
}
# ===================== end of recipe config =================================

# ---------------------------------------------------------------------------
# Shared machinery (identical in every sxuff recipe repo).
# ---------------------------------------------------------------------------
CHECK=0; ACTION=run
usage() {
  cat <<EOF
Usage: ./run.sh [--check] [--stop] [--help]

Sets up and launches ${RECIPE_NAME} on one NVIDIA GB10 (SM121).

  (no flag)  check the machine, build or pull the pinned engine, download the
             pinned weights, verify SHA-256, start the server, print a test.
             Safe to re-run: finished steps are skipped.
  --check    print what would happen; change nothing.
  --stop     stop the server this script started.
  --help     show this text.

Environment:
  HF_TOKEN     Hugging Face token, read from the environment only, never saved.
  MODELS_DIR   weights        (default: ${MODELS_DIR})
  SRC_DIR      engine source  (default: ${SRC_DIR})
  STATE_DIR    pid and log    (default: ${STATE_DIR})
  PORT         server port    (default: ${PORT})

Expected: ${EXPECTED_LINE}
Receipt:  ${RECEIPT_PATH}
EOF
}
for a in "$@"; do
  case "$a" in
    --check) CHECK=1 ;;
    --stop)  ACTION=stop ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $a (see ./run.sh --help)" >&2; exit 2 ;;
  esac
done

say()  { printf '\n==> %s\n' "$*"; }
ok()   { printf '    ok    %s\n' "$*"; }
plan() { printf '    would %s\n' "$*"; }
die()  { printf '\nerror: %s\n' "$*" >&2; exit 1; }
CHECK_FAILED=0   # under --check, a failed pre-flight check is reported and the plan continues
fail() { if [ "$CHECK" = 1 ]; then printf '    FAIL  %s\n' "$*"; CHECK_FAILED=1; else die "$*"; fi; }
human(){ numfmt --to=iec --suffix=B "$1" 2>/dev/null || echo "$1 B"; }
doit() { if [ "$CHECK" = 1 ]; then plan "$*"; else "$@"; fi; }   # mutate, or describe under --check

PIDFILE="${STATE_DIR}/server.pid"
LOGFILE="${STATE_DIR}/server.log"
STOP_CMD="cd ${REPO_DIR} && ./run.sh --stop"

server_running() {
  if [ "$ENGINE_KIND" = docker ]; then
    [ -n "$(docker ps -q --filter "name=^${CONTAINER_NAME}\$" 2>/dev/null)" ]
  else
    [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null
  fi
}

stop_step() {   # may be overridden in the config block (e.g. to call scripts/stop.sh)
  if ! server_running; then ok "nothing started by this script is running"; return; fi
  if [ "$ENGINE_KIND" = docker ]; then doit docker stop "$CONTAINER_NAME"
  else
    local pid; pid="$(cat "$PIDFILE")"; doit kill "$pid"
    if [ "$CHECK" = 0 ]; then
      for _ in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
      kill -0 "$pid" 2>/dev/null && kill -9 "$pid"; rm -f "$PIDFILE"
    fi
  fi
  ok "stopped"
}
declare -f stop_override >/dev/null && stop_step() { stop_override; }

if [ "$ACTION" = stop ]; then say "Stopping ${RECIPE_NAME}"; stop_step; exit 0; fi
[ "$CHECK" = 1 ] && say "--check: nothing will be changed"

# git checkout of an exact commit, idempotent. Usage: git_pin <url> <sha> <dir>
git_pin() {
  local url="$1" sha="$2" dir="$3"
  if [ -d "$dir/.git" ] && [ "$(git -C "$dir" rev-parse HEAD)" = "$sha" ]; then ok "source at ${sha:0:12}: $dir"; return; fi
  if [ ! -d "$dir/.git" ]; then doit git clone --quiet "$url" "$dir"; fi
  if [ "$CHECK" = 1 ]; then plan "check out $sha in $dir"; return; fi
  [ -z "$(git -C "$dir" status --porcelain)" ] || die "$dir has local changes; refusing to move it to $sha"
  git -C "$dir" fetch --quiet origin "$sha" 2>/dev/null || git -C "$dir" fetch --quiet origin
  git -C "$dir" checkout --quiet --detach "$sha"
  [ "$(git -C "$dir" rev-parse HEAD)" = "$sha" ] || die "could not check out $sha in $dir"
  ok "source at ${sha:0:12}: $dir"
}

# docker image by digest, idempotent. Usage: image_pin <repo@sha256:...>
image_pin() {
  if docker image inspect "$1" >/dev/null 2>&1; then ok "image present: $1"; else doit docker pull "$1"; fi
}

# --- 1. machine ------------------------------------------------------------
say "Machine"
[ "$(uname -m)" = aarch64 ] || die "expected aarch64 (GB10), got $(uname -m)"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found"
GPU="$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1)"
case "$GPU" in *GB10*", 12.1") ok "GPU: $GPU" ;; *) die "measured on an NVIDIA GB10 (SM121); found: ${GPU:-none}" ;; esac
for t in $NEEDS_TOOLS; do command -v "$t" >/dev/null || die "missing tool: $t"; done
ok "tools: $NEEDS_TOOLS"
ALREADY=0; server_running && ALREADY=1
MEM_AVAIL=$(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) * 1024 ))
if [ "$ALREADY" = 1 ]; then ok "server already running; memory check skipped"
elif [ "$MEM_AVAIL" -lt "$NEED_MEM_BYTES" ]; then
  fail "need $(human "$NEED_MEM_BYTES") available memory, have $(human "$MEM_AVAIL"). Stop other models first (nvidia-smi, docker ps)."
else ok "memory: $(human "$MEM_AVAIL") available, need $(human "$NEED_MEM_BYTES")"; fi
if [ "$ALREADY" = 1 ]; then ok "port ${PORT}: served by this recipe"
else
  # The server binds 127.0.0.1 (or the host network). A listener on 127.0.0.1, a wildcard or ::1
  # blocks it; one bound to another interface address does not, and is only reported.
  PORT_ALL="$(ss -ltnH "sport = :${PORT}" 2>/dev/null || true)"
  PORT_CLASH="$(printf '%s\n' "$PORT_ALL" | awk '{print $4}' | grep -E '^(127\.0\.0\.1|0\.0\.0\.0|\*|\[::\]|\[::1\]|\[::ffff:127\.0\.0\.1\]):' || true)"
  PORT_DOCKER="$(docker ps --format '{{.Names}} {{.Ports}}' 2>/dev/null | grep -E "(127\.0\.0\.1|0\.0\.0\.0|\[::\]|:::?):${PORT}->" || true)"
  if [ -n "$PORT_CLASH$PORT_DOCKER" ]; then
    printf '    port %s is taken:\n%s\n%s\n' "$PORT" "$PORT_CLASH" "$PORT_DOCKER"
    fail "free port ${PORT} or run: PORT=<other> ./run.sh"
  elif [ -n "$PORT_ALL" ]; then
    ok "port ${PORT}: free on 127.0.0.1 (also bound elsewhere: $(printf '%s' "$PORT_ALL" | awk '{print $4}' | tr '\n' ' '))"
  else ok "port ${PORT}: free"; fi
fi

# --- 2. disk ---------------------------------------------------------------
say "Disk"
nearest_dir() { local d="$1"; while [ ! -d "$d" ]; do d="$(dirname "$d")"; done; echo "$d"; }
NEED_DISK=$ENGINE_DISK_BYTES
while read -r sha size repo rev rpath dest; do
  [ -z "${sha:-}" ] && continue
  have=0; [ -f "$MODELS_DIR/$dest" ] && have=$(stat -c %s "$MODELS_DIR/$dest")
  [ "$have" -lt "$size" ] && NEED_DISK=$(( NEED_DISK + size - have ))
done <<< "$WEIGHTS"
FREE_DISK=$(df -B1 --output=avail "$(nearest_dir "$MODELS_DIR")" | tail -1 | tr -d ' ')
if [ "$FREE_DISK" -ge "$NEED_DISK" ]; then ok "need $(human "$NEED_DISK") more, have $(human "$FREE_DISK")"
else fail "need $(human "$NEED_DISK") more disk under $MODELS_DIR, have $(human "$FREE_DISK"). Nothing was deleted."; fi

# --- 3. engine -------------------------------------------------------------
say "Engine"
engine_step

# --- 4. weights: pinned revision + SHA-256 -----------------------------------
hf_fetch() {   # repo rev path dest. The token reaches curl on stdin: never in argv or a file.
  { [ -n "${HF_TOKEN:-}" ] && printf 'header = "Authorization: Bearer %s"\n' "$HF_TOKEN"; true; } |
    curl --fail --location --retry 5 --retry-delay 5 --continue-at - --config - \
         --output "$4.part" "https://huggingface.co/$1/resolve/$2/$3"
  mv "$4.part" "$4"
}
default_weights_step() {
  while read -r sha size repo rev rpath dest; do
    [ -z "${sha:-}" ] && continue
    local f="$MODELS_DIR/$dest" mark="$MODELS_DIR/$dest.sha256-ok"
    if [ -f "$mark" ] && [ "$(cat "$mark")" = "$sha $size" ] && [ "$(stat -c %s "$f" 2>/dev/null)" = "$size" ]; then
      ok "verified earlier: $dest"; continue
    fi
    if [ "$(stat -c %s "$f" 2>/dev/null || echo 0)" != "$size" ]; then
      if [ "$CHECK" = 1 ]; then plan "download $repo@${rev:0:12} $rpath ($(human "$size"))"; continue; fi
      mkdir -p "$(dirname "$f")"; printf '    get   %s (%s)\n' "$rpath" "$(human "$size")"
      hf_fetch "$repo" "$rev" "$rpath" "$f"
    elif [ "$CHECK" = 1 ]; then plan "verify SHA-256 of $dest"; continue; fi
    local got; got="$(sha256sum "$f" | cut -d' ' -f1)"
    if [ "$got" != "$sha" ]; then mv "$f" "$f.sha256-mismatch"; die "SHA-256 mismatch for $dest: expected $sha, got $got (kept as $f.sha256-mismatch)"; fi
    echo "$sha $size" > "$mark"; ok "sha256 ok: $dest"
  done <<< "$WEIGHTS"
}
declare -f weights_step >/dev/null || weights_step() { default_weights_step; }
say "Weights"
weights_step

# --- 5. serve --------------------------------------------------------------
say "Server"
if [ "$ALREADY" = 1 ]; then ok "already running"
elif [ "$CHECK" = 1 ]; then serve_step
else
  mkdir -p "$STATE_DIR"; serve_step
  printf '    wait  for %s (up to %ss; log: %s)\n' "$READY_URL" "$READY_TIMEOUT" "$LOGFILE"
  t=0
  until curl -fsS -o /dev/null "$READY_URL" 2>/dev/null; do
    server_running || die "server exited during start-up; see $LOGFILE"
    [ "$t" -ge "$READY_TIMEOUT" ] && die "not ready after ${READY_TIMEOUT}s; see $LOGFILE"
    sleep 5; t=$((t + 5))
  done
  ok "ready after ${t}s"
fi

# --- 6. done ---------------------------------------------------------------
if [ "$CHECK" = 1 ]; then
  if [ "$CHECK_FAILED" = 1 ]; then say "Check finished with FAIL lines above; nothing was changed. ./run.sh would stop at the first one."; exit 1; fi
  say "Check finished; nothing was changed. Run ./run.sh to do it."; exit 0
fi
say "Test it"
curl_test
say "Stop it"
echo "$STOP_CMD"
echo
echo "Expected: ${EXPECTED_LINE} - receipt: ${RECEIPT_PATH}"

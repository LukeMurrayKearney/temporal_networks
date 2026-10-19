#!/usr/bin/env bash
# Pull new sweep results down from the cluster, skipping whatever is already here.
#
#   ./fetch_results.sh summary          # the two CSVs + manifest   (~30 MB)
#   ./fetch_results.sh shards           # + per-(cell,beta) shards + done markers
#   ./fetch_results.sh series           # + daily_series/*.csv.gz   (~100 MB)
#   ./fetch_results.sh logs             # + transmission_logs/*.csv.gz  (TENS OF GB)
#   ./fetch_results.sh all              # everything
#   ./fetch_results.sh shards --dry-run # show what WOULD transfer, move nothing
#
# Set the remote once, in your shell profile or on the command line:
#
#   REMOTE=matbvh@avon.scrtp.warwick.ac.uk ./fetch_results.sh shards
#
# RPATH defaults to "temporal_networks/seir_sweep", resolved RELATIVE TO YOUR
# REMOTE HOME. Leave it alone unless the repo lives somewhere else there.
#
# ---------------------------------------------------------------------------
# Do NOT write RPATH=~/something
# ---------------------------------------------------------------------------
# An unquoted tilde is expanded by YOUR shell before this script ever runs, so
# RPATH=~/temporal_networks/seir_sweep becomes /home/<your-local-user>/... and
# that path is then looked for ON THE CLUSTER, where it does not exist:
#
#   rsync: change_dir "/home/lmurraykearney/temporal_networks/..." failed:
#          No such file or directory (2)
#
# A remote path that starts with your LOCAL home is always this mistake, so the
# script detects it below and rewrites it to a relative path rather than failing
# eight times over. Use a bare relative path, or quote it: RPATH='~/foo'.
#
# ---------------------------------------------------------------------------
# Why rsync rather than scp/sftp
# ---------------------------------------------------------------------------
# scp has no notion of "already have it" -- it re-copies every byte every time.
# rsync compares size+mtime first and transfers only what differs, which is
# exactly "download only the new files" and is also restartable. The sweep's
# outputs suit it perfectly: shards, done-markers and the per-run .csv.gz files
# are written once and never edited, so after the first pull each later run
# moves only the files that appeared since.
#
# --ignore-existing is used for the write-once directories: it skips a file that
# exists locally without even stat-ing the remote copy, which is much faster over
# tens of thousands of files. The two summary CSVs and manifest.json are the
# exception -- they are REWRITTEN every time --collect runs -- so they are pulled
# in a separate pass without that flag, by the normal size+mtime comparison.
# Mixing the two in one call is the easy mistake: --ignore-existing would pin you
# to the first, partial runs_summary.csv you ever downloaded.
#
# If rsync is unavailable on the cluster, the sftp fallback is at the bottom.
# ---------------------------------------------------------------------------
set -uo pipefail

REMOTE="${REMOTE:-user@cluster.example.ac.uk}"
RPATH="${RPATH:-temporal_networks/seir_sweep}"      # relative to the REMOTE home
LOCAL="${LOCAL:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

WHAT="${1:-summary}"
shift || true
EXTRA=("$@")                                        # e.g. --dry-run

say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
warn() { printf '\033[33m   %s\033[0m\n' "$*"; }
err()  { printf '\033[31m   %s\033[0m\n' "$*"; }

# --- the locally-expanded-tilde trap ---------------------------------------
case "$RPATH" in
    "$HOME"/*)
        RPATH="${RPATH#"$HOME"/}"
        warn "RPATH pointed at your LOCAL home - your shell expanded the ~."
        warn "Using '$RPATH' relative to your remote home instead."
        ;;
    "~/"*) RPATH="${RPATH#\~/}" ;;
    /*)    ;;                                       # a real absolute remote path
esac

# --- one SSH connection, so 2FA is entered once ----------------------------
# Every transfer below is a separate rsync, and each would otherwise open its own
# SSH session and ask for a fresh 2FA code -- eight prompts for `all`.
# ControlMaster shares one authenticated connection between them; ControlPersist
# keeps it alive briefly after the last transfer so a follow-up run is free too.
CTL_DIR="${TMPDIR:-/tmp}/.fetch_results_ssh"
mkdir -p "$CTL_DIR" && chmod 700 "$CTL_DIR"
SSH_OPTS=(-o ControlMaster=auto -o "ControlPath=$CTL_DIR/%r@%h:%p" -o ControlPersist=10m)
SSH=(ssh "${SSH_OPTS[@]}")
RSH="ssh ${SSH_OPTS[*]}"

# -P = --partial --progress: a killed transfer resumes mid-file instead of
# restarting it, which matters for the multi-GB log directories.
RS=(rsync -avhP -e "$RSH")
# The sweep writes .gz files that are already compressed, so -z would burn CPU on
# both ends for nothing. Only the plain CSVs are worth compressing.
RS_TEXT=(rsync -avhPz -e "$RSH")

FAILED=0
SKIPPED=()

remote_has() { "${SSH[@]}" "$REMOTE" "test -e '$RPATH/$1'" 2>/dev/null; }

preflight() {
    say "connecting to $REMOTE"
    if ! "${SSH[@]}" -o BatchMode=no "$REMOTE" "test -d '$RPATH'"; then
        err "cannot find '$RPATH' on $REMOTE (or the connection failed)."
        err "Check the path - a typo there is the usual cause, e.g."
        err "  temporal_neworks  ->  temporal_networks"
        echo
        err "What IS there, for reference:"
        "${SSH[@]}" "$REMOTE" "ls -d ~/*/ 2>/dev/null | head -20" || true
        exit 1
    fi
    printf '   remote OK: %s:%s\n' "$REMOTE" "$RPATH"
}

# $1 = subdir under outputs*/ (empty for the summary pass), $2.. = rsync args
pull() {
    local sub="$1"; shift
    local d
    for d in outputs outputs_structure; do
        local rel="$d${sub:+/$sub}"
        if ! remote_has "$rel"; then
            SKIPPED+=("$rel")                       # not produced on the cluster yet
            continue
        fi
        mkdir -p "$LOCAL/$rel"
        if ! "$@" "${EXTRA[@]}" "$REMOTE:$RPATH/$rel/" "$LOCAL/$rel/"; then
            err "FAILED: $rel"
            FAILED=$((FAILED + 1))
        fi
    done
}

fetch_summary() {
    say "summary tables (re-pulled every time: --collect rewrites them)"
    pull "" "${RS_TEXT[@]}" \
        --include='runs_summary.csv' --include='network_summary.csv' \
        --include='manifest.json' --include='beta_r0_calibration.csv' \
        --include='fits_cache.pkl' --exclude='*'
}

fetch_dir() {
    say "$1 (new files only)"
    pull "$1" "${RS[@]}" --ignore-existing
}

disk_guard() {
    # The transmission logs are the one output that can fill a disk: the main
    # sweep's are already 12 GB here. Check before starting, not after.
    local avail
    avail=$(df -BG --output=avail "$LOCAL" | tail -1 | tr -dc '0-9')
    printf '   local free space: %s GB\n' "$avail"
    if [ "${avail:-0}" -lt 40 ]; then
        warn "under 40 GB free. The structure sweep adds ~21,000 transmission"
        warn "logs. Consider pulling only what you need, e.g."
        printf "     rsync -avhP --ignore-existing --include='*hh_static*' --exclude='*' \\\\\n"
        printf '       %s:%s/outputs_structure/transmission_logs/ ./outputs_structure/transmission_logs/\n' \
               "$REMOTE" "$RPATH"
        read -r -p '   continue anyway? [y/N] ' ans
        [ "$ans" = "y" ] || exit 1
    fi
}

case "$WHAT" in
    summary|shards|series|logs|all) ;;
    *) echo "usage: $0 {summary|shards|series|logs|all} [rsync flags]" >&2; exit 2 ;;
esac

preflight

case "$WHAT" in
    summary) fetch_summary ;;
    shards)  fetch_summary; fetch_dir shards; fetch_dir _done; fetch_dir gen_counts ;;
    series)  fetch_summary; fetch_dir shards; fetch_dir _done; fetch_dir gen_counts
             fetch_dir daily_series ;;
    logs)    disk_guard; fetch_dir transmission_logs ;;
    all)     disk_guard; fetch_summary; fetch_dir shards; fetch_dir _done
             fetch_dir gen_counts
             fetch_dir daily_series; fetch_dir transmission_logs ;;
esac

# ---------------------------------------------------------------------------
# Report honestly. An earlier version of this script ran every transfer with
# `|| true` and printed "done" whether or not anything moved, which turned a
# completely failed run into something that looked like a successful one.
# ---------------------------------------------------------------------------
if [ ${#SKIPPED[@]} -gt 0 ]; then
    say "not on the remote yet (skipped, not an error)"
    printf '   %s\n' "${SKIPPED[@]}"
fi

if [ "$FAILED" -gt 0 ]; then
    say "FINISHED WITH $FAILED FAILED TRANSFER(S)"
    exit 1
fi

say "done"
du -sh "$LOCAL"/outputs*/ 2>/dev/null
printf '   local done-markers: outputs=%s  outputs_structure=%s  (2100 = structure sweep complete)\n' \
    "$(ls "$LOCAL/outputs/_done" 2>/dev/null | wc -l)" \
    "$(ls "$LOCAL/outputs_structure/_done" 2>/dev/null | wc -l)"

# ---------------------------------------------------------------------------
# Watching a job finish without babysitting it
# ---------------------------------------------------------------------------
#   ssh "$REMOTE" "squeue -u \$USER"                            # still running?
#   ssh "$REMOTE" "ls $RPATH/outputs_structure/_done | wc -l"   # 2100 when done
#                                                               # (100 cells x 21 betas)
# The sweep commits each (cell, beta) as it finishes, so partial results are
# always consistent -- pulling mid-run is safe and the next pull picks up the
# rest. While the ControlPersist window is open those ssh calls need no 2FA.
#
# ---------------------------------------------------------------------------
# If rsync is not installed on the cluster
# ---------------------------------------------------------------------------
# lftp does the same "only what is missing or newer" mirror over plain sftp:
#
#   lftp -u "$USER", sftp://avon.scrtp.warwick.ac.uk -e "
#       mirror --only-newer --parallel=4 --verbose \
#           $RPATH/outputs_structure/shards \
#           $LOCAL/outputs_structure/shards ; bye"
#
# Add --ignore-time to compare on size alone (the sweep never rewrites a shard,
# so size is enough and it avoids re-pulling on a clock skew between machines).
#
# With nothing but openssh's own sftp, the closest equivalent is a reget of a
# listed set -- it has no directory-level "skip what I have", so build the list
# locally and ask only for the missing names:
#
#   ssh "$REMOTE" "ls $RPATH/outputs_structure/shards" > /tmp/remote.txt
#   ls  "$LOCAL/outputs_structure/shards"              > /tmp/local.txt
#   comm -23 <(sort /tmp/remote.txt) <(sort /tmp/local.txt) > /tmp/missing.txt
#   sed 's|^|get -a '"$RPATH"'/outputs_structure/shards/|' /tmp/missing.txt \
#       | cat - <(echo bye) \
#       | sftp -b - "$REMOTE:"
#
# (`get -a` resumes a partially transferred file rather than starting over.)

#!/usr/bin/env bash
set -euo pipefail

# ===== 用户可改参数 =====
END_DATE="2025-09-30"
DAYS="30"
MODEL="moirai2"
PKG="experiments0124change.run_experiment"

# 日志目录（自动创建）
LOG_DIR="logs_moirai2_${END_DATE}"
mkdir -p "${LOG_DIR}"

# 你要跑的实验列表（按顺序）
TYPES=(
  "tsfm_5"
  "tsfm_4"
  "tsfm_3"
  "tsfm_2"
  "tsfm_1"
  "baseline"
)

echo "[INFO] Start sweep: end_date=${END_DATE}, days=${DAYS}, model=${MODEL}"
echo "[INFO] Logs: ${LOG_DIR}"
echo

# 逐个跑
for T in "${TYPES[@]}"; do
  TS="$(date +%Y%m%d_%H%M%S)"
  LOG_FILE="${LOG_DIR}/${TS}__${T}.log"

  echo "============================================================"
  echo "[INFO] Running: type=${T}  (log: ${LOG_FILE})"
  echo "CMD: python -m ${PKG} --type ${T} --days ${DAYS} --end-date ${END_DATE} --model ${MODEL}"
  echo "============================================================"

  # 用 tee 同时输出到屏幕和日志
  python -m "${PKG}" \
    --type "${T}" \
    --days "${DAYS}" \
    --end-date "${END_DATE}" \
    --model "${MODEL}" 2>&1 | tee "${LOG_FILE}"

  echo "[INFO] Finished: ${T}"
  echo
done

echo "[INFO] All experiments finished successfully."

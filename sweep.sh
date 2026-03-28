#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/run_server_experiment.sh"

# ===== 用户可改参数 =====
END_DATE="2025-09-30"
DAYS="30"
MODEL="moirai2"

# 日志目录（自动创建）
LOG_DIR="logs_moirai2_${END_DATE}"
mkdir -p "${LOG_DIR}"

# 运行类型顺序改由统一 registry 提供，但保持原来的选择和顺序
mapfile -t TYPES < <(
  python3 - <<PY
import importlib.util
from pathlib import Path

module_path = Path("${SCRIPT_DIR}") / "format_registry.py"
spec = importlib.util.spec_from_file_location("format_registry", module_path)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

desired_order = ("tsfm_5", "tsfm_4", "tsfm_3", "tsfm_2", "tsfm_1", "baseline")
available = {format_spec.cli_name for format_spec in module.FORMAT_SPECS}
for cli_name in desired_order:
    if cli_name in available:
        print(cli_name)
PY
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
  echo "CMD: bash ${RUNNER} --type ${T} --days ${DAYS} --end-date ${END_DATE} --model ${MODEL}"
  echo "============================================================"

  # 用统一 wrapper 固定服务器环境；timesfm 会自动注入官方 2.5 torch API。
  bash "${RUNNER}" \
    --type "${T}" \
    --days "${DAYS}" \
    --end-date "${END_DATE}" \
    --model "${MODEL}" 2>&1 | tee "${LOG_FILE}"

  echo "[INFO] Finished: ${T}"
  echo
done

echo "[INFO] All experiments finished successfully."

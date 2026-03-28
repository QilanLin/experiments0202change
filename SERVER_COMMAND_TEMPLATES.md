# Server Command Templates

服务器默认目录：

```bash
cd /root/private_data/experiments0202change
```

统一入口：

```bash
bash ./run_server_experiment.sh ...
```

说明：

- `timesfm` 会自动走官方 `timesfm_official/src`
- `chronos` / `moirai2` / `toto` 也可以走同一个 wrapper
- wrapper 会自动固定：
  - `HF_HOME=/root/private_data/.cache`
  - `HUGGINGFACE_HUB_CACHE=/root/private_data/.cache/huggingface/hub`
  - `HF_HUB_OFFLINE=1`
  - `TRANSFORMERS_OFFLINE=1`

## 1. 前台单跑

### baseline

```bash
cd /root/private_data/experiments0202change
bash ./run_server_experiment.sh \
  --type baseline \
  --start-date 2025-08-31 \
  --end-date 2025-09-30
```

### chronos

```bash
cd /root/private_data/experiments0202change
bash ./run_server_experiment.sh \
  --type tsfm_7a \
  --model chronos \
  --start-date 2025-08-31 \
  --end-date 2025-09-30
```

### timesfm

```bash
cd /root/private_data/experiments0202change
bash ./run_server_experiment.sh \
  --type tsfm_7a \
  --model timesfm \
  --start-date 2025-08-31 \
  --end-date 2025-09-30
```

### moirai2

```bash
cd /root/private_data/experiments0202change
bash ./run_server_experiment.sh \
  --type tsfm_7a \
  --model moirai2 \
  --start-date 2025-08-31 \
  --end-date 2025-09-30
```

### toto

```bash
cd /root/private_data/experiments0202change
bash ./run_server_experiment.sh \
  --type tsfm_7a \
  --model toto \
  --start-date 2025-08-31 \
  --end-date 2025-09-30
```

## 2. 后台 detached 跑

### 通用模板

```bash
cd /root/private_data/experiments0202change
RUN_NAME="timesfm_7a_2025-09_$(date +%Y%m%d_%H%M%S)"
mkdir -p "/root/private_data/logs/${RUN_NAME}"
nohup bash -lc "
  cd /root/private_data/experiments0202change && \
  bash ./run_server_experiment.sh \
    --type tsfm_7a \
    --model timesfm \
    --start-date 2025-08-31 \
    --end-date 2025-09-30
" > "/root/private_data/logs/${RUN_NAME}/run.log" 2>&1 < /dev/null & echo $!
```

### 3 个月 TimesFM

```bash
cd /root/private_data/experiments0202change
RUN_NAME="timesfm_7a_3months_$(date +%Y%m%d_%H%M%S)"
mkdir -p "/root/private_data/logs/${RUN_NAME}"
nohup bash -lc "
  cd /root/private_data/experiments0202change && \
  bash ./run_server_experiment.sh \
    --type tsfm_7a \
    --model timesfm \
    --start-date 2025-08-31 \
    --end-date 2026-01-31
" > "/root/private_data/logs/${RUN_NAME}/run.log" 2>&1 < /dev/null & echo $!
```

## 3. 看进度

### 看日志尾部

```bash
tail -n 40 /root/private_data/logs/<RUN_NAME>/run.log
```

### 持续盯日志

```bash
tail -f /root/private_data/logs/<RUN_NAME>/run.log
```

### 看 GPU

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
```

### 看最新结果目录

```bash
ls -td /root/private_data/experiment_results/*/* | head
```

### 看某次 run 的产物数量

```bash
RUN_DIR="/root/private_data/experiment_results/llm_tsfm_format_7a/20260328_093137"
find "${RUN_DIR}/llm_inputs" -type f | wc -l
find "${RUN_DIR}/llm_outputs" -type f | wc -l
```

## 4. 跑完后看结果

### 直接看 summary

```bash
cat /root/private_data/experiment_results/llm_tsfm_format_7a/<RUN_ID>/simulation_result.json
```

### 只抽关键指标

```bash
python - <<'PY'
import json
path = "/root/private_data/experiment_results/llm_tsfm_format_7a/<RUN_ID>/simulation_result.json"
with open(path) as f:
    x = json.load(f)
print("final_value", x["final_value"])
print("total_return", x["total_return"])
print("sharpe_ratio", x["sharpe_ratio"])
print("max_drawdown", x["max_drawdown"])
PY
```

## 5. 和旧 run 对比

```bash
python - <<'PY'
import json
old_path = "/root/private_data/worktrees/e7ff19f/experiments0202change/experiment_results/llm_tsfm_format_7a/20260325_021835/simulation_result.json"
new_path = "/root/private_data/experiment_results/llm_tsfm_format_7a/20260328_093137/simulation_result.json"
with open(old_path) as f:
    old = json.load(f)
with open(new_path) as f:
    new = json.load(f)
for key in ["final_value", "total_return", "sharpe_ratio", "max_drawdown"]:
    print(key, "old=", old[key], "new=", new[key], "diff=", new[key] - old[key])
PY
```

## 6. 直接比较 tsfm 输出

```bash
python - <<'PY'
import json
old_path = "/root/private_data/worktrees/e7ff19f/experiments0202change/experiment_results/llm_tsfm_format_7a/20260325_021835/tsfm_outputs/AAPL_2025-09-02.json"
new_path = "/root/private_data/experiment_results/llm_tsfm_format_7a/20260328_093137/tsfm_outputs/AAPL_2025-09-02.json"
with open(old_path) as f:
    old = json.load(f)
with open(new_path) as f:
    new = json.load(f)
for key in ["ratio_1d", "ratio_1w", "ratio_4w"]:
    print(key, "old=", old[key], "new=", new[key])
print("old first5", old["numeric_30d"][:5])
print("new first5", new["numeric_30d"][:5])
PY
```

## 7. 常用安全检查

### 确认当前 timesfm 会走官方 API

```bash
cd /root/private_data/experiments0202change
bash ./run_server_experiment.sh --model timesfm --help | sed -n '1,20p'
```

期望看到：

```text
[INFO] TimesFM official source: /root/private_data/timesfm_official/src
```

### 确认代码版本

```bash
cd /root/private_data/experiments0202change
git rev-parse HEAD
git status --short
```

## 8. 一键模板

### 2025-09 + format_7a + timesfm

```bash
cd /root/private_data/experiments0202change
bash ./run_server_experiment.sh \
  --type tsfm_7a \
  --model timesfm \
  --start-date 2025-08-31 \
  --end-date 2025-09-30
```

### 2025-09 + format_7a + chronos

```bash
cd /root/private_data/experiments0202change
bash ./run_server_experiment.sh \
  --type tsfm_7a \
  --model chronos \
  --start-date 2025-08-31 \
  --end-date 2025-09-30
```

### 2025-09 + format_2 + timesfm

```bash
cd /root/private_data/experiments0202change
bash ./run_server_experiment.sh \
  --type tsfm_2 \
  --model timesfm \
  --start-date 2025-08-31 \
  --end-date 2025-09-30
```

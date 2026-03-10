# Portfolio Weight Allocation Experiments

LLM vs LLM+TSFM (Chronos-2 / TimeFM) 对比实验框架

## 实验设计

### 目标
比较纯LLM和LLM+TSFM在MAG7股票组合权重分配任务上的表现。

### 实验设置
- **交易对象**: AAPL, GOOGL, AMZN, MSFT, META, TSLA, NVDA (MAG7)
- **本金**: $1,000,000
- **交易方式**: Fractional shares（权重交易）
- **模拟周期**: 30天（可配置）
- **再平衡频率**: 每日

### LLM配置
- **Debug模式**: `Qwen/Qwen3-4B-Instruct-2507`
- **生产模式**: `Qwen/Qwen3-30B-A3B-Thinking-2507`

### 实验类型

| 类型 | 描述 |
|------|------|
| `baseline` | 仅LLM（输入：基本面 + 过去一年股价） |
| `tsfm_1` | LLM + TSFM格式1：数字，30天预测 |
| `tsfm_2` | LLM + TSFM格式2：比例，30天预测 |
| `tsfm_3` | LLM + TSFM格式3：比例，多时间窗口（1天/1周/2周/3周/4周） |
| `tsfm_4` | LLM + TSFM格式4：数字，分位数{0.05,0.25,0.5,0.75,0.95}，30天 |
| `tsfm_5` | LLM + TSFM格式5：比例，分位数，30天 |
| `tsfm_6` | LLM + TSFM格式6：比例，分位数，多时间窗口 |

### 评估指标
- 累计收益 (Total Return)
- 年化收益 (Annualized Return)
- Sharpe Ratio
- Sortino Ratio
- 最大回撤 (Max Drawdown)

## 使用方法

### 1. 环境准备

```bash
# 设置Alpha Vantage API Key
export ALPHA_VANTAGE_API_KEY="your_api_key"

# 或在.env文件中设置
echo "ALPHA_VANTAGE_API_KEY=your_api_key" >> .env
```

### 2. 本地调试（无需GPU）

```bash
# 使用Mock模式测试流程
python -m experiments.run_experiment --type baseline --mock --days 5
python -m experiments.run_experiment --type tsfm_3 --mock --days 5
python -m experiments.run_all_experiments --mock --days 5
```

### 3. GPU服务器完整实验

```bash
# 方法1: 使用启动脚本（推荐）
bash run_experiment.sh          # Linux
run_experiment.bat              # Windows

# 方法2: 直接运行
python experiments/run_full_experiment.py --days 30

# 方法3: 使用30B生产模型
python experiments/run_full_experiment.py --production --days 30

# 方法4: 后台运行（防止SSH断开）
nohup python experiments/run_full_experiment.py --days 30 > experiment.log 2>&1 &
```

### 4. 单独运行实验

```bash
# Baseline实验
python -m experiments.run_experiment --type baseline --debug --days 30

# TSFM格式3实验（使用默认 Chronos 模型）
python -m experiments.run_experiment --type tsfm_3 --debug --days 30

# TSFM格式3实验（使用 TimeFM 模型）
python -m experiments.run_experiment --type tsfm_3 --debug --days 30 --model timesfm
```

### 5. 批量运行

```bash
# 运行所有7个实验
python -m experiments.run_all_experiments --debug --days 30

# 只运行特定TSFM格式
python -m experiments.run_all_experiments --debug --formats 1 3 6
```

### 4. 对比实验结果

```bash
python -m experiments.compare_results --results-dir ./experiment_results
```

## 输出结构

```
experiment_results/
├── baseline_llm_only/
│   └── 20240115_143022/
│       ├── simulation_result.json    # 完整模拟结果
│       ├── llm_outputs/              # LLM中间输出
│       │   ├── decision_2024-01-01.json
│       │   └── ...
│       └── tsfm_outputs/             # TSFM预测输出（如果有）
│           ├── AAPL_2024-01-01.json
│           └── ...
├── llm_tsfm_format_3/
│   └── ...
├── batch_run_20240115_150000.json    # 批量运行结果
└── comparison_report.txt             # 对比报告
```

## TSFM输出格式详解

### 格式1: 数字，30天
```
TSFM Forecast for AAPL (30-day price prediction):
Day 1-5: ['185.23', '186.45', '187.12', '186.89', '188.01']
Day 26-30: ['192.34', '193.12', '192.89', '194.01', '195.23']
```

### 格式2: 比例，30天
```
TSFM Forecast for AAPL (30-day return prediction):
Day 1-5: ['0.52%', '1.18%', '1.54%', '1.42%', '2.03%']
Day 26-30: ['4.38%', '4.81%', '4.68%', '5.29%', '5.95%']
```

### 格式3: 比例，多时间窗口
```
TSFM Forecast for AAPL (multi-horizon returns):
1 Day: 0.52%
1 Week: 2.03%
2 Weeks: 3.45%
3 Weeks: 4.12%
4 Weeks: 5.29%
```

### 格式4: 数字，分位数，30天
```
TSFM Forecast for AAPL (30-day quantile prices):
Median (50%): Day30=$195.23
5th percentile: Day30=$182.45
95th percentile: Day30=$208.67
```

### 格式5: 比例，分位数，30天
```
TSFM Forecast for AAPL (30-day quantile returns):
  0.05 quantile: -1.23%
  0.25 quantile: 2.34%
  0.5 quantile: 5.95%
  0.75 quantile: 9.12%
  0.95 quantile: 13.45%
```

### 格式6: 比例，分位数，多时间窗口
```
TSFM Forecast for AAPL (quantile returns by horizon):
  1d: [-0.5%, 0.5%, 1.2%]
  1w: [-1.2%, 2.0%, 4.5%]
  2w: [-2.1%, 3.5%, 7.2%]
  3w: [-2.8%, 4.1%, 9.1%]
  4w: [-3.5%, 5.3%, 11.2%]
```

## 代码结构

```
experiments/
├── __init__.py
├── config.py              # 实验配置
├── data_loader.py         # Alpha Vantage数据加载
├── tsfm_forecaster.py     # Chronos-2预测模块
├── portfolio_agent.py     # 组合权重决策Agent
├── simulator.py           # 组合模拟器
├── run_experiment.py      # 单实验运行脚本
├── run_all_experiments.py # 批量运行脚本
├── compare_results.py     # 结果对比分析
└── README.md
```

## TSFM 模型选择

实验支持两种时序预测模型：

- **Chronos-2** (默认): Amazon 的 Chronos-2 模型，需要安装 `chronos-forecasting` 包
- **TimeFM**: Google 的 TimeFM 模型，需要安装 `timesfm` 包

### 使用 TimeFM

```bash
# 安装 TimeFM 依赖
pip install timesfm torch numpy

# 运行实验时指定模型
python -m experiments.run_experiment --type tsfm_3 --model timesfm --days 30
```

### 模型对比

两种模型使用相同的接口，输出格式完全兼容，可以无缝切换：

```bash
# 使用 Chronos（默认）
python -m experiments.run_experiment --type tsfm_3 --model chronos

# 使用 TimeFM
python -m experiments.run_experiment --type tsfm_3 --model timesfm
```

## 注意事项

1. **API限制**: Alpha Vantage免费API限制5次/分钟，代码已内置延迟处理
2. **数据缓存**: 数据会缓存到`./data_cache`目录，24小时内不会重复请求
3. **GPU加速**: Chronos-2 和 TimeFM 都会自动检测并使用GPU（如果可用）
4. **Fallback机制**: 如果TSFM模型不可用，会使用naive forecast（假设价格不变）
5. **模型依赖**: 
   - Chronos-2: 需要 `chronos-forecasting>=2.0`
   - TimeFM: 需要 `timesfm>=0.1.0`, `torch>=2.0.0`, `numpy>=1.24.0`

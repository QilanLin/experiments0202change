# GPU服务器实验完整指南

## 1. 上传文件到GPU服务器

```bash
# 方法1: 使用scp上传整个项目
scp -r /path/to/tradingagents_tsfm_modified_v5 user@gpu-server:/home/user/

# 方法2: 使用rsync（推荐，支持断点续传）
rsync -avz --progress /path/to/tradingagents_tsfm_modified_v5 user@gpu-server:/home/user/

# 方法3: 先打包再上传
tar -czvf project.tar.gz tradingagents_tsfm_modified_v5
scp project.tar.gz user@gpu-server:/home/user/
# 在服务器上解压
ssh user@gpu-server "cd /home/user && tar -xzvf project.tar.gz"
```

## 2. 服务器环境配置

```bash
# SSH登录服务器
ssh user@gpu-server

# 进入项目目录
cd /home/user/tradingagents_tsfm_modified_v5

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux
# 或 .\venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 安装额外依赖（如果需要）
pip install chronos-forecasting>=2.0
pip install python-dotenv

# 验证GPU可用
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

## 3. 配置环境变量

```bash
# 创建.env文件
cat > .env << EOF
ALPHA_VANTAGE_API_KEY=your_api_key_here
EOF

# 或者直接export
export ALPHA_VANTAGE_API_KEY=your_api_key_here
```

## 4. 运行实验

### 4.0 TimesFM 服务器运行约束
```bash
# TimesFM 请统一走项目自带 wrapper。
# 这样会自动固定：
#   - 官方 timesfm_official/src
#   - HF_HOME / HUGGINGFACE_HUB_CACHE
#   - HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE
#
# 目录约定：
#   /root/private_data/experiments0202change
#   /root/private_data/timesfm_official
#
# 示例：
bash run_server_experiment.sh \
  --type tsfm_7a \
  --model timesfm \
  --start-date 2025-08-31 \
  --end-date 2025-09-30
```

常用命令模板见：

- `SERVER_COMMAND_TEMPLATES.md`

### 4.1 快速测试（验证环境）
```bash
# 先用mock模式测试流程
python -m experiments.run_experiment --type baseline --mock --days 5

# 测试LLM是否正常工作
python -m experiments.run_experiment --type baseline --debug --days 5
```

### 4.2 完整实验运行

```bash
# 方法1: 使用批量运行脚本（推荐）
python -m experiments.run_all_experiments --debug --days 30

# 方法2: 使用专用实验脚本
python experiments/run_full_experiment.py

# 方法3: 后台运行（防止SSH断开）
nohup python -m experiments.run_all_experiments --debug --days 30 > experiment.log 2>&1 &

# 方法4: 使用screen（推荐，可以重新连接）
screen -S experiment
python -m experiments.run_all_experiments --debug --days 30
# Ctrl+A+D 分离会话
# screen -r experiment 重新连接
```

### 4.3 使用生产模型运行
```bash
# 使用30B模型（需要更多显存）
python -m experiments.run_all_experiments --days 30
```

## 5. 监控实验进度

```bash
# 查看日志
tail -f experiment.log

# 查看GPU使用情况
watch -n 1 nvidia-smi

# 查看已完成的实验
ls -la experiment_results/
```

## 6. 下载结果

```bash
# 在本地执行
scp -r user@gpu-server:/home/user/tradingagents_tsfm_modified_v5/experiment_results ./

# 或使用rsync
rsync -avz user@gpu-server:/home/user/tradingagents_tsfm_modified_v5/experiment_results ./
```

## 7. 常见问题

### GPU显存不足
```bash
# 使用较小的模型
# 修改 experiments/config.py 中的 debug_llm
"debug_llm": "Qwen/Qwen3-4B-Instruct-2507"  # 4B模型，约8GB显存
```

### API限流
```bash
# Alpha Vantage免费API限制5次/分钟
# 数据会自动缓存，第二次运行会更快
# 如果遇到限流，等待1分钟后重试
```

### 模型下载慢
```bash
# 设置HuggingFace镜像（中国大陆）
export HF_ENDPOINT=https://hf-mirror.com

# 或预先下载模型
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-4B-Instruct-2507')"
```

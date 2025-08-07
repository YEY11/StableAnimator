#!/bin/bash

# 脚本功能：根据项目关键字，安全地查找并终止当前用户在特定项目下的 train.py 进程
# 优化：如果没有提供关键字，则自动使用当前目录名作为默认关键字。

# 检查用户是否提供了项目关键字作为参数
if [ -z "$1" ]; then
  # 如果没有提供参数，则使用当前工作目录的名称作为默认关键字
  PROJECT_KEYWORD=$(basename "$(pwd)")
  echo "ℹ️  未提供项目关键字，将使用当前目录名 '$PROJECT_KEYWORD' 作为默认值。"
else
  # 如果提供了参数，则使用该参数
  PROJECT_KEYWORD="$1"
  echo "👍  使用指定的项目关键字: '$PROJECT_KEYWORD'"
fi

echo "=========================================================="
echo "正在查找属于用户 '$USER' 且与项目 '$PROJECT_KEYWORD' 相关的 'train.py' 进程..."
echo "=========================================================="

# 使用 pgrep 进行更精确和安全的查找
# -u "$USER": 限制为当前用户
# -f: 匹配完整的命令行参数，而不仅仅是进程名
# "$PROJECT_KEYWORD.*train.py": 一个正则表达式，匹配同时包含项目关键字和 "train.py" 的命令行
PIDS=$(pgrep -u "$USER" -f "$PROJECT_KEYWORD.*train.py")

# 检查是否找到了任何进程
if [ -z "$PIDS" ]; then
  echo "✅ 恭喜！没有找到与项目 '$PROJECT_KEYWORD' 相关的 train.py 进程。"
else
  echo "找到以下进程 PID，将执行 kill -9:"
  echo "--------------------"
  # 为了清晰，将找到的PID打印成列表
  echo "$PIDS" | tr ' ' '\n'
  echo "--------------------"
  
  # 执行 kill 命令
  kill -9 $PIDS
  
  echo "✅ 终止命令已发送！"
  echo "请稍后使用 'ps -ef | grep train.py' 来验证进程是否已完全停止。"
fi

echo "=========================================================="

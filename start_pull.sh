#!/bin/bash
# start_pull.sh - 从远程 GitHub 仓库拉取最新代码

git config user.email "guoweiyang2024@163.com"
git config user.name "yuki-younai"

set -e

echo "========== Git Pull =========="
echo "正在从远程仓库拉取最新代码..."
# 放弃所有本地修改（包括暂存区）
git reset --hard HEAD

# 删除所有未跟踪的文件和目录（⚠️ 危险，慎用）
git clean -fd

# 再拉取最新
git pull origin main
echo "========== 拉取完成 =========="

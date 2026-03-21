#!/bin/bash
# start_push.sh - 推送本地修改到远程 GitHub 仓库

set -e

# 获取 commit message，默认为 "update"
COMMIT_MSG="${1:-update}"

echo "========== Git Push =========="
echo "正在添加所有修改..."
git add -A

# 检查是否有需要提交的内容
if git diff --cached --quiet; then
    echo "没有需要提交的修改。"
    exit 0
fi

echo "正在提交修改: ${COMMIT_MSG}"
git commit -m "${COMMIT_MSG}"

echo "正在推送到远程仓库..."
git push origin main

echo "========== 推送完成 =========="

#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Updating GitLab Repository${NC}"
echo "=================================="

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    echo -e "${RED}❌ Error: Not in a git repository${NC}"
    exit 1
fi

# Get commit message from user or use default
if [ -z "$1" ]; then
    echo -e "${YELLOW}💬 Enter commit message (or press Enter for default):${NC}"
    read -r commit_message
    if [ -z "$commit_message" ]; then
        commit_message="Update trading bot project - $(date '+%Y-%m-%d %H:%M')"
    fi
else
    commit_message="$*"
fi

echo -e "${BLUE}📋 Checking repository status...${NC}"
git status --porcelain

# Check if there are any changes
if [ -z "$(git status --porcelain)" ]; then
    echo -e "${YELLOW}⚠️  No changes detected to commit${NC}"
    echo -e "${BLUE}🔄 Checking if local is behind remote...${NC}"
    git fetch
    if [ "$(git rev-parse HEAD)" != "$(git rev-parse @{u})" ]; then
        echo -e "${BLUE}📥 Pulling latest changes...${NC}"
        git pull origin main
        echo -e "${GREEN}✅ Repository updated!${NC}"
    else
        echo -e "${GREEN}✅ Repository is already up to date${NC}"
    fi
    exit 0
fi

echo -e "${BLUE}📦 Adding all changes...${NC}"
git add .

echo -e "${BLUE}💾 Committing changes...${NC}"
echo -e "${YELLOW}📝 Commit message: ${commit_message}${NC}"
git commit -m "$commit_message"

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Commit failed${NC}"
    exit 1
fi

echo -e "${BLUE}🌐 Pushing to GitLab...${NC}"
git push origin main

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}🎉 Successfully updated GitLab!${NC}"
    echo -e "${GREEN}✅ All changes have been pushed to remote repository${NC}"
    
    # Show latest commit info
    echo ""
    echo -e "${BLUE}📊 Latest commit:${NC}"
    git log --oneline -1
    
    # Show repository URL
    echo ""
    echo -e "${BLUE}🔗 Repository URL:${NC}"
    git remote get-url origin
else
    echo -e "${RED}❌ Push failed${NC}"
    echo -e "${YELLOW}💡 You may need to pull latest changes first:${NC}"
    echo "   git pull origin main"
    exit 1
fi

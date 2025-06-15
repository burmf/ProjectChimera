# 🚀 ProjectChimera - Git Workflow Guide

**Professional Git Management for Trading System Development**

## 📋 Current Repository Status

### **Branch Structure**
```
main (production-ready)    ← Current
├── v2.0.0-clean-architecture (tagged)
└── develop (development)
```

### **Repository Statistics**
- **Total Commits**: 15
- **Files Tracked**: 107
- **Current Version**: v2.0.0-clean-architecture
- **Architecture**: Clean, modular design

---

## 🏗️ Directory Structure (Git Tracked)

| Directory | Files | Purpose |
|-----------|-------|---------|
| `core/`     | 33    | API client, risk management, core logic |
| `tests/`    | 14    | All testing files organized |
| `archive/`  | 10    | Legacy files, old versions |
| `data/`     | 8     | Performance data, results |
| `systems/`  | 4     | Trading systems, deployment |
| `ui/`       | 3     | User interfaces, dashboards |
| `docs/`     | 1     | Documentation |
| Root        | 34    | Config, launcher, documentation |

---

## 🚀 Git Workflow Commands

### **Daily Development**
```bash
# Check status
git status

# Stage changes
git add .

# Commit with meaningful message
git commit -m "feat(component): description

🎯 Changes made:
- Specific change 1
- Specific change 2

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"

# Push to remote
git push origin main
```

### **Feature Development**
```bash
# Create feature branch
git checkout -b feature/new-feature

# Develop and commit
git add .
git commit -m "feat(feature): implement new feature"

# Merge back to main
git checkout main
git merge feature/new-feature
git branch -d feature/new-feature
```

### **Version Management**
```bash
# Create version tag
git tag -a v2.1.0 -m "Version 2.1.0 - New Features"

# List all tags
git tag -l

# Push tags
git push origin --tags
```

---

## 📝 Commit Message Convention

### **Format**
```
<type>(<scope>): <subject>

<body>

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

### **Types**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting
- `refactor`: Code refactoring
- `test`: Tests
- `chore`: Build/config changes

### **Examples**
```bash
# New feature
git commit -m "feat(trading): add 50x leverage support

🎯 Enhanced leverage capabilities:
- Maximum leverage increased to 50x
- Dynamic risk adjustment
- Improved profit targets

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"

# Bug fix
git commit -m "fix(api): resolve Bitget connection timeout

🐛 Fixed connection issues:
- Increased timeout to 30 seconds
- Added retry mechanism
- Better error handling

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## 🔄 Branching Strategy

### **Main Branches**
- `main`: Production-ready code
- `develop`: Integration branch for features

### **Supporting Branches**
- `feature/`: New features (`feature/advanced-alerts`)
- `hotfix/`: Critical fixes (`hotfix/api-timeout`)
- `release/`: Release preparation (`release/v2.1.0`)

### **Workflow**
```bash
# Start new feature
git checkout develop
git checkout -b feature/new-feature

# Development...
git add .
git commit -m "feat: implement feature"

# Finish feature
git checkout develop
git merge feature/new-feature
git branch -d feature/new-feature

# Release to main
git checkout main
git merge develop
git tag -a v2.1.0 -m "Release v2.1.0"
```

---

## 🛡️ File Management

### **Files Tracked by Git**
✅ **Source Code**: All `.py` files in organized directories  
✅ **Configuration**: `.env.example`, configs  
✅ **Documentation**: `README.md`, `docs/`  
✅ **Build Files**: `Dockerfile`, `docker-compose.yml`  

### **Files Ignored (.gitignore)**
❌ **Secrets**: `.env`, API keys  
❌ **Logs**: `logs/*.log`, runtime logs  
❌ **Cache**: `__pycache__/`, `.cache/`  
❌ **Data**: Large data files, temporary files  

---

## 📊 Repository Health

### **Quality Metrics**
- ✅ **Clean History**: Meaningful commit messages
- ✅ **Organized Structure**: Modular directory layout
- ✅ **Proper Ignoring**: Sensitive files excluded
- ✅ **Tagged Versions**: Release tracking

### **Best Practices**
- 🎯 **Atomic Commits**: One logical change per commit
- 📝 **Clear Messages**: Descriptive commit messages
- 🏷️ **Version Tags**: Tag important releases
- 🔒 **Secure**: Never commit secrets or keys

---

## 🚀 Quick Reference

### **Check Current State**
```bash
git status                    # Working directory status
git log --oneline -5         # Recent commits
git branch -a                # All branches
git tag -l                   # All tags
```

### **Common Operations**
```bash
git add -A                   # Stage all changes
git commit --amend           # Modify last commit
git reset HEAD~1             # Undo last commit
git stash                    # Temporarily save changes
```

### **Remote Operations**
```bash
git push origin main         # Push to main branch
git pull origin main         # Pull latest changes
git push origin --tags       # Push all tags
git remote -v                # Show remotes
```

---

## 🎯 Current Achievement

**✅ Clean Architecture v2.0.0 Successfully Committed!**

- **Perfect Organization**: All files properly categorized
- **Professional Structure**: Industry-standard layout
- **Version Control**: Properly tagged and documented
- **Ready for Production**: Clean, maintainable codebase

```bash
# Your current position
Branch: main
Commit: 753be1f
Tag: v2.0.0-clean-architecture
Status: Clean and organized 🚀
```

---

**Ready to continue developing your profit-maximizing trading system with professional Git workflow!** 💰✨
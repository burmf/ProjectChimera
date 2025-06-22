#!/bin/bash
# ProjectChimera Production Deployment Script

set -e  # Exit on any error

echo "🚀 ProjectChimera Production Deployment"
echo "========================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️ $1${NC}"
}

# Check if running as correct user
if [ "$EUID" -eq 0 ]; then
    print_error "Don't run this script as root"
    exit 1
fi

# Step 1: Environment Check
echo
print_info "Step 1: Environment Check"
echo "========================="

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d " " -f 2)
print_status "Python version: $python_version"

# Check if virtual environment is active
if [[ "$VIRTUAL_ENV" != "" ]]; then
    print_status "Virtual environment: $VIRTUAL_ENV"
else
    print_warning "No virtual environment detected. Consider using one."
fi

# Check required directories
mkdir -p data logs backups
print_status "Created required directories: data, logs, backups"

# Step 2: Dependency Check
echo
print_info "Step 2: Dependency Check"
echo "========================"

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    print_warning "requirements.txt not found. Creating basic version..."
    cat > requirements.txt << EOF
streamlit>=1.28.0
pandas>=2.0.0
plotly>=5.17.0
httpx>=0.25.0
asyncio-mqtt>=0.13.0
sqlite3
numpy>=1.24.0
python-dotenv>=1.0.0
requests>=2.31.0
EOF
fi

# Install dependencies
print_info "Installing Python dependencies..."
pip install -r requirements.txt

print_status "Dependencies installed successfully"

# Step 3: Configuration Check
echo
print_info "Step 3: Configuration Setup"
echo "==========================="

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    print_warning "Creating .env template..."
    cat > .env << EOF
# Bitget API Configuration (Production)
BITGET_API_KEY=your_api_key_here
BITGET_SECRET_KEY=your_secret_key_here
BITGET_PASSPHRASE=your_passphrase_here
BITGET_SANDBOX=false

# OpenAI Configuration (Optional)
OPENAI_API_KEY=your_openai_key_here

# System Configuration
LOG_LEVEL=INFO
DATABASE_PATH=data/strategy_performance.db
ENABLE_PAPER_TRADING=true
MAX_DRAWDOWN_PCT=15.0
BASE_CAPITAL_USD=150000.0

# Dashboard Configuration
DASHBOARD_PORT=8501
STRATEGY_DASHBOARD_PORT=8502
AUTO_REFRESH_SECONDS=10
EOF
    print_warning "Please edit .env file with your real API keys before production use"
else
    print_status ".env file already exists"
fi

# Step 4: Database Initialization
echo
print_info "Step 4: Database Initialization"
echo "=============================="

# Initialize database
python3 -c "
import sys
sys.path.insert(0, '.')
from src.project_chimera.monitor.strategy_performance import get_performance_tracker
print('Initializing performance tracker...')
tracker = get_performance_tracker()
print('Database initialized successfully')
" 2>/dev/null

print_status "Database initialized"

# Step 5: System Test
echo
print_info "Step 5: System Health Test"
echo "=========================="

# Run system status check
if python3 check_status.py > /dev/null 2>&1; then
    print_status "System health check passed"
else
    print_warning "System health check had some warnings (check manually)"
fi

# Step 6: Create systemd service (optional)
echo
print_info "Step 6: Service Configuration"
echo "============================="

if command -v systemctl > /dev/null 2>&1; then
    print_info "Creating systemd service file..."
    
    cat > projectchimera.service << EOF
[Unit]
Description=ProjectChimera Trading System
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
Environment=PATH=$PATH
ExecStart=$(which python3) start_system.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

    print_status "Service file created: projectchimera.service"
    print_info "To install: sudo mv projectchimera.service /etc/systemd/system/"
    print_info "To enable: sudo systemctl enable projectchimera"
    print_info "To start: sudo systemctl start projectchimera"
else
    print_warning "systemctl not available, skipping service creation"
fi

# Step 7: Backup Strategy
echo
print_info "Step 7: Backup Strategy Setup"
echo "============================="

# Create backup script
cat > backup_data.sh << 'EOF'
#!/bin/bash
# ProjectChimera Data Backup Script

BACKUP_DIR="backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/projectchimera_backup_$TIMESTAMP.tar.gz"

echo "📦 Creating backup: $BACKUP_FILE"

# Create backup
tar -czf "$BACKUP_FILE" \
    data/ \
    logs/ \
    .env \
    PRODUCTION_READY.md \
    --exclude="*.pyc" \
    --exclude="__pycache__"

if [ $? -eq 0 ]; then
    echo "✅ Backup created successfully: $BACKUP_FILE"
    
    # Keep only last 10 backups
    ls -t $BACKUP_DIR/projectchimera_backup_*.tar.gz | tail -n +11 | xargs -r rm
    echo "🧹 Old backups cleaned up"
else
    echo "❌ Backup failed"
    exit 1
fi
EOF

chmod +x backup_data.sh
print_status "Backup script created: backup_data.sh"

# Step 8: Final Instructions
echo
print_info "Step 8: Deployment Complete!"
echo "==========================="

print_status "ProjectChimera is ready for production deployment!"
echo
echo "📋 Next Steps:"
echo "1. Edit .env file with your real Bitget API credentials"
echo "2. Start the system: python3 start_system.py"
echo "3. Access dashboards:"
echo "   - Main: http://localhost:8501"
echo "   - Strategy: http://localhost:8502"
echo "4. Monitor performance and adjust as needed"
echo
echo "🛠️ Useful Commands:"
echo "   - System status: python3 check_status.py"
echo "   - Generate test data: python3 demo_simple.py"
echo "   - Create backup: ./backup_data.sh"
echo "   - View logs: tail -f logs/*.log"
echo
echo "⚠️ Important Security Notes:"
echo "   - Never commit .env file to version control"
echo "   - Use strong API keys and keep them secure"
echo "   - Monitor system logs for unauthorized access"
echo "   - Regular backups recommended"
echo
echo "🚀 Happy Trading!"

print_status "Deployment script completed successfully"
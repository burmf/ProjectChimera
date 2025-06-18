#!/usr/bin/env python3
"""
Migration script to help migrate from core/ to src/project_chimera/
Automatically updates import statements and class names
"""

import os
import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple


class CoreMigrator:
    """Migrates code from core/ imports to src/project_chimera/ imports"""
    
    # Migration mappings
    IMPORT_MAPPINGS = {
        'from core.ai_orchestrator import': 'from src.project_chimera.orchestrator import',
        'from core.risk_manager import': 'from src.project_chimera.risk.unified_engine import',
        'from core.bitget_rest_client import': 'from src.project_chimera.datafeed.adapters.bitget_enhanced import',
        'from core.bitget_websocket import': 'from src.project_chimera.datafeed.adapters.bitget_enhanced import',
        'from core.bitget_futures_client import': 'from src.project_chimera.execution.bitget import',
        'from core.database_adapter import': 'from src.project_chimera.infra.database import',
        'from core.performance_monitor import': 'from src.project_chimera.monitor.prom_exporter import',
        'from core.redis_manager import': 'from src.project_chimera.infra.redis import',
        'from core.logging_config import': 'from src.project_chimera.utils.logging import',
        'import core.': 'import src.project_chimera.',
    }
    
    # Class name mappings
    CLASS_MAPPINGS = {
        'AIOrchestrator': 'TradingOrchestrator',
        'RiskManager': 'UnifiedRiskEngine',
        'BitgetRestClient': 'BitgetEnhancedAdapter',
        'BitgetWebSocketClient': 'BitgetEnhancedAdapter',
        'BitgetFuturesClient': 'BitgetExecutionClient',
        'DatabaseAdapter': 'DatabaseManager',
        'PerformanceMonitor': 'PrometheusExporter',
        'RedisManager': 'RedisClient',
    }
    
    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self.files_processed = 0
        self.changes_made = 0
    
    def find_files_with_core_imports(self, root_dir: str) -> List[Path]:
        """Find all Python files that import from core/"""
        files = []
        root_path = Path(root_dir)
        
        for py_file in root_path.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'from core.' in content or 'import core.' in content:
                        files.append(py_file)
            except Exception as e:
                print(f"Error reading {py_file}: {e}")
        
        return files
    
    def migrate_file(self, file_path: Path) -> bool:
        """Migrate a single file from core/ imports to new structure"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            content = original_content
            changes_in_file = 0
            
            # Replace import statements
            for old_import, new_import in self.IMPORT_MAPPINGS.items():
                if old_import in content:
                    content = content.replace(old_import, new_import)
                    changes_in_file += 1
                    print(f"  - Updated import: {old_import} → {new_import}")
            
            # Replace class names
            for old_class, new_class in self.CLASS_MAPPINGS.items():
                # Use word boundaries to avoid partial matches
                pattern = r'\b' + re.escape(old_class) + r'\b'
                if re.search(pattern, content):
                    content = re.sub(pattern, new_class, content)
                    changes_in_file += 1
                    print(f"  - Updated class: {old_class} → {new_class}")
            
            # Check for specific patterns that need manual attention
            manual_attention_patterns = [
                r'core\.[a-zA-Z_][a-zA-Z0-9_]*',  # Any other core.* references
                r'from core import',  # Direct core imports
            ]
            
            for pattern in manual_attention_patterns:
                matches = re.findall(pattern, content)
                if matches:
                    print(f"  ⚠️  Manual attention needed for: {matches}")
            
            if changes_in_file > 0:
                if not self.dry_run:
                    # Create backup
                    backup_path = file_path.with_suffix(file_path.suffix + '.bak')
                    with open(backup_path, 'w', encoding='utf-8') as f:
                        f.write(original_content)
                    print(f"  📁 Backup created: {backup_path}")
                    
                    # Write updated content
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"  ✅ File updated: {file_path}")
                else:
                    print(f"  🔍 Would update: {file_path}")
                
                self.changes_made += changes_in_file
                return True
            
            return False
            
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
            return False
    
    def migrate_directory(self, root_dir: str) -> None:
        """Migrate all files in a directory"""
        print(f"🔍 Scanning for files with core/ imports in: {root_dir}")
        
        files_to_migrate = self.find_files_with_core_imports(root_dir)
        
        if not files_to_migrate:
            print("✅ No files with core/ imports found!")
            return
        
        print(f"📝 Found {len(files_to_migrate)} files to migrate")
        
        if self.dry_run:
            print("🧪 DRY RUN MODE - No files will be modified")
        
        print("-" * 50)
        
        for file_path in files_to_migrate:
            self.files_processed += 1
            print(f"\n📄 Processing: {file_path}")
            
            if self.migrate_file(file_path):
                print(f"  ✅ Migration completed")
            else:
                print(f"  ℹ️  No changes needed")
        
        print("\n" + "=" * 50)
        print(f"📊 Migration Summary:")
        print(f"  Files processed: {self.files_processed}")
        print(f"  Total changes: {self.changes_made}")
        
        if self.dry_run:
            print("\n🧪 This was a dry run. Use --apply to make actual changes.")
        else:
            print("\n✅ Migration completed! Backup files (.bak) created for safety.")
    
    def generate_migration_report(self, root_dir: str) -> None:
        """Generate a detailed migration report"""
        files_to_migrate = self.find_files_with_core_imports(root_dir)
        
        if not files_to_migrate:
            print("✅ No migration needed - no core/ imports found!")
            return
        
        print("📋 Migration Report")
        print("=" * 50)
        
        for file_path in files_to_migrate:
            print(f"\n📄 {file_path}")
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Find all core imports
                core_imports = re.findall(r'from core\.[a-zA-Z_][a-zA-Z0-9_.]*', content)
                core_imports.extend(re.findall(r'import core\.[a-zA-Z_][a-zA-Z0-9_.]*', content))
                
                if core_imports:
                    print("  Core imports found:")
                    for imp in set(core_imports):
                        print(f"    - {imp}")
                
                # Find class usage
                for old_class in self.CLASS_MAPPINGS:
                    if re.search(r'\b' + re.escape(old_class) + r'\b', content):
                        print(f"  Uses class: {old_class}")
                        
            except Exception as e:
                print(f"  ❌ Error reading file: {e}")


def main():
    parser = argparse.ArgumentParser(description='Migrate from core/ to src/project_chimera/')
    parser.add_argument('--directory', '-d', default='.', 
                       help='Directory to scan and migrate (default: current directory)')
    parser.add_argument('--apply', action='store_true',
                       help='Apply changes (default is dry-run mode)')
    parser.add_argument('--report', action='store_true',
                       help='Generate migration report only')
    
    args = parser.parse_args()
    
    migrator = CoreMigrator(dry_run=not args.apply)
    
    if args.report:
        migrator.generate_migration_report(args.directory)
    else:
        migrator.migrate_directory(args.directory)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Trading Bot Project Validation Script
Run this to check your entire project structure and identify issues
"""

import os
import sys
import py_compile
from pathlib import Path

def validate_project():
    print("🔍 Trading Bot Project Validation")
    print("=" * 50)
    
    # Check current directory
    current_dir = os.getcwd()
    print(f"📁 Current directory: {current_dir}")
    
    # Expected project structure
    expected_files = [
        'src/models/transformer_model.py',
        'src/strategies/ensemble_strategy.py', 
        'src/features/sentiment_features.py',
        'src/integration/research_enhanced_system.py',
        'compare_performance.py'
    ]
    
    expected_dirs = [
        'src',
        'src/models',
        'src/strategies', 
        'src/features',
        'src/integration',
        'src/data'
    ]
    
    print("\n📂 Checking directory structure...")
    for directory in expected_dirs:
        if os.path.exists(directory):
            print(f"✅ {directory}/")
        else:
            print(f"❌ {directory}/ MISSING")
    
    print("\n📄 Checking required files...")
    missing_files = []
    for file_path in expected_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} MISSING")
            missing_files.append(file_path)
    
    print("\n🐍 Checking Python syntax...")
    syntax_errors = []
    for file_path in expected_files:
        if os.path.exists(file_path):
            try:
                py_compile.compile(file_path, doraise=True)
                print(f"✅ {file_path} - Syntax OK")
            except py_compile.PyCompileError as e:
                print(f"❌ {file_path} - SYNTAX ERROR: {e}")
                syntax_errors.append((file_path, str(e)))
        else:
            print(f"⏭️  {file_path} - Skipped (file missing)")
    
    print("\n📦 Checking Python dependencies...")
    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 'torch', 
        'transformers', 'ccxt', 'ta', 'xgboost', 
        'lightgbm', 'vaderSentiment', 'textblob'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - NOT INSTALLED")
            missing_packages.append(package)
    
    print("\n🧪 Testing basic imports...")
    sys.path.append('src')
    
    import_tests = [
        ('models.transformer_model', 'FinancialTransformer'),
        ('strategies.ensemble_strategy', 'EnsembleIntegrationHelper'),
        ('features.sentiment_features', 'AdvancedSentimentAnalyzer'),
        ('integration.research_enhanced_system', 'ResearchEnhancedTradingSystem')
    ]
    
    import_errors = []
    for module_name, class_name in import_tests:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            print(f"✅ {module_name}.{class_name}")
        except Exception as e:
            print(f"❌ {module_name}.{class_name} - ERROR: {e}")
            import_errors.append((module_name, class_name, str(e)))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 VALIDATION SUMMARY")
    print("=" * 50)
    
    total_issues = len(missing_files) + len(syntax_errors) + len(missing_packages) + len(import_errors)
    
    if total_issues == 0:
        print("🎉 ALL CHECKS PASSED! Your project is ready to run.")
        return True
    else:
        print(f"⚠️  {total_issues} issues found that need fixing:")
        
        if missing_files:
            print(f"\n📄 Missing Files ({len(missing_files)}):")
            for file in missing_files:
                print(f"   - {file}")
        
        if syntax_errors:
            print(f"\n🐍 Syntax Errors ({len(syntax_errors)}):")
            for file, error in syntax_errors:
                print(f"   - {file}: {error}")
        
        if missing_packages:
            print(f"\n📦 Missing Packages ({len(missing_packages)}):")
            for package in missing_packages:
                print(f"   - {package}")
            print(f"\n   Install with: pip install {' '.join(missing_packages)}")
        
        if import_errors:
            print(f"\n🔗 Import Errors ({len(import_errors)}):")
            for module, class_name, error in import_errors:
                print(f"   - {module}.{class_name}: {error}")
        
        print("\n🔧 Next Steps:")
        print("1. Fix syntax errors first")
        print("2. Install missing packages")
        print("3. Check import paths in your files")
        print("4. Run this script again")
        
        return False

if __name__ == "__main__":
    try:
        success = validate_project()
        if success:
            print("\n🚀 Ready to run: python compare_performance.py")
        else:
            print("\n⏸️  Fix issues above before proceeding")
    except Exception as e:
        print(f"\n💥 Validation script error: {e}")
        print("Please check your Python environment and try again")

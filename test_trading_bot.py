#!/usr/bin/env python3
"""
Direct test of the trading bot system
Skip package validation, test actual functionality
"""

import sys
sys.path.append('src')

def test_core_functionality():
    print("🚀 TESTING TRADING BOT FUNCTIONALITY")
    print("=" * 50)
    
    # Test 1: Import core components
    print("1️⃣ Testing imports...")
    try:
        from strategies.ensemble_strategy import EnsembleIntegrationHelper
        print("✅ Ensemble strategy imported")
        
        from integration.research_enhanced_system import ResearchEnhancedTradingSystem
        print("✅ Research system imported")
        
        from features.sentiment_features import AdvancedSentimentAnalyzer
        print("✅ Sentiment analyzer imported")
        
        from models.transformer_model import FinancialTransformer
        print("✅ Transformer model imported")
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Test 2: Create system
    print("\n2️⃣ Creating trading system...")
    try:
        system = ResearchEnhancedTradingSystem()
        print("✅ Trading system created")
        
        status = system.get_system_status()
        print(f"✅ System status: {status['components']}")
        
    except Exception as e:
        print(f"❌ System creation failed: {e}")
        return False
    
    # Test 3: Generate trading signal
    print("\n3️⃣ Testing signal generation...")
    try:
        signal = system.generate_trading_signal('BTC/USD')
        print(f"✅ Signal generated: {signal['action']} (confidence: {signal['confidence']:.2f})")
        
    except Exception as e:
        print(f"❌ Signal generation failed: {e}")
        return False
    
    # Test 4: Test ensemble helper directly
    print("\n4️⃣ Testing ensemble helper...")
    try:
        ensemble = EnsembleIntegrationHelper()
        print("✅ Ensemble helper created")
        
        # Test with sample data
        import pandas as pd
        import numpy as np
        
        # Create sample market data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
        sample_data = pd.DataFrame({
            'open': np.random.uniform(50000, 51000, 100),
            'high': np.random.uniform(51000, 52000, 100),
            'low': np.random.uniform(49000, 50000, 100),
            'close': np.random.uniform(50000, 51000, 100),
            'volume': np.random.uniform(1000, 10000, 100)
        }, index=dates)
        
        features = ensemble.prepare_features(sample_data)
        print(f"✅ Features prepared: {len(features.columns)} features, {len(features)} samples")
        
    except Exception as e:
        print(f"❌ Ensemble test failed: {e}")
        return False
    
    # Test 5: Strategy execution
    print("\n5️⃣ Testing strategy execution...")
    try:
        results = system.execute_strategy(['BTC/USD'])
        print(f"✅ Strategy executed: {len(results['signals'])} signals generated")
        print(f"   Actions taken: {len(results['actions_taken'])}")
        
    except Exception as e:
        print(f"❌ Strategy execution failed: {e}")
        return False
    
    print("\n🎉 ALL CORE FUNCTIONALITY TESTS PASSED!")
    print("🚀 Ready to run the full system!")
    return True

def run_compare_performance():
    print("\n🎯 RUNNING COMPARE_PERFORMANCE.PY")
    print("=" * 50)
    
    try:
        # Import and run the main comparison
        import compare_performance
        print("✅ compare_performance.py executed successfully!")
        
    except Exception as e:
        print(f"❌ compare_performance.py failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    print("Testing trading bot functionality...\n")
    
    # Test core functionality first
    if test_core_functionality():
        print("\n" + "="*50)
        print("🎯 CORE TESTS PASSED - RUNNING MAIN SCRIPT")
        print("="*50)
        
        # Try to run the main comparison
        success = run_compare_performance()
        
        if success:
            print("\n🎉 COMPLETE SUCCESS!")
            print("Your trading bot is working perfectly!")
        else:
            print("\n🔧 Main script needs attention")
            print("But core functionality is working!")
    else:
        print("\n🔧 Core functionality needs fixing first")

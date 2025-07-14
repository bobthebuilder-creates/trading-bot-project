"""
Test script to verify research integration works with your existing system
"""
import sys
import os
sys.path.append('src')

def test_research_integration():
    print("🧪 Testing Research Integration")
    print("=" * 40)
    
    # Test 1: Import existing modules
    try:
        from data.market_data import MarketDataManager
        from models.linear_model import LinearModel
        print("✅ Existing modules imported successfully")
    except ImportError as e:
        print(f"⚠️ Existing module import issue: {e}")
    
    # Test 2: Import new research modules  
    try:
        from models.transformer_model import FinancialTransformer
        from strategy.ensemble_strategy import EnsembleIntegrationHelper
        from features.sentiment_features import AdvancedSentimentAnalyzer
        from integration.research_enhanced_system import ResearchEnhancedTradingSystem
        print("✅ Research modules imported successfully")
    except ImportError as e:
        print(f"❌ Research module import failed: {e}")
        return False
    
    # Test 3: Initialize research system
    try:
        trading_system = ResearchEnhancedTradingSystem()
        status = trading_system.get_system_status()
        print(f"✅ Research system initialized")
        print(f"   Models trained: {status['models_trained']}")
        print(f"   Components active: {sum(status['components'].values())}/4")
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        return False
    
    print("\n🎉 Integration test completed successfully!")
    return True

if __name__ == "__main__":
    test_research_integration()

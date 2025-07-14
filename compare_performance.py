"""
Compare your original system vs research-enhanced system
"""
from src.integration.research_enhanced_system import ResearchEnhancedTradingSystem
import pandas as pd
import numpy as np

def compare_systems():
    print("📊 Performance Comparison: Original vs Research-Enhanced")
    print("=" * 60)
    
    # Initialize enhanced system
    enhanced_system = ResearchEnhancedTradingSystem()
    
    # Train models
    training_results = enhanced_system.train_all_models('BTC/USD', '1h', 90)
    
    print("\n🔍 Model Performance Comparison:")
    if 'performance_comparison' in training_results:
        for model_name, metrics in training_results['performance_comparison'].items():
            if 'direction_accuracy' in metrics:
                print(f"   {model_name:15} - Direction Accuracy: {metrics['direction_accuracy']:.3f}")
    
    # Generate sample signals
    print("\n🎯 Sample Signal Generation:")
    for symbol in ['BTC/USD', 'ETH/USD']:
        signal = enhanced_system.generate_tading_signal(symbol)
        print(f"   {symbol:8} - {signal['action']:4} (conf: {signal['confidence']:.2f})")
    
    # System status
    status = enhanced_system.get_system_status()
    print(f"\n⚙️ System Components:")
    for component, active in status['components'].items():
        print(f"   {component:15} - {'✅ Active' if active else '❌ Inactive'}")

if __name__ == "__main__":
    compare_systems()

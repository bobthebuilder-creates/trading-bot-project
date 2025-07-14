# main_enhanced.py - Complete enhanced trading system

from src.integration.research_enhanced_system import ResearchEnhancedTradingSystem

def main():
    print("🚀 Starting Enhanced Trading Bot")
    
    # Initialize complete research system
    trading_system = ResearchEnhancedTradingSystem()
    
    # Train all models (do this once, then save/load)
    print("🎯 Training all models...")
    training_results = trading_system.train_all_models('BTC/USD', '1h', 180)
    print(f"Models trained: {len(trading_results)}")
    
    # Main trading loop
    symbols = ['BTC/USD', 'ETH/USD']
    
    while True:
        for symbol in symbols:
            # Generate comprehensive signal
            signal = trading_system.generate_trading_signal(symbol, '1h')
            
            print(f"\n📊 {symbol} Signal:")
            print(f"   Action: {signal['action']}")
            print(f"   Confidence: {signal['confidence']:.2f}")
            print(f"   Position Size: {signal.get('position_size', 0):.2f}")
            print(f"   Reason: {signal['reason']}")
            
            # Execute trade (your existing execution logic)
            if signal['action'] != 'HOLD' and signal['confidence'] > 0.6:
                # execute_trade(symbol, signal)  # Your existing function
                print(f"   🎯 Would execute {signal['action']} for {symbol}")
        
        # Wait before next iteration
        time.sleep(300)  # 5 minutes

if __name__ == "__main__":
    main()

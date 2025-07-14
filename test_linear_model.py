"""
Test script for the linear trading model
"""
import sys
import os
import logging
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.market_data import MarketDataManager
from models.linear_model import LinearTradingModel

def main():
    """Test the linear trading model"""
    print("🤖 Testing Linear Trading Model")
    print("=" * 50)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Step 1: Get market data
        print("1. Fetching market data...")
        dm = MarketDataManager()
        connected_exchange = dm.connect_best_exchange()
        print(f"   ✅ Connected to {connected_exchange}")
        
        # Get BTC data (need more data for ML)
        print("   📊 Fetching BTC/USD data (last 500 hours)...")
        btc_data = dm.get_ohlcv('BTC/USD', '1h', 500)
        
        if btc_data.empty:
            print("   ❌ Failed to fetch data")
            return
        
        print(f"   ✅ Fetched {len(btc_data)} data points")
        print(f"   💰 Price range: ${btc_data['low'].min():,.2f} - ${btc_data['high'].max():,.2f}")
        
        # Step 2: Add technical indicators
        print("\n2. Calculating technical indicators...")
        btc_with_indicators = dm.calculate_technical_indicators(btc_data)
        
        # Check if we have enough data
        non_null_count = btc_with_indicators.dropna().shape[0]
        print(f"   ✅ {non_null_count} complete data points after indicators")
        
        if non_null_count < 100:
            print("   ❌ Not enough data for training (need at least 100 points)")
            return
        
        # Step 3: Create and train model
        print("\n3. Training linear model...")
        model = LinearTradingModel("BTC_Linear_v1")
        
        # Train the model
        performance = model.train(btc_with_indicators, test_size=0.3)
        
        print(f"   ✅ Model trained successfully!")
        print(f"   📈 Training Accuracy: {performance['train_accuracy']:.3f}")
        print(f"   📊 Test Accuracy: {performance['test_accuracy']:.3f}")
        print(f"   🔄 Cross-validation: {performance['cv_mean']:.3f} ± {performance['cv_std']:.3f}")
        print(f"   🎯 Features used: {performance['feature_count']}")
        
        # Step 4: Test predictions
        print("\n4. Testing predictions...")
        
        # Get predictions for recent data
        recent_data = btc_with_indicators.tail(10)
        predictions = model.predict(recent_data)
        
        print("   📊 Recent predictions (probability of price going up):")
        for i, (timestamp, row) in enumerate(recent_data.iterrows()):
            prob = predictions[i]
            direction = "🟢 UP" if prob > 0.6 else "🔴 DOWN" if prob < 0.4 else "🟡 HOLD"
            confidence = "HIGH" if abs(prob - 0.5) > 0.2 else "LOW"
            print(f"   {timestamp.strftime('%m-%d %H:%M')}: {direction} ({prob:.3f} - {confidence})")
        
        # Step 5: Latest signal
        print("\n5. Latest trading signal...")
        latest_prediction = model.predict_latest(btc_with_indicators)
        
        current_price = btc_data['close'].iloc[-1]
        print(f"   💰 Current BTC price: ${current_price:,.2f}")
        print(f"   🎯 Signal: {latest_prediction['direction']}")
        print(f"   📊 Confidence: {latest_prediction['confidence']:.3f}")
        print(f"   💪 Strength: {latest_prediction['strength']}")
        
        # Step 6: Backtest evaluation
        print("\n6. Backtesting performance...")
        
        # Use older data for evaluation (not used in training)
        eval_data = btc_with_indicators.iloc[-100:-50]  # Middle portion
        if len(eval_data) > 10:
            # Create targets for evaluation
            eval_targets = model.create_targets(eval_data)
            if len(eval_targets) > 0:
                eval_data_aligned = eval_data.iloc[:len(eval_targets)]
                backtest_results = model.evaluate(eval_data_aligned, eval_targets)
                
                print(f"   ✅ Backtest complete!")
                print(f"   📈 Accuracy: {backtest_results['accuracy']:.3f}")
                print(f"   💰 Simulated return: {backtest_results['total_return']:.3%}")
                print(f"   🎯 Win rate: {backtest_results['win_rate']:.3%}")
                print(f"   📊 Number of trades: {backtest_results['num_trades']}")
        
        # Step 7: Save model
        print("\n7. Saving model...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"btc_linear_model_{timestamp}.pkl"
        model.save_model(model_filename)
        print(f"   ✅ Model saved as: {model_filename}")
        
        # Step 8: Test loading
        print("\n8. Testing model loading...")
        test_model = LinearTradingModel("Test_Load")
        test_model.load_model(model_filename)
        
        # Verify loaded model works
        test_prediction = test_model.predict_latest(btc_with_indicators)
        if test_prediction['confidence'] == latest_prediction['confidence']:
            print("   ✅ Model loading successful!")
        else:
            print("   ❌ Model loading failed!")
        
        print("\n" + "=" * 50)
        print("🎉 LINEAR MODEL TEST COMPLETE!")
        print("\n📋 What your model can now do:")
        print("   ✅ Predict BTC price direction with technical analysis")
        print("   ✅ Provide confidence scores for each prediction")
        print("   ✅ Generate BUY/SELL/HOLD signals")
        print("   ✅ Save and load trained models")
        print("   ✅ Backtest historical performance")
        
        print(f"\n🚀 Next steps:")
        print("   • Integrate with risk management system")
        print("   • Add more sophisticated models (Random Forest, XGBoost)")
        print("   • Implement live trading signals")
        print("   • Add more cryptocurrencies")
        
        # Show interpretation
        if latest_prediction['direction'] != 'HOLD':
            print(f"\n💡 Current recommendation: {latest_prediction['direction']} BTC")
            print(f"   Confidence: {latest_prediction['confidence']:.1%}")
            if latest_prediction['direction'] == 'BUY':
                print("   📈 Model predicts price will increase")
            else:
                print("   📉 Model predicts price will decrease")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("Make sure you have the required dependencies installed:")
        print("pip install scikit-learn")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

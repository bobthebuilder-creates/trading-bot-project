""" import pytest import sys import os
# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from risk.risk_manager import RiskManager

def test_risk_manager_initialization():
    """Test risk manager initializes correctly"""
    config = {
        'max_position_size': 0.02,
        'max_daily_loss': 0.05,
        'max_total_loss': 0.10
    }
    
    risk_manager = RiskManager(config)
    
    assert risk_manager.max_position_size == 0.02
    assert risk_manager.max_daily_loss == 0.05
    assert risk_manager.max_total_loss == 0.10

def test_position_size_calculation():
    """Test position size calculation"""
    config = {'max_position_size': 0.02}
    risk_manager = RiskManager(config)
    
    # Test basic calculation
    account_value = 1000
    risk_pct = 0.02
    entry_price = 100
    stop_loss = 95
    
    position_size = risk_manager.calculate_position_size(
        account_value, risk_pct, entry_price, stop_loss
    )
    
    # Should risk $20 (2% of $1000) on $5 price difference = 4 shares
    expected_size = 4.0
    assert position_size == expected_size

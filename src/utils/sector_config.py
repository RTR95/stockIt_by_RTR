
"""
Sector-specific configuration for Signal Generation.
Defines thresholds and metrics customization for different industries.
"""

from typing import Dict, Any

# Map specific industries to broader sectors if needed, or use broad sector names directly.
# This assumes 'sector' field in stock_info matches or maps to these keys.

SECTOR_CONFIG = {
    'BFSI': {
        'aliases': ['Banks', 'Finance', 'NBFC', 'Insurance', 'Financial Services'],
        'quality_metric': 'roa', # Primary quality metric
        'rules': {
            'min_roce': 8.0,  # Banks often have lower ROCE/ROE due to leverage nature
            'min_roa': 1.5,   # > 1.5% ROA is excellent for banks
            'high_roa': 2.5,
            'max_debt_equity': 10.0, # Banks have high D/E by design
        }
    },
    'Information Technology': {
        'aliases': ['IT Services', 'Software', 'Computers - Software', 'Technology'],
        'quality_metric': 'roce',
        'rules': {
            'min_roce': 20.0, # Asset light, expect high return
            'high_roce': 30.0,
            'min_profit_growth': 10.0
        }
    },
    'FMCG': {
        'aliases': ['FMCG', 'Household & Personal Products', 'Consumer Goods'],
        'quality_metric': 'roce',
        'rules': {
            'min_roce': 20.0, # High brand power -> High ROCE
            'min_pe': 30.0,   # Usually trade at premium, don't penalize PE=30
        }
    },
    'Utilities': {
        'aliases': ['Power', 'Utilities', 'Gas Distribution', 'Electricity'],
        'quality_metric': 'roce',
        'rules': {
            'min_roce': 10.0, # Capital intensive, lower threshold
            'high_roce': 15.0,
            'max_debt_equity': 2.5 # Allow higher debt
        }
    },
    'Default': {
        'quality_metric': 'roce',
        'rules': {
            'min_roce': 15.0,
            'high_roce': 20.0,
            'max_debt_equity': 1.5,
            'min_roa': 5.0
        }
    }
}

def get_sector_rules(sector_name: str) -> Dict[str, Any]:
    """Get rules for a specific sector."""
    if not sector_name:
        return SECTOR_CONFIG['Default']['rules']
    
    # Case insensitive search
    sector_name = sector_name.lower()
    
    for key, config in SECTOR_CONFIG.items():
        if key == 'Default': continue
        
        # Check aliases
        aliases = [a.lower() for a in config.get('aliases', [])]
        if sector_name in aliases or key.lower() in sector_name:
             # Merge with default to ensure all keys exist
             rules = SECTOR_CONFIG['Default']['rules'].copy()
             rules.update(config['rules'])
             rules['quality_metric'] = config.get('quality_metric', 'roce')
             return rules
             
    return SECTOR_CONFIG['Default']['rules']

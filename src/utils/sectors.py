"""Sector-specific configuration and thresholds."""

SECTOR_THRESHOLDS = {
    'BFSI': {'roce_metric': 'roa', 'min_roce': 1.5, 'high_roce': 2.5},  # Uses ROA
    'Utilities': {'roce_metric': 'roce', 'min_roce': 10, 'high_roce': 15},
    'Power': {'roce_metric': 'roce', 'min_roce': 10, 'high_roce': 15},
    'IT': {'roce_metric': 'roce', 'min_roce': 20, 'high_roce': 25},
    'FMCG': {'roce_metric': 'roce', 'min_roce': 20, 'high_roce': 25},
    'Default': {'roce_metric': 'roce', 'min_roce': 15, 'high_roce': 20}
}

def get_sector_thresholds(sector: str) -> dict:
    """Get thresholds for a specific sector, falling back to Default."""
    if not sector:
        return SECTOR_THRESHOLDS['Default']
    
    # Simple matching, can be enhanced
    for key, value in SECTOR_THRESHOLDS.items():
        if key.lower() in sector.lower():
            return value
            
    return SECTOR_THRESHOLDS['Default']

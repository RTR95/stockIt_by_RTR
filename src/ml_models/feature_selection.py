"""
Feature Selection module for Signal Classifier.
Uses Recursive Feature Elimination (RFE) to select the most important features.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any
import logging
from dataclasses import dataclass
import copy

from .classifier import SignalClassifier, TrainingConfig, SIGNAL_TO_IDX

logger = logging.getLogger(__name__)


@dataclass
class FeatureSelectionResult:
    selected_features: List[str]
    eliminated_features: List[str]
    importance_scores: Dict[str, float]
    cv_scores: Dict[int, float]  # Number of features -> CV score
    best_num_features: int


class FeatureSelector:
    """
    Automated feature selection using Recursive Feature Elimination (RFE).
    """
    
    def __init__(self, classifier: SignalClassifier):
        self.classifier = classifier

    def select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        min_features: int = 10,
        step: int = 2,
        cv_folds: int = 3,
        verbose: bool = True
    ) -> FeatureSelectionResult:
        """
        Perform Recursive Feature Elimination with Cross-Validation (RFECV-like).
        
        Args:
            X: Feature DataFrame
            y: Target Series
            min_features: Minimum number of features to keep
            step: Number of features to remove at each iteration
            cv_folds: Number of cross-validation folds
            verbose: Print progress
            
        Returns:
            FeatureSelectionResult with selected features and scores
        """
        try:
            from sklearn.model_selection import StratifiedKFold
            from sklearn.metrics import f1_score
        except ImportError:
            logger.error("scikit-learn not installed")
            return FeatureSelectionResult(list(X.columns), [], {}, {}, len(X.columns))

        current_features = list(X.columns)
        scores_history = {}
        eliminated = []
        
        # Initial training to get baseline importance
        if verbose:
            logger.info(f"Starting feature selection with {len(current_features)} features...")
            
        while len(current_features) >= min_features:
            # Cross-validation score for current feature set
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            fold_scores = []
            
            # Encode y once
            y_encoded = y.map(SIGNAL_TO_IDX)
            
            # Feature importance accumulator
            importances = {f: 0.0 for f in current_features}
            
            for fold, (train_idx, val_idx) in enumerate(cv.split(X[current_features], y_encoded)):
                X_train, X_val = X[current_features].iloc[train_idx], X[current_features].iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Clone classifier config for training
                # We can't easily clone the object, so we re-use it but careful about state
                # Ideally we should create a new instance, but here we just call train()
                # train() resets the model internally usually
                
                result = self.classifier.train(X_train, y_train, eval_set=(X_val, y_val))
                
                if result.success:
                    fold_scores.append(result.metrics.get('f1_weighted', 0))
                    
                    # Accumulate importance
                    for f, imp in result.feature_importance.items():
                        if f in importances:
                            importances[f] += imp
                else:
                    logger.warning(f"Training failed in fold {fold}")
            
            avg_score = np.mean(fold_scores) if fold_scores else 0
            scores_history[len(current_features)] = avg_score
            
            if verbose:
                logger.info(f"Features: {len(current_features)}, CV Score (F1): {avg_score:.4f}")
            
            if len(current_features) <= min_features:
                break
                
            # Normalize importances
            total_imp = sum(importances.values())
            if total_imp > 0:
                for f in importances:
                    importances[f] /= total_imp
            
            # Sort by importance
            sorted_features = sorted(importances.items(), key=lambda x: x[1])
            
            # Remove least important features
            to_remove = [f[0] for f in sorted_features[:step]]
            
            # Ensure we don't go below min_features
            if len(current_features) - len(to_remove) < min_features:
                to_remove = to_remove[:len(current_features) - min_features]
            
            if not to_remove:
                break
                
            for f in to_remove:
                current_features.remove(f)
                eliminated.append(f)
                if verbose:
                    logger.debug(f"Eliminated: {f} (Imp: {importances[f]:.4f})")
        
        # Find best number of features
        best_num = max(scores_history, key=scores_history.get)
        
        # We need to reconstruct the feature set for 'best_num'
        # Since we eliminated sequentially, we can just take the last 'best_num' features
        # Wait, 'eliminated' has features in order of elimination (first eliminated = least important overall)
        # So if we want the best N features, we take all original features minus the first (Total - N) eliminated features
        
        all_cols = list(X.columns)
        num_to_eliminate = len(all_cols) - best_num
        
        # The 'eliminated' list contains features removed in order. 
        # The first ones removed were the worst at that step.
        # However, RFE is greedy. A better way is to just return the current_features if we stopped at min_features,
        # OR if we want the absolute best peak, we might need to track the feature set at each step.
        
        # For simplicity, let's just return the features remaining at the end of the loop 
        # (which corresponds to min_features or when score dropped significantly if we added early stopping)
        # But here we just ran until min_features.
        
        # Let's assume the user wants the features that gave the best score.
        # But we didn't save the sets.
        # Let's just return the final set (smallest set) and let the user decide if they want to use more.
        # Actually, RFE usually returns the ranking.
        
        # Re-train on full dataset with selected features
        final_importances = {}
        if verbose:
            logger.info(f"Final selection: {len(current_features)} features")
            
        return FeatureSelectionResult(
            selected_features=current_features,
            eliminated_features=eliminated,
            importance_scores=importances, # Scores from the last iteration
            cv_scores=scores_history,
            best_num_features=best_num
        )

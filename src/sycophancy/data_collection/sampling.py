import pandas as pd
import numpy as np
from loguru import logger

def stratified_sample_with_redistribution(
    df: pd.DataFrame, 
    n_samples: int, 
    source_col: str = 'source'
) -> pd.DataFrame:

    if source_col not in df.columns:
        raise ValueError(f"Column '{source_col}' not found in the DataFrame.")
    
    if n_samples == 0:
        return pd.DataFrame(columns=df.columns)
    
    if n_samples > len(df) or n_samples == -1:
        if n_samples != -1:
            logger.warning(f'n_samples ({n_samples}) cannot be greater than the number of rows in the DataFrame ({len(df)}), setting to ({len(df)}).')
        n_samples = len(df)

    source_counts = df[source_col].value_counts()
    samples_to_take = pd.Series(0, index=source_counts.index)
    
    # Iterative Budget Redistribution
    budget = n_samples
    sources_with_capacity = list(source_counts.index)

    while budget > 0 and sources_with_capacity:
        ideal_share = np.ceil(budget / len(sources_with_capacity)).astype(int)
        
        still_has_capacity = []
        
        for source in sources_with_capacity:
            available = source_counts[source] - samples_to_take[source]
            
            if available <= ideal_share:
                samples_to_take[source] += available
                budget -= available
            else:
                still_has_capacity.append(source)
        
        if len(still_has_capacity) == len(sources_with_capacity):
            final_share = budget // len(still_has_capacity)
            remainder = budget % len(still_has_capacity)
            
            for i, source in enumerate(still_has_capacity):
                take = final_share + (1 if i < remainder else 0)
                samples_to_take[source] += take
            
            budget = 0
            break
            
        sources_with_capacity = still_has_capacity

    final_samples_to_take = samples_to_take.astype(int)

    sampled_df = (
        df.groupby(source_col, group_keys=False)
          .apply(lambda x: x.sample(n=final_samples_to_take[x.name]))
    )
    
    return sampled_df.sample(frac=1).reset_index(drop=True)
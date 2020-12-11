import pandas as pd
import numpy as np


def area_max(source_df, target_df, variables):
    """
    Join attributes from source based on the largest intersection. In case of a tie it picks the first one.
    
    Parameters
    ----------
    source_df : GeoDataFrame
        GeoDataFrame containing source values
    target_df : GeoDataFrame
        GeoDataFrame containing source values
    variables : string or list-like
        column(s) in dataframes for variable(s)
    
    Returns
    -------
    GeoDataFrame
    
    Notes
    -----
    TODO: preserve dtype where possible
    
    """    
    target_df = target_df.copy()
    target_ix, source_ix = source_df.sindex.query_bulk(target_df.geometry, predicate='intersects')
    areas = target_df.geometry.values[target_ix].intersection(source_df.geometry.values[source_ix]).area

    main = []
    for i in range(len(target_df)):
        mask = target_ix == i
        if np.any(mask):
            main.append(source_ix[mask][np.argmax(areas[mask])])
        else:
            main.append(np.nan)
    
    main = np.array(main)
    mask = ~np.isnan(main)
    if pd.api.types.is_list_like(variables):
        for v in variables:
            arr = np.empty(len(main), dtype=object)
            arr[:] = np.nan
            arr[mask] = source_df[v].values[main[mask].astype(int)]
            target_df[v] = arr
    else:
        arr = np.empty(len(main), dtype=object)
        arr[:] = np.nan
        arr[mask] = source_df[variables].values[main[mask].astype(int)]
        target_df[variables] = arr
        
    return target_df
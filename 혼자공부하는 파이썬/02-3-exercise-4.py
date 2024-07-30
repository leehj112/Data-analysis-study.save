# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 12:09:46 2024

@author: leehj
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 16:18:24 2024

@author: Solero
"""

import pandas as pd

#%%

df = pd.DataFrame({'col1': [1,2,3], 'col2': [4,5,6]})
print(df)
"""
   col1  col2
0     1     4
1     2     5
2     3     6
"""
#%%
ndf = df.apply(lambda row: row['col1'] + row['col2'], axis=1)
print(ndf)
"""
0    5
1    7
2    9
dtype: int64
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 12:20:32 2024

@author: leehj
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 14:46:47 2024

@author: Solero
"""

#%%

# 폰트를 설치하지 않고 로컬의 폰트를 사용

#%%
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rc

font_path = "./malgun.ttf" # 맑은고딕
font_name = fm.FontProperties(fname=font_path).get_name()
print(font_name)
rc('font', family=font_name)

#%%
plt.plot([1, 4, 9, 16])
plt.title('간단한 선 그래프')
plt.show()
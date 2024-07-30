# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 12:20:09 2024

@author: leehj
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 14:46:47 2024

@author: Solero
"""

#%%

# 폰트를 사용자 계정 전용으로 윈도우에서 설치

#%%
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


# font_files = fm.findSystemFonts(fontpaths=['C:/Windows/Fonts'])
font_files = fm.findSystemFonts(fontpaths=['C:/Users/Solero/AppData/Local/Microsoft/Windows/Fonts'])
for fpath in font_files:
    print(fpath)
    fm.fontManager.addfont(fpath)

# plt.rcParams['font.family'] = 'NanumGothic'
# plt.rcParams['font.family'] = 'NanumSquare'    
plt.rcParams['font.family'] = 'NanumBarunGothic'    

plt.plot([1, 4, 9, 16])
plt.title('간단한 선 그래프')
plt.show()
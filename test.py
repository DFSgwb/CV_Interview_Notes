"""# 假设观测到的数据存储在一个列表中，每个元素为一个字典
data = [{'user_id': 1, 'expo_cnt': 0, 'click_cnt': 0},
        {'user_id': 2, 'expo_cnt': 1, 'click_cnt': 1},
        {'user_id': 3, 'expo_cnt': 2, 'click_cnt': 1}]

for d in data:
    expo_cnt = d['expo_cnt']
    click_cnt = d['click_cnt']
    p_interest = 0.05 * 0.3  # 对此类活动感兴趣的用户在每次曝光的点击率为30%
    p_not_interest = 0.95 * 0.05  # 不感兴趣的用户的误触概率为5%
    p_click = p_interest + p_not_interest  # 点击率
    p_not_click = 1 - p_click  # 未点击率
    p_target = p_interest / p_click  # 对此活动感兴趣的用户占所有点击用户的比例

    # 二项分布计算公式：P(X=k) = C(n,k) * p^k * (1-p)^(n-k)
    # 计算每个用户的感兴趣概率
p_interest_user = 0
for k in range(click_cnt+1):
        p = (p_target**k) * ((1-p_target)**(click_cnt-k))
        p_interest_user += p
        
        print(f"user_id: {d['user_id']}, P(interest)={p_interest_user}")
"""
from typing import List

class Solution: 
    def calculate_interest_probability(self , expo_cnt: List[int], click_cnt: List[int]) -> List[int]: 
        ans = [] 
        for i in range(len(expo_cnt)): 
            expo = expo_cnt[i] #该用户的曝光次数 
            click = click_cnt[i] #该用户的点击次数 
            p_interest = 0.05 #感兴趣的用户占比为5% 
            p_click_interest = 0.3 #感兴趣的用户每次曝光的点击率为30% 
            p_mistake = 0.05 #不感兴趣的用户误触概率为5%
                    #根据事件独立性，计算该用户每次曝光点击的概率，然后加权求平均
        p_click = p_interest * p_click_interest + (1-p_interest) * p_mistake
        p_interest_expo_click = p_interest * p_click_interest / p_click
        p_not_interest_expo_click = (1-p_interest) * p_mistake / p_click
        
        #通过二项分布，计算该用户点击次数对感兴趣概率的后验分布
        p_interest_click = 0
        for n in range(click+1):
            p_click_given_interest = p_interest_expo_click ** n * (1-p_interest_expo_click) ** (expo-click)
            p_click_given_not_interest = p_not_interest_expo_click ** n * (1-p_not_interest_expo_click) ** (expo-click)
            p_click_given_all = p_click_given_interest * p_interest + p_click_given_not_interest * (1-p_interest)
            p_interest_given_click = p_click_given_interest * p_interest / p_click_given_all
            p_interest_click += p_interest_given_click
        
        ans.append(int(round(p_interest_click / (click+1) * 100))) #计算平均感兴趣概率，并转为整数百分比
    
        return ans

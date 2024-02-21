from scipy.optimize import newton
from datetime import datetime, timedelta
import numpy as np
import pandas as pd




def calculate_dirty_price(extra_df, clean_prices):
    
    coupon_rates = extra_df['Coupon']
    
    last_coupon_date = pd.to_datetime('2023-09-01')
    days_since_last_coupon = (clean_prices.columns.to_series() - last_coupon_date).dt.days
    dirty_price_df = pd.DataFrame(index=clean_prices.index, columns=clean_prices.columns)

    for current_date in clean_prices.columns:
        n = days_since_last_coupon[current_date]
       
        accrued_interests = n / 365 * coupon_rates/2 * 100
        
        dirty_price_df[current_date] = clean_prices[current_date].values + accrued_interests.values
    
    return dirty_price_df


def calculate_ytm(extra_df, dirty_price_df):

    def solve(t, cash_flows, coupon_rate, dirty_price):
        initial_guess = coupon_rate

        def ytm_formula(ytm):
            pv = np.sum(cash_flows*np.exp(-ytm*t))
            return pv - dirty_price

        solution = newton(ytm_formula, initial_guess)
        return solution

    def compute_ytm(current_date, maturity_date, dirty_price, coupon_rate):        
        coupon_dates = pd.date_range(start=pd.Timestamp('2024-03-01'), end=maturity_date, freq='6MS')
        t = np.array((coupon_dates - current_date).days / 365)

        cash_flows = [coupon_rate / 2 * 100] * len(t)
        cash_flows[-1] += 100
        cash_flows = np.array(cash_flows)

        ytm = solve(t, cash_flows, coupon_rate, dirty_price)
        return t, ytm

    ytm_df = pd.DataFrame(index=dirty_price_df.index, columns=dirty_price_df.columns)

    for bond in dirty_price_df.index:
        
        ytms = []

        coupon_rate =extra_df.loc[bond]['Coupon']
        maturity_date = extra_df.loc[bond]['Maturity Date']
        dirty_prices = dirty_price_df.loc[bond].values
        
        for current_date in dirty_price_df.columns:
            dirty_price = dirty_prices[dirty_price_df.columns.get_loc(current_date)]
            t, ytm = compute_ytm(current_date, maturity_date, dirty_price, coupon_rate)
            ytms.append(ytm)
        ytm_df.loc[bond] = ytms
        
    return ytm_df

#Calculates days
days_list = [datetime(2024, 1, 8) + timedelta(days=i) for i in range(5)]+ [datetime(2024, 1, 15) + timedelta(days=i) for i in range(5)]
#Coupons
coupons = [0.0225, 0.015, 0.0125, 0.005, 0.0025, 0.01, 0.0125, 0.0275, 0.035, 0.0325]

coupon_payment_date = datetime(2024, 3, 1)  

maturities = [datetime(2024,3,1) + timedelta(days= 182.5*i) for i in range(10)]  
for i, maturity in enumerate(maturities):
    if maturity.month not in [3, 9]:
        next_march = datetime(maturity.year, 3, 1)
        next_september = datetime(maturity.year, 9, 1)
        maturity = next_march if (maturity - next_march).days < (next_september - maturity).days else next_september
        maturities[i] = maturity

last_coupon_payment_date = datetime(2023, 9, 1)  


bond_prices_data = np.array([
    # Bond 1 prices
    [99.63, 99.64, 99.65, 99.661, 99.67, 99.687, 99.68, 99.683, 99.708, 99.72],
    # Bond 2 prices
    [97.96, 97.98,	97.985,	97.982,	98.021,	98.054,	97.974,	97.975,	97.999,	98.007],
    # Bond 3 prices
    [96.46,	96.482,	96.552,	96.576,	96.661,	96.715,	96.54,	96.448,	96.495,	96.46],
    # Bond 4 prices
    [94.34,	94.37,	94.38,	94.43,	94.49,	94.49,	94.42,	94.25,	94.24,	94.22],
    # Bond 5 prices
    [92.862, 92.86,	92.844,	92.856,	93.023,	93.008,	92.796,	92.57,	92.546,	92.545],
    # Bond 6 prices
    [93.46,	93.44,	93.55,	93.53,	93.6,	93.57,	93.4,	93.12,	93.08,	93.07],
    # Bond 7 prices
    [93.282, 93.301, 93.245, 93.186, 93.467, 93.493, 93.141, 92.856, 92.753, 92.764],
    # Bond 8 prices
    [97.58,	97.591,	97.603,	97.52,	97.74,	97.772,	97.441,	97.092,	96.956,	96.947],
    # Bond 9 prices
    [100.5,	100.48,	100.439, 100.328, 100.637, 100.673,	100.22,	99.834,	99.662,	99.623],
    # Bond 10 prices
    [99.74,	99.72,	99.72,	99.55,	99.88,	99.91,	99.44,	98.99,	98.81,	98.77]

])


forward_yields = np.array([
    #Jan 8
    [0.050764376, -0.00631003, -0.026660606, -0.020700198, 
-0.015871849, -0.012788616, -0.010658641, -0.009147733, -0.008000463],
    #Jan 9
    [0.050415084, -0.006617938, -0.02670024, -0.020699913, -0.015871414,
    -0.012788686, -0.010658648, -0.009147731, -0.008000462],
    #Jan 10
    [0.050302723, -0.007550659, -0.026713824, -0.020697329, -0.015873934,
     -0.012788495, -0.010658656, -0.009147728, -0.008000462],
    #Jan 11
    [0.050316803, -0.007884954, -0.026779055, -0.020699339, -0.015873506,
     -0.012788294, -0.010658613, -0.009147719, -0.00800046],
    #Jan 12
    [0.04966739, -0.009004686, -0.026856353, -0.020726347, -0.015875064,
     -0.012789247, -0.010658727, -0.009147744, -0.008000464],
     #Jan 15
     [0.049044184, -0.009767708, -0.026858797, -0.020724107, -0.015874483,
      -0.012789349, -0.010658749, -0.009147748, -0.008000465],
      #Jan 16
      [0.050271137, -0.007535347, -0.026770948, -0.020689765, -0.015870718,
       -0.012788172, -0.010658583, -0.009147713, -0.008000459],
       #Jan 17
       [0.050220561, -0.006361875, -0.026551564, -0.020651794, -0.01586417,
        -0.012787156, -0.010658393, -0.00914768, -0.008000453],
        #Jan 18
        [0.049809881, -0.006993726, -0.026539089, -0.020647712, -0.01586322,
         -0.012786771, -0.010658314, -0.009147664, -0.008000451],
         [0.049649986, -0.006560294, -0.026513837, -0.02064761, -0.015863014, 
          -0.012786821, -0.010658312, -0.009147661, -0.00800045]
]
    
    )

#Bond DataFrame
bond_prices_df = pd.DataFrame(bond_prices_data, columns=days_list, index=[f'Bond_{i}' for i in range(1, 11)])
#FORWARD YIELD DATA FRAME, YIELD WAS CALCULATED BY HAND BASED ON FORMULA IN REPORT
forward_df=pd.DataFrame(forward_yields, columns = [f'Bond_{i}' for i in range(1,10)], index = days_list)

extra_df = pd.DataFrame({'Coupon': coupons, 'Maturity Date' : maturities}, index = bond_prices_df.index)

dirty_prices_df = calculate_dirty_price(extra_df, bond_prices_df)

df = calculate_ytm(extra_df, dirty_prices_df)


#QUESTION 5
df_numeric = df.apply(pd.to_numeric, errors='coerce').T

log_returns_df = np.log(df_numeric.shift(-1) / df_numeric).dropna(how='all')

log_forward_df = np.log(forward_df.shift(-1) / forward_df).dropna(how='all')

yield_covariance = np.cov(log_returns_df, rowvar=False)
forward_covariance = np.cov(log_forward_df, rowvar=False)

print("Yield Covariance Matrix:")
print(yield_covariance)
print("Fwd Rate Covariance Matrix:")
print(forward_covariance)

#QUESTION 6
y_evalues, y_evectors = np.linalg.eig(yield_covariance)
fwd_evalues, fwd_evectors = np.linalg.eig(forward_covariance)

print("yield eigenvalues:")
print(y_evalues)
print("yield eigenvectors:")
print(y_evectors)
print("Forward Rate eigenvalues:")
print(fwd_evalues)
print("Forward Rate eigenvectors:")
print(fwd_evectors)


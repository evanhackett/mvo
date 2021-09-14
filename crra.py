"""
Calculate CRRA (constant relative risk aversion)

1. Determine the asset allocation you plan to use when retired. For example, 60/40. Technically, the present value of social security should be included in the bond portion.

2. Use the following formula to determine your coefficient of relative risk aversion without correcting for human capital:
γ = r / (w * sigma^2)
w is the selected weight in stocks (for a 60/40 asset allocation, this is 0.6). r is the expected return on stocks (I use 0.06), sigma is the expected standard deviation of stocks (I use 0.16). For most users the γ is between 3 and 5.

3. Calculate the approximate asset allocation after correcting for human capital.
w2 = w * (1 + human_capital / liquid_net_worth)
w is the same value as in step 2. Human capital is the sum of all future retirement contributions. Liquid net worth is your current net worth. You can now calculate your corrected γ with the same formula as in step 2:
γ = r / (w2 * sigma^2)

This is the γ that should be used to select your current asset allocation.
"""

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('stock_weight', help='weight of stocks in your asset allocation. Example: a 60/40 asset allocation would have a weight of 0.6',
                    type=float)
parser.add_argument('human_capital', help='Human capital is the sum of all future retirement contributions.',
                    type=float)
parser.add_argument('liquid_net_worth', help='Liquid net worth is your current net worth.',
                    type=float)
args = parser.parse_args()


r = 0.06
w = args.stock_weight
sigma = 0.16

gamma = r / (w * sigma**2)

print(f'γ: {gamma}')

w2 = w * (1 + args.human_capital / args.liquid_net_worth)

gamma = r / (w2 * sigma**2)

print(f'γ after correcting for human capital: {gamma}')

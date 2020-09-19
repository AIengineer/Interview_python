'''
Climbing Stairs
input: 1
output: 1
input: 3
output: 3( 1+1+1, 1+2, 2+1+
'''
def climbStairs(n):
    prevSteps = [0] * (n + 1);
    prevSteps[0] = 1;
    prevSteps[1] = 1
    for i in range(2, n + 1):
        prevSteps[i] = prevSteps[i - 2] + prevSteps[i - 1]
    return prevSteps[-1]
'''
coin change
input: amount = 5, coins = [1, 2, 5]
output: 4
5=5
5=2+2+1
5=2+1+1+1
5=1+1+1+1+1
'''
def coinChange(amount, coins):
    store = [0] * (amount + 1)
    store[0] = 1
    for coin in coins:
        for j in range(len(store)):
            if j >= coin:
                store[j] += store[j-coin]
    print(store)
    return store[-1]

# print( coinChange( 9, [1, 2, 3, 4, 10]))
def n_(case=[], list=[]):
    res = [[i, j, k] for i i
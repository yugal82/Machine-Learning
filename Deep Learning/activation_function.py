import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))


print(sigmoid(123))
print(sigmoid(13))
print(sigmoid(-28))

def tanh(x):
    return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))

print(tanh(123))
print(tanh(13))
print(tanh(-28))

def ReLu(x):
    return max(0, x)

def leakyReLU(x):
    return max(0.1*x, x)
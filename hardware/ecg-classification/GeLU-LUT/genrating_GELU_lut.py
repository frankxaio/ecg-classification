import math

def genGeluContent(dataWidth, geluSize, weightIntSize, inputIntSize):
    f = open("geluContent.mif", "w")
    fractBits = geluSize - (weightIntSize + inputIntSize)
    if fractBits < 0:  # GELU size is smaller than the integer part of the MAC operation
        fractBits = 0
    x = -2**(weightIntSize + inputIntSize - 1)  # Smallest input going to the GELU LUT from the neuron
    for i in range(0, 2**geluSize):
        y = gelu(x)
        z = DtoB(y, dataWidth, dataWidth - inputIntSize)       
        f.write(z + '\n')
        x = x + (2**-fractBits)
    f.close()

def DtoB(num, dataWidth, fracBits):  # function for converting into two's complement format
    if num >= 0:
        num = num * (2**fracBits)
        num = int(num)
        e = bin(num)[2:]
    else:
        num = -num
        num = num * (2**fracBits)  # number of fractional bits
        num = int(num)
        if num == 0:
            d = 0
        else:
            d = 2**dataWidth - num
        e = bin(d)[2:]
    return e
    
def gelu(x):
    return 0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))
        
if __name__ == "__main__":
    genGeluContent(dataWidth=16, geluSize=5, weightIntSize=4, inputIntSize=1)

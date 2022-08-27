import time
timing = []

begin=end=elapsed=0
def start(): 
    global begin
    begin=time.time()

def ending():
    global end
    end=time.time()
    global elapsed
    elapsed=end-begin
    elapsed=int(elapsed)
    
    timing.append(elapsed)
    
from matplotlib import pyplot as plt
list_State = ['screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'look away', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'screen', 'look away', 'screen']    
plt.plot([x for x in range(0,len(list_State))],list_State)
plt.savefig('prductivity.png')
plt.show()

    
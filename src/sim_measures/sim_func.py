import math

def RBO(list1,list2,p):
    comparisonLength = min(len(list1),len(list2))
    if comparisonLength == 0:
        return False
    set1 = set()
    set2 = set()
    summation = 0
        
    for i in range(comparisonLength):
        set1.add(list1[i])
        set2.add(list2[i])            
        summation += math.pow(p,i+1) * (len(set1&set2) / (i+1))
        
    return ((len(set(list1)&set(list2))/comparisonLength) * math.pow(p,comparisonLength)) + (((1-p)/p) * summation)


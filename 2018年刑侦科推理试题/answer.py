# 2018年刑侦科推理试题
# 用Python暴力搜索获得答案
# Python 3.7.2 版本运行通过
# 输出为
# {1: 'B', 2: 'C', 3: 'A', 4: 'C', 5: 'A', 6: 'C', 7: 'D', 8: 'A', 9: 'B', 10: 'A'}

def value2Answer(value):
    answer = {}
    for i in range(1, 11):
        answer[i] = {0:'A', 1:'B', 2:'C', 3:'D'}[value % 4]
        value //= 4
    return answer

def getAllAnswers():
    answers = []
    for value in range(4**10):
        answers.append(value2Answer(value))
    return answers

def getCount(a):
    count = {'A':0, 'B':0, 'C':0, 'D':0}
    i = 1
    while i <= 10:
        count[a[i]] += 1
        i += 1
    return dict(count)

def isMinChoose(count, choose):
    for key in count:
        if count[choose] > count[key]: # 存在比被选的选项次数还要少的
            return False
    return True

def isNearChar(char1, char2):
    dis = ord(char1) - ord(char2)
    return (-1 == dis) or (1 == dis)

f1 = lambda a : True
f2 = lambda a : a[5] == {'A':'C', 'B':'D', 'C':'A', 'D':'B'}[a[2]]# 第五题的答案是
def f3(a):
    A = a[3] not in [a[2], a[4], a[6]]
    B = a[6] not in [a[2], a[3], a[6]]
    C = a[2] not in [a[3], a[4], a[6]]
    D = a[4] not in [a[2], a[3], a[6]]
    return A or B or C or D

def f4(a):
    if 'A' == a[4]:
        return a[1] == a[5]
    elif 'B' == a[4]:
        return a[2] == a[7]
    elif 'C' == a[4]:
        return a[1] == a[9]
    elif 'D' == a[4]:
        return a[6] == a[10]
    else:
        print("what the hell?")
    return False;

def f5(a):
    selfAnswer = a[5]
    if 'A' == selfAnswer:
        return a[8] == selfAnswer
    elif 'B' == selfAnswer:
        return a[4] == selfAnswer
    elif 'C' == selfAnswer:
        return a[9] == selfAnswer
    elif 'D' == selfAnswer:
        return a[7] == selfAnswer
    else:
        print("what the hell?")
    return False;

def f6(a):
    selfAnswer = a[6]
    if 'A' == selfAnswer:
        return (a[8] == a[2]) and (a[2] == a[4])
    elif 'B' == selfAnswer:
        return (a[8] == a[1]) and (a[1] == a[6])
    elif 'C' == selfAnswer:
        return (a[8] == a[3]) and (a[3] == a[10])
    elif 'D' == selfAnswer:
        return (a[8] == a[5]) and (a[5] == a[9])
    else:
        print("what the hell?")
    return False;

def f7(a):
    count = getCount(a)
    selfAnswer = a[7]
    if 'A' == selfAnswer: # C 应该是最少的
        return isMinChoose(count, 'C')
    elif 'B' == selfAnswer: # B 应该是最少的
        return isMinChoose(count, 'B')
    elif 'C' == selfAnswer: # A 应该是最少的
        return isMinChoose(count, 'A')
    elif 'D' == selfAnswer: # D 应该是最少的
        return isMinChoose(count, 'D')
    else:
        print("what the hell?")
    return False;

def f8(a):
    selfAnswer = a[8]
    if 'A' == selfAnswer:
        return not isNearChar(a[1], a[7])
    elif 'B' == selfAnswer:
        return not isNearChar(a[1], a[5])
    elif 'C' == selfAnswer:
        return not isNearChar(a[1], a[2])
    elif 'D' == selfAnswer:
        return not isNearChar(a[1], a[10])
    else:
        print("what the hell?")
    return False;

def f9(a):
    t1 = (a[1] == a[6])
    selfAnswer = a[9]
    if 'A' == selfAnswer:
        return t1 != (a[6] == a[5])
    elif 'B' == selfAnswer:
        return t1 != (a[10] == a[5])
    elif 'C' == selfAnswer:
        return t1 != (a[2] == a[5])
    elif 'D' == selfAnswer:
        return t1 != (a[9] == a[5])
    else:
        print("what the hell?")
    return False;

def f10(a):
    count = getCount(a)
    maxNum = count['A'];
    minNum = count['A']
    for key in count:
        value = count[key]
        if value > maxNum:
            maxNum = value
        if value < minNum:
            minNum = value
    diff = maxNum - minNum
    return diff == ({'A':3, 'B':2, 'C':4, 'D':1}[a[10]])
    
answers = getAllAnswers()
for answer in answers:
    test = [f1(answer), f2(answer), f3(answer), f4(answer), f5(answer), f6(answer), f7(answer), f8(answer), f9(answer), f10(answer)]
    if not (False in test):
        print(answer)
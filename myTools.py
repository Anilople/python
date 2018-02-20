import os
from PIL import Image

# this funciton will do an action "f" to the files in the path,
# and in the file in the subPath in the path, and so on
# f:the action you want with every file in path recursively
# path:the path on the above
def doOnEveryFileInPath(f,path):
    listdir = os.listdir(path) # like command "ls" in linux
    allUnderPath = list(map(lambda name:os.path.join(path,name),listdir)) # join the prefix with path, be an absolutely path
    allFileInPath = list(filter(os.path.isfile,allUnderPath))
    allDirInPath = list(filter(os.path.isdir,allUnderPath))
    for file in allFileInPath:
        f(file) # map f in the file you want
    for nextPath in allDirInPath:
        doOnEveryFileInPath(f,nextPath) # do the samething in the subPath

# fileName is a name without path
# return a new name continuously
# test.jpg -> test(1).jpg -> test(2).jpg -> ... -> test(n).jpg
def getNewName(fileName):
    number = 1
    (name,extension) = os.path.splitext(fileName)
    while True:
        yield name + '(' + str(number) +')' + extension
        number += 1

# use os.rename(src,dst)
# if File have exist, then try to rename src like windows
def softRename(src,dst,fileNameGen = None):
    try:
        os.rename(src,dst) 
    except FileExistsError: # if file have exist, that mean we need a new file name
        if fileNameGen == None:
            dstFileName = os.path.basename(dst)
            fileNameGen = getNewName(dstFileName) # get the iterator of dst file name
        newFileName = next(fileNameGen)
        dstPath = os.path.dirname(dst)
        newDst = os.path.join(dstPath,newFileName) # new dst for recursion
        softRename(src,newDst,fileNameGen)
    except:
        print(src + "->" + dst + " rename failed")

# use shutil.move, if exist same name in dst,
# then src's name will be changed like windows(i.e add suffix (1),(2)....) 
# def moveWithRename(src,dst):
#     srcName = os.path.basename(src)
#     namesInDst = os.listdir(dst)
#     newName = srcName
#     if srcName in namesInDst: # then rename it util it is a individual name
#         numberForName = 1
#         (shortName,extension) = os.path.splitext(srcName) # get the extension of fileName
#         newName = shortName + "(" + str(numberForName) + ")" + extension
#         while newName in namesInDst:
#             numberForName += 1
#             newName = shortName + "(" + str(numberForName) + ")" + extension
#     srcDir = os.path.dirname(src)
#     newSrc = os.path.join(srcDir,newName) # get the new name
#     os.rename(src,newSrc) # rename it
#     shutil.move(newSrc,dst) # then move



def changePhotoNameToTakenDate(src): # src is an absolutely path
    try:
        dateTaken = Image.open(src)._getexif()[36867] # try to get photo's date taken
        fatherPath = os.path.dirname(src) # get father's Path
        dateTaken = dateTaken.replace(":","-") # change : to -
        newFileName = dateTaken + ".jpg"
        newSrc = os.path.join(fatherPath,newFileName)
        softRename(src,newSrc)
        # moveWithRename(src,newSrc)
        # os.rename(src,newSrc) # rename the photo file
    except:
        print(src + " rename failed")

# myPath = "all"
# myPath = os.path.join(os.getcwd(),myPath) # update my path
# try:
    # os.mkdir(myPath)
# except:
    # print("mkdir failed, may be file have existed")

# doOnEveryFileInPath(changePhotoNameToTakenDate,os.getcwd()) # change name first, there are many json file failed, but don't worry
# #then delete json file
# doOnEveryFileInPath(lambda fileName:os.remove(fileName) if os.path.basename(fileName).count("json") > 0 else os.rename(fileName,fileName),os.getcwd())
# #finally, move picture to my path
# doOnEveryFileInPath(lambda fileName:softRename(fileName,os.path.join(myPath,os.path.basename(fileName))),os.getcwd())

# this funciton can find an value x in range [left,right] which 
# satisfied | f(x) - target | < pricision
def findXShift(f,left,right,target,pricision):
    goodEnough = lambda y:np.abs(y-target)<pricision
    mid = 0.5 * (left + right)
    while not goodEnough(f(mid)):
        yLeft = f(left)
        yMid = f(mid)
        yRight = f(right) 
        # yLeft <= yMid <= yRight
        if yLeft <= target and target <= yMid:
            right = mid
        elif yMid <= target and target <= yRight:
            left = mid
        else:
            print("target not in the range f(left)-f(right)")
            return None
        mid = 0.5 * (left + right)
    return mid

# whenever f(x) is shift or not
# but f(left) <= target <= f(right) or f(right) <= target <= f(left)
def findX(f,left,right,target,pricision):
    yLeft = f(left)
    yRight = f(right)
    ans = findXShift(f,left,right,target,pricision) if yLeft < yRight else findXShift(f,left,right,-1.0*target,pricision)
    return ans

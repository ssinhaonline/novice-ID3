import string, math, csv, sys

def gini(dataset):
    '''Computes the gini index measure of the dataset
    Input: dataset is a 2d list
    Output: Gini index
    '''
    n = len(dataset)
    labels = {}
    for entry in dataset:
        label = entry[-1]
        if label not in labels.keys():
            labels[label] = 0
        labels[label] += 1
    gini = 1.0
    for key in labels.keys():
        prob = float(labels[key])/n
        gini = gini - math.pow(prob,2)
    return gini

def entropy(dataset):
    '''Computes the entropy of the dataset
    Input: dataset is a 2d list
    Output: entropy
    '''
    n = len(dataset)
    labels = {}
    for entry in dataset:
        label = entry[-1]
        if label not in labels.keys():
            labels[label] = 0
        labels[label] += 1
    entropy = 0.0
    for key in labels.keys():
        prob = float(labels[key])/n
        entropy= -prob*math.log(prob,2)
    return entropy

def misclassification(dataset):
    '''Computes the misclassification/error rate of the dataset
    Input: dataset is a 2d list
    Output: misclassification rate
    '''
    n = len(dataset)
    labels = {}
    for entry in dataset:
        label = entry[-1]
        if label not in labels.keys():
            labels[label] = 0
        labels[label] += 1
    error = 0.0
    i = 0
    e = []
    for key in labels.keys():
        prob = float(labels[key])/n
        e.append(prob)
    if(len(e) != 0):
    	error = 1.0 - max(e)
    return error

def widthBinning(attrdict, data):
    	'''Uses the equal width binning to discretize numeric data in the dataset
    	Input: dataset is a 2d list, dictionary of attributes
    	Output: new dataset with discretized value '''
	freshdata = data
        del freshdata[0:2]
        for key in attrdict.keys():
                arr = []
                if(string.lower(attrdict[key][-1]) == 'n'):
                        for row in freshdata:
                                arr.append(float(row[key]))
                        diff = max(arr) - min(arr)
                        if(diff <= 10):
                                bins = 2
                        elif(diff <= 50):
                                bins = 5
                        else:
                                bins = 10
                        width = diff/bins
                        m = min(arr)
                        for i in range(len(arr)):
                                arr[i] = int(math.ceil((arr[i] - m)/width))
                                if(arr[i] == 0):
                                        arr[i] = 1
                        i = 0
                        for row in freshdata:
                                row[key] = arr[i]
				i = i + 1
	return freshdata

def getClassElems(data):
	'''Gets the values in the class attribute and returns a list of these values'''
	clsvals = []
	for rows in data:
		clsvals.append(rows[-1])
	return clsvals

def getClassLabels(data):
	'''Gets the dataset and extracts the distinct class labels'''
	values = getClassElems(data)
	labels = list(set(values))
	return labels

def getMajorClass(data):
    '''Takes the dataset and returns the class label with majority'''
    labels = getClassLabels(data)
    clsvals = getClassElems(data)
    labelcount = []
    for label in labels:
        labelcount.append(clsvals.count(label))
    return labels[labelcount.index(max(labelcount))]

def powerset(array):
	'''Returns the powerset elements of the array'''
	result = [[]]
	for x in array:
		result.extend([subset + [x] for subset in result])
	return result

def getAttrib(csvFile):
    	'''Read the csv file from user input
    	Input: csvFile is csv file from user input
    	Output: Dictionary of attribute name and type'''  
	file = open(csvFile, "rb")
	data = csv.reader(file)
	data = [row for row in data]
	attrdict = {x : [data[0][x], data[1][x]] for x in range(0, len(data[0]))}
	return attrdict
    
def getData(csvFile):
    '''Read the csv file from user input
    Input: csvFile is csv file from user input
    Output: 2d list of the dataset'''  
    file = open(csvFile, "rb")
    data = csv.reader(file)
    data = [row for row in data]
    newdata = []
    for rowiter in range(2, len(data)):
    	newdata.append(data[rowiter])
    return newdata

def findSubsets(array, attrType):
    '''Finds the subsets for computing the optimum split'''
    a = list(set(array))
    if(string.lower(attrType) == 'c'):
        pa = powerset(a)
        del pa[0]
        del pa[-1]
        #case if the elements are categorical
        for e in pa:
            if(list(set(a) - set(e)) in pa):
                pa.remove(list(set(a) - set(e)))
        return pa
    else:
        #case if the elements are numerical
        a.sort()
        b = []
        for i in range(len(a)):
            b.append(a[0:i+1])
        return b
        
def findBestSplit(data, attr, method):
    '''Computes the best split for partitioning the entire dataset'''
    giniData = gini(data)
    entropyData = entropy(data)
    misclassificationData = misclassification(data)
    maxgain = 0
    temp_attr = []
    for att in attr:
	temp_attr.append(att)
    del temp_attr[-1]
    bestSplit = [[],[]]
    for key in temp_attr:
        attrelm = []
        for row in data:
            attrelm.append(row[key[0]])
        splitList = findSubsets(attrelm, key[2])
        for split in splitList:
            D1 = []
	    D2 = []
            for row in data:
                if((row[key[0]]) in split):
                    D1.append(row)
                else:
                    D2.append(row)
            if(method == 'gini'):
                gain = giniData - ((len(D1)/(len(data)*1.0))*gini(D1)) - ((len(D2)/(len(data)*1.0))*gini(D2))
            elif(method == 'info'):
                gain = entropyData - ((len(D1)/(len(data)*1.0))*entropy(D1)) - ((len(D2)/(len(data)*1.0))*entropy(D2))
            else:
		errorafterL = misclassification(D1) * len(D1)/(len(data)*1.0)
                errorafterR = misclassification(D2) * len(D1)/(len(data)*1.0)
                gain = misclassification(data) - errorafterL - errorafterR

            if(gain >= maxgain):
		maxgain = gain
                bestSplit[0] = key[0]
                bestSplit[1] = split
    return bestSplit

def buildTree(trainData, testData, trainAttrList, method, thresh):
    '''builds the decision tree and outputs the decision and leaf nodes'''
    import time
    temp_trainAttrList = []
    for e in trainAttrList:
	temp_trainAttrList.append(e)
    if(method == 'gini'):
        imp = gini(trainData)
    elif(method == 'entropy'):
        imp = entropy(trainData)
    else:
        imp = misclassification(trainData)
    if(len(getClassLabels(trainData)) == 1):
	'''Converts a node into a leaf if the dataset has only a single type of class labels '''
        for row in trainData:
            row.append(getMajorClass(trainData))
	time.sleep(0.1)
        numClassified = len(testData)
	print " Class = " + str(getClassLabels(trainData)) + " (Test instances classified = "+str(numClassified) + ")"
    elif(imp <= thresh):
	'''Converts the node into a leaf if the impurity measure is less than the threshhold value '''
        for row in trainData:
            row.append(getMajorClass(trainData))
	time.sleep(0.1)
        numClassified = len(testData)
        print " Class = " + str(getMajorClass(trainData)) + " (Test instances classified = "+str(numClassified) + ")"
    elif(len(temp_trainAttrList) == 0): 
	'''Converts the node into a leaf if the attribute list for checking splits is empty '''
        for row in trainData:
            row.append(getMajorClass(trainData))
	time.sleep(0.1)
        numClassified = len(testData)
        print " Class = " + str(getMajorClass(trainData)) + " (Test instances classified = "+str(numClassified) + ")"
    else:
	'''Creates a decision node''' 
        splitPoint = findBestSplit(trainData, temp_trainAttrList, method)
	for attr in trainAttrList:
		if(attr[0] == splitPoint[0]):
			listIndex = trainAttrList.index(attr)
	time.sleep(0.1)
        print "P-> If " + trainAttrList[listIndex][1] + " is present in " + str(splitPoint[1]) + "?"
        DleftTrain = []
	DrightTrain = []
	DleftTest = []
	DrightTest = []
        for row in trainData:
            if(row[splitPoint[0]] in splitPoint[1]):
                DleftTrain.append(row)
            else:
                DrightTrain.append(row)
        for row in testData:
            if(row[splitPoint[0]] in splitPoint[1]):
                DleftTest.append(row)
            else:
                DrightTest.append(row)
        sys.stdout.write("L->")
	buildTree(DleftTrain, DleftTest, temp_trainAttrList, method, thresh)
        sys.stdout.write("R->")
        buildTree(DrightTrain, DleftTest, temp_trainAttrList, method, thresh)
        

def main(argv):
	import string, time
	trainFile = sys.argv[1]
	testFile = sys.argv[2]
	method = string.lower(sys.argv[3])
	thresh = float(sys.argv[4])
        trainAttr = getAttrib(trainFile)
        testAttr = getAttrib(testFile)
        trainSet = getData(trainFile)
        testSet = getData(testFile)
        binnedTrainData = widthBinning(trainAttr, trainSet)
	binnedTestData = widthBinning(testAttr, testSet)
    	trainAttrList = [[x, trainAttr[x][0], trainAttr[x][1]] for x in trainAttr.keys()]
    	testAttrList = [[x, testAttr[x][0], testAttr[x][1]] for x in testAttr.keys()]
	print "Printing Training Set:"
	time.sleep(1)
	for row in trainSet:
		print row
		time.sleep(0.1)
	print
	print "Printing Test Set:"
	time.sleep(1)			
	for row in testSet:
		print row
		time.sleep(0.1)
	print
	if(method == 'gini'):
		print "Gini: " + str(gini(binnedTrainData))
	elif(method == 'info'):
		print "Information Entropy: "+ str(entropy(binnedTrainData))
	else:
		print "Error: " + str(misclassification(binnedTrainData))
	print
	print "Binned Training Set"
	time.sleep(1)
	for row in binnedTrainData:
		print row
		time.sleep(0.1)
	print
	print "Binned Testing Set"
	time.sleep(1)
	for row in binnedTestData:
		print row
		time.sleep(0.1)
	print
	print "Building ID3 tree"
	time.sleep(1)
	buildTree(binnedTrainData, binnedTestData, trainAttrList, method, thresh)
	
if __name__ == "__main__":
        main(sys.argv[1:5])

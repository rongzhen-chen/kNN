from numpy import zeros
import numpy as np
import operator
from os import listdir
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils.validation import check_random_state

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import RidgeClassifierCV


# convert image to vector
def img2vector(fileName,width_pixels,height_pixels):
    fr = open(fileName)
    returnVect = zeros((1,width_pixels*height_pixels))
    for i in range(height_pixels):
        lineStr = fr.readline()
        for j in range(width_pixels):
            returnVect[0,height_pixels*i+j] = int(lineStr[j]) 
    return returnVect


# calculate the error rate using kNN 
def handwritingClassTest():

    width_pixels = 32
    height_pixels = 32

    maxK = 10

    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)

    trainingMat = zeros((m,width_pixels*height_pixels))

    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        labelNumStr = int(fileStr.split('_')[0])
        hwLabels.append(labelNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/{0}'.format(fileNameStr),width_pixels,height_pixels)

    testFileList = listdir('testDigits')
    mTest = len(testFileList)

    dictErrorRate={}
    for iWeights, weights in enumerate(['uniform', 'distance']):
        listErrorRate=[]
        kList=np.arange(1,maxK+1)
        for numNeigh in kList:
            errorCount = 0.0
            for i in range(mTest):
                fileNameStr = testFileList[i]
                fileStr = fileNameStr.split('.')[0]
                labelNumStr = int(fileStr.split('_')[0])
                vectorUnderTest = img2vector('testDigits/{0}'.format(fileNameStr),width_pixels,height_pixels)
                neigh = KNeighborsClassifier(weights=weights, algorithm='auto',leaf_size=30,n_neighbors=numNeigh)
                neigh.fit(trainingMat, hwLabels) 
                returnResult = neigh.predict(vectorUnderTest)
                #print "the classifier came back with: {0}, the real answer is: {1}".format(returnResult,labelNumStr)
                if (returnResult != labelNumStr): errorCount += 1.0
            #print "\nthe total number of errors is: {0}".format(errorCount)
            #print "\nthe total error rate is: {0}".format(errorCount/float(mTest))
            listErrorRate.append(errorCount/float(mTest))
        dictErrorRate[weights]=listErrorRate

        plt.subplot(2, 1, iWeights + 1)
        plt.plot(kList,listErrorRate,'o')
        plt.ylabel('Total error rate')
        plt.title("K-nearest neighbors classification (weights = {0})".format(weights))
        if iWeights: plt.xlabel('Number of k')
        plt.axis('tight')
        plt.xlim([-1,11])
    plt.savefig('Total_error_rate.jpg', format='jpg', dpi=1000)    
    plt.show()
        
def imagPredictTest():

    width_pixels = 32
    height_pixels = 32

    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m,width_pixels*height_pixels))

    for i in range(m):
        fileNameStr = trainingFileList[i]
        trainingMat[i,:] = img2vector('trainingDigits/{0}'.format(fileNameStr),width_pixels,height_pixels)

    testFileList = listdir('testDigits')
    mTest = len(testFileList)
    testMat = zeros((mTest,width_pixels*height_pixels))

    for i in range(mTest):
        fileNameStr = testFileList[i]
        testMat[i,:] = img2vector('testDigits/{0}'.format(fileNameStr),width_pixels,height_pixels)

    # simple adapt from example of "Face completion with a multi-output estimators" in Python Manual
    train = trainingMat 
    test = testMat 

    # Test on a subset of people
    n_faces = 5
    rng = check_random_state(10)
    face_ids = rng.randint(test.shape[0], size=(n_faces, ))
    test = test[face_ids, :]

    n_pixels = width_pixels*height_pixels
    # Upper half of the faces
    X_train = train[:, :(n_pixels + 1) // 2]
    # Lower half of the faces
    y_train = train[:, n_pixels // 2:]
    X_test = test[:, :(n_pixels + 1) // 2]
    y_test = test[:, n_pixels // 2:]

    # Fit estimators (Regression)
    '''
    ESTIMATORS = {
        "Extra trees": ExtraTreesRegressor(n_estimators=10, max_features=2,
                                           random_state=0),
        "K-nn": KNeighborsRegressor(),
        "Linear regression": LinearRegression(),
        "Ridge": RidgeCV(),
    }
    '''
    # Fit estimators (mix Regression and Classifier)
    ESTIMATORS = {
        "Extra trees cf ": ExtraTreesClassifier(max_depth=3, n_estimators=10, random_state=0),
        "K-nn cf": KNeighborsClassifier(),
        "Linear regression": LinearRegression(),
        "Ridge": RidgeCV(),
    }

    y_test_predict = dict()
    for name, estimator in ESTIMATORS.items():
        estimator.fit(X_train, y_train)
        y_test_predict[name] = estimator.predict(X_test)

    # Plot the completed faces
    image_shape = (width_pixels, height_pixels)

    n_cols = 1 + len(ESTIMATORS)
    plt.figure(figsize=(2. * n_cols, 2.26 * n_faces))
    plt.suptitle("Number completion with multi-output estimators (cf=classfier)", size=16)

    for i in range(n_faces):
        true_face = np.hstack((X_test[i], y_test[i]))

        if i:
            sub = plt.subplot(n_faces, n_cols, i * n_cols + 1)
            print("i={0} in if i".format(i))
        else:
            sub = plt.subplot(n_faces, n_cols, i * n_cols + 1,
                              title="True numbers")
            print("i={0} in else".format(i))

        sub.axis("off")
        sub.imshow(true_face.reshape(image_shape),
                   cmap=plt.cm.gray,
                   interpolation="nearest")
        
        for j, est in enumerate(sorted(ESTIMATORS)):
            completed_face = np.hstack((X_test[i], y_test_predict[est][i]))

            if i:
                sub = plt.subplot(n_faces, n_cols, i * n_cols + 2 + j)

            else:
                sub = plt.subplot(n_faces, n_cols, i * n_cols + 2 + j,
                                  title=est)

            sub.axis("off")
            sub.imshow(completed_face.reshape(image_shape),
                       cmap=plt.cm.gray,
                       interpolation="nearest")
    plt.savefig('Number_completion_classifier.jpg', format='jpg', dpi=1000)    
    plt.show()

if __name__ == '__main__':

    handwritingClassTest()
    #imagPredictTest()

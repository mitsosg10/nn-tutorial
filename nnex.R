
#################### efarmogi 1i########################################################
##library(neuralnet)
#################### efarmogi 1i########################################################

##load the dataset and split into training and test sets
data(iris)
ind=sample(2,nrow(iris), replace=T, prob=c(0.7,0.3))
trainset=iris[ind==1,]
testset=iris[ind==2,]
set.seed(1)
head(iris)
##load library neuralnet
install.packages("neuralnet")
library("neuralnet")


##add the columns versicolor,setosa and virginica based on the names in Species variable
trainset$setosa=trainset$Species=="setosa"
trainset$virginica=trainset$Species=="virginica"
trainset$versicolor=trainset$Species=="versicolor"

##train the NN with neuralnet function with 3 hidden neurons 
##in each layer.use set.seed to have the same results in every training process

network= neuralnet(versicolor+virginica+setosa~
                     Sepal.Length+Sepal.Width+Petal.Length+Petal.Width,
                   data=trainset, hidden=3)


##View the summary information by accessing the 
##result.matrix attribute of the built neural network model

network$result.matrix

##view the generalized weights
head(network$generalized.weights)

##visualize the trained nn with plot
plot(network)

##furthermore you can use gwplot to visualize generalized weights

par(mfrow=c(2,2))
gwplot(network,selected.covariate="Petal.Width")
gwplot(network,selected.covariate="Sepal.Width")
gwplot(network,selected.covariate="Petal.Length")
gwplot(network,selected.covariate="Sepal.Length")

##predicting labels based on a model trained by neuralnet

##generate prediction probability matrix based on a trained nnand the testing
##dataset, testset

net.predict=compute(network,testset[-20])$net.result
net.predict
##obtain the possible labels by finding the column with the 
##gratest probability:

net.prediction=c("versicolor","virginica", "setosa")[apply(net.predict,1,which.max)]
net.predict
##generate a classification table based on the predicted labels
##and the labels of the testing dataset:

predict.table=table(testset$Species,net.prediction)
predict.table

##finally use confusionmatrix
library(caret)
install.packages("ggplot2")
install.packages("lettice")
confusionMatrix(predict.table)


#################### efarmogi 2i########################################################
library(nnet)
#################### efarmogi 2i########################################################

##split the dataset
data(iris)
str(iris)
set.seed(2)
ind=sample(2,nrow(iris),replace=T, prob=c(0.7,0.3))
trainset=iris[ind==1,]
testset=iris[ind==2,]

##load library nnet
library(nnet)

##train nn
iris.nn= nnet(Species~.,data=trainset,size=2,rang=0.1,decay=5e-4,
              maxit=200)

##predicting labels on a trained model

##generate prediction of the test
iris.predict=predict(iris.nn,testset,type="class")

##generate cof matrix
nn.table=table(testset$Species,iris.predict)

library(caret)
confusionMatrix(nn.table)


#################### efarmogi 3i########################################################
##library(RSNNS)
#################### efarmogi 3i########################################################
library(RSNNS)
data(iris)


#shuffle the vector
iris <- iris[sample(1:nrow(iris),length(1:nrow(iris))),1:ncol(iris)]
irisValues <- iris[,1:4]
irisTargets <- decodeClassLabels(iris[,5])

iris <- splitForTrainingAndTest(irisValues, irisTargets, ratio=0.15)
iris <- normTrainingAndTestSet(iris)

model <- mlp(iris$inputsTrain, iris$targetsTrain, size=c(2,5), 
             maxit=50, inputsTest=iris$inputsTest, targetsTest=iris$targetsTest)
model

weightMatrix(model)

extractNetInfo(model)

predictions <- predict(model,iris$inputsTest)

par(mfrow=c(2,2))
plotIterativeError(model)
plotRegressionError(predictions[,2], iris$targetsTest[,2])
plotROC(fitted.values(model)[,2], iris$targetsTrain[,2])
plotROC(predictions[,2], iris$targetsTest[,2])

library(caret)
confusionMatrix(iris$targetsTrain,fitted.values(model))
confusionMatrix(iris$targetsTest,predictions)


library(NeuralNetTools)
plotnet(model)
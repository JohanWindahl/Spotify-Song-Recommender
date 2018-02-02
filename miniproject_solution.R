library(caret)

#Load files
setwd("E:/Dropbox/SkolaATM/ML-S/ProjectSolution") #@home
#setwd("C:/Dropbox/SkolaATM/ML-S/ProjectSolution") #@laptop

dataset <- read.csv('training_data.csv',header = T)
pred <- read.csv('songs_to_classify.csv',header = T)
levels(dataset$label) <- c(0,1) #Rename levels to: 0=dislike 1=like


#Controls
trainRatio = 0.8 #Data ratio
k = 10 #K-fold cross validation
r = 3  #Cross validation repeats

createNewModels = TRUE #Calculate new models?
printResults = TRUE #Print summary?
plotResults = TRUE #Plot summary?
checkAllModelsAgainstValidation = TRUE #Evaluate all models?
seed <- 1 #Set seed 
set.seed(seed);print(paste0("Seed: ",seed))

#Data splitting
ratioTest = 1 - trainRatio #Test ratio
trainNORows <- trainRatio*nrow(dataset) #Number of training rows of dataset
evalNORows <- nrow(dataset) - trainNORows #Number of validation rows of dataset

dataset_shuffled <- dataset[sample(nrow(dataset)),] #Shuffle rows
dataset_train <- dataset_shuffled[1:trainNORows,] #Training dataset
dataset_eval <- dataset_shuffled[(trainNORows + 1):nrow(dataset_shuffled),] #Validation dataset

print(paste0("Samples: ",nrow(dataset), 
             " | Training on rows ", 1,":",  trainNORows, " Ratio:", trainRatio, 
             " | Validation on rows ", (trainNORows + 1),":",nrow(dataset_shuffled), " Ratio:", ratioTest))


#Model/Hyper parameters
metric <- "Accuracy"
preProcess = c("center", "scale")



control <- trainControl(method = "repeatedcv", number = k, repeats = r)
print(paste0("Using K-Fold cross validation, with K=",k,", Repeats=",r))
model <- label ~ danceability + energy + key + loudness +   mode + speechiness + 
                 acousticness + instrumentalness + liveness + valence + tempo + duration + 
                 time_signature


#Model creation
if (createNewModels) {
  print("Creating Models")
  
  # (i) Logistic Regression - Generalized Linear Models
  print("Generalized Linear Models -> fit.glm")
  set.seed(seed)
  fit.glm <- train(model, data = dataset_train, method = "glm", metric = metric, trControl = control)
  
  # (ii) Linear Discriminant Analysis
  print("Linear Discriminant Analysis -> fit.lda")
  set.seed(seed)
  fit.lda <- train(model, data = dataset_train, method = "lda", metric = metric, preProc = c("center", "scale"), trControl = control)
  
  # (ii) Quadratic Discriminant Analysis	
  print("Quadratic Discriminant Analysis -> fit.qda")
  set.seed(seed)
  fit.qda <- train(model, data = dataset_train, method = "qda", metric = metric, preProc = c("center", "scale"), trControl = control)
  
  # (iii) K-nearest neighbor
  print("K-nearest neighbor -> fit.knn")
  set.seed(seed)
  fit.knn <- train(model, data = dataset_train, method = "knn", metric = metric, preProc = c("center", "scale"), trControl = control)
  
  # (iv) Random Forest
  print("Random Forest -> fit.rf")
  set.seed(seed)
  fit.rf <- train(model, data = dataset_train, method = "rf", metric = metric, trControl = control)
  
  # (v) Stochastic Gradient Boosting (Generalized Boosted Modeling)
  print("Generalized Boosted Modeling -> fit.gbm")
  set.seed(seed)
  fit.gbm <- train(model, data = dataset_train, method = "gbm", metric = metric, trControl = control, verbose = FALSE)
  
  
  #Some other models
  
  # SVM Radial
  print("SVM Radial -> fit.svmRadial")
  set.seed(seed)
  fit.svmRadial <- train(model, data = dataset_train, method = "svmRadial", metric = metric, preProc = c("center", "scale"), trControl = control, fit = FALSE)
  
  # Naive Bayes
  print("Naive Bayes -> fit.nb")
  set.seed(seed)
  fit.nb <- train(model, data = dataset_train, method = "nb", metric = metric, trControl = control) #warnings
  
  # GLMNET
  print("GLMNET -> fit.glmnet")
  set.seed(seed)
  fit.glmnet <- train(model, data = dataset_train, method = "glmnet", metric = metric, preProc = c("center", "scale"), trControl = control)
  
  # Neural Network
  print("Neural Network -> fit.nnet")
  set.seed(seed)
  #fit.nnet <- train(model, data = dataset_train, method = "nnet", metric = metric, preProc = c("center", "scale"), trControl = control)

  
  # Deep Neural Network
  print("Deep Neural Network -> fit.dnn")
  set.seed(seed)
  #fit.dnn <- train(model, data = dataset_train, method = "dnn", metric = metric, preProc = c("center", "scale"), trControl = control)


  listOfModels = list(glm = fit.glm, lda = fit.lda, qda = fit.qda, knn = fit.knn, 
                      rf = fit.rf, gbm = fit.gbm, svm = fit.svmRadial, nb = fit.nb, 
                      glmnet = fit.glmnet)#, nnet = fit.nnet, dnn = fit.dnn)
  
  results <- resamples(listOfModels)

  print("Models Done")
}

#Print/Plot results
if (printResults) {
  results <- resamples(listOfModels)
  print(summary(results))
}
if (plotResults) {
  bwplot(results)
  dotplot(results)
}

#Validate each model against validation data
if (checkAllModelsAgainstValidation) {
  for (i in listOfModels) {
    tempModelName <- paste0("fit.",i$method)
    assign("tempModelPred", predict(eval(parse(text = tempModelName)), dataset_eval[,1:13]))
    evalCorrect <- sum(tempModelPred == dataset_eval$label)
    print(paste0(tempModelName, " evaluation: ", evalCorrect, "/",length(tempModelPredNum), " correct,  ",(evalCorrect/length(tempModelPredNum)*100),"%"))
  }
}



#Production prediction
resultSongsToClassify <- predict(fit.gbm, pred)

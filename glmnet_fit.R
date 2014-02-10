# Author: Andrew Beam #
# beam.andrew@gmail.com #
# Code to accompany the manuscript:
# Beam, A.L., Ghosh, S.J., Doyle, J. Fast Hamiltonian Monte Carlo Using GPU Computing. #

# Load required libraries #
library(glmnet)
library(doMC)
registerDoMC()

# Set the location of the data #
data.dir <- "/data/mnist/"
X <- as.matrix(read.csv(paste0(data.dir,"train_x.csv"),header=F))
Y <- as.matrix(read.csv(paste0(data.dir,"train_y.csv"),header=F))
X_test <- as.matrix(read.csv(paste0(data.dir,"test_x.csv"),header=F))
Y_test <- as.matrix(read.csv(paste0(data.dir,"test_y.csv"),header=F))

# Set glmnet parameters
nlambda=500
print("Fitting the model...")
print(paste0("Number of lambda values: ",nlambda))
# Time how long it takes to fit the model #
# Note this was fit on a multicore system, so each fold is fit on a seperate CPU core #
system.time(cv.fit <- cv.glmnet(x=X, y=Y, type.measure="class",nfolds=5,parallel=TRUE,standardize=FALSE,
                    nlambda=nlambda,family="multinomial"))
preds <- (predict(cv.fit,X,type="response"))[,,1]
test_preds <- (predict(cv.fit,X_test,type="response"))[,,1]

train_truth <- apply(Y,1,which.max)
train_labels <- apply(preds,1,which.max)
errors <- length(which(abs(train_truth-train_labels)>0))
print(paste0("Training accuracy: ",1-errors/nrow(preds)))

test_truth <- apply(Y_test,1,which.max)
test_labels <- apply(test_preds,1,which.max)
test_errors <- length(which(abs(test_truth-test_labels)>0))
print(paste0("Test accuracy: ",1-test_errors/nrow(test_preds)))

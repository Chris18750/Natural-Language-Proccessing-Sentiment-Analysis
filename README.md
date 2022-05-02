# Natural-Language-Proccessing: Sentiment Analysis
This Github respository contains the source code for five methods that do sentiment analysis on a IDMB data set which contains multiple film reviews.

## Table of contents
* [Intro](#intro)
* [Convolutional Neural Network](#convolutional-neural-network)
* [Multinomial Naive Bayes](#multinomial-naive-bayes)
* [Gaussian Naive Bayes](#gaussian-naive-bayes)
* [Complement Naive Bayes](#complement-naive-bayes)
* [Bernoulli Naive Bayes](#bernoulli-naive-bayes)
* [Results](#results)
* [References](#references)

## Intro
As a group we decided to study sentiment analysis a technique in natural language processing. The goal of this study is to find out what machine learning methods are used to do sentiment analysis and how accurate they can be. Our hypothesis is that there are multiple machine learning methods which vary in accuracy. We used the programming language Python and multiple libraries to test five different machine learning methods. Each five methods produced different accuracies relatively close to 85%. However, Gaussian Naive Bayes was slightly below 80%. Meaning most methods were about the same in performance except Gaussian Naive Bayes

## Convolutional Neural Network
A convolutional neural network is a deep learning algorithm usually used for image processing. However, we decided to implement it for sentiment analysis to test its accuracy on determining the sentiment in text. The training accuracy was 85.37% and the test accuracy was 84.64%. 

## Multinomial Naive Bayes
The Multinomial Naive Bayes algorithm is a machine learning method that is popular in natural language processing. The algorithm uses a multinomial distribution for each feature. The training accuracy was 85.7% and the test accuracy was 84.63%. 

## Gaussian Naive Bayes
The Gaussian Naive Bayes algorithm is a machine learning method which is usually used when the features have continuous values. The training accuracy was 81.26% and the test accuracy was 79.22%. 

## Complement Naive Bayes
The Complement Naive Bayes algorithm is a machine learning method which was designed to correct severe assumptions. The training accuracy was 85.69% and the test accuracy was 84.64%. 

## Bernoulli Naive Bayes
The Bernoulli Naive Bayes algorithm is a machine learning method which is used on data that is meant to be in binary value such as true or false. The training accuracy was 85.39% and the test accuracy was 84.35%. 

## Results
| Data  | CNN | MNB | GNB | CNB | BNB |
|-------|-----|-----|-----|-----|-----|
| Train | 85.37% |  85.7%  | 81.26  | 85.69%  |  85.39%  |
| Test  | 84.64%  | 84.63%  |  79.22%  | 84.64%  | 85.35%  |

## References
https://docs.w3cub.com/scikit_learn/modules/generated/sklearn.naive_bayes.multinomialnb
https://www.analyticsvidhya.com/blog/2021/07/performing-sentiment-analysis-with-naive-bayes-classifier/
https://machinelearningmastery.com/develop-word-embedding-model-predicting-movie-review-sentiment/
https://builtin.com/data-science/how-build-neural-network-keras
https://www.analyticsvidhya.com/blog/2021/07/performing-sentiment-analysis-with-naive-bayes-classifier/
https://www.analyticsvidhya.com/blog/2021/07/performing-sentiment-analysis-with-naive-bayes-classifier/
https://www.analyticsvidhya.com/blog/2021/07/performing-sentiment-analysis-with-naive-bayes-classifier/

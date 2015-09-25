1. generate a file 'user_item_score': userid, item, score
2. spark-submit negative.py user_item_score itemFeature.txt predict_lambda5.txt 5
3. python pca_data.py predict_lambda5.txt itemFeature.txt: get the vectors for each item
4. generate training data: 


als method:
1. python generateData.py data/predict_4_lambda5.txt 1,5 619
   python generateData.py data/predict_4_lambda5.txt 2,7 620
   python generateData.py data/predict_4_lambda5.txt 3,4,6,8,10 621
   python generateData.py data/predict_4_lambda5.txt 9 622

2. spark-submit als_update.py 620 data/618_4.txt out620.txt 5  #get the result



----------
spark-submit svm.py data/618_4.txt 3,4,6,8,10 als/out621.txt #compare two methods 

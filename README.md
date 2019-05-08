# RNN_LSTM-on-Energy-Data-
RNN_LSTM Implementation on open source energy data using Python 3.0 with Tensorflow back end
Hello
There are various solutions can be generated using given energy data. I have tried Radom Forest and Logistic Regression machine learning techniques to optimize results however I am getting accuracy of 50% approx... So I thought to do some experiment using deep learning. I have started working on deep learning before 6 months and now I am much familiar with it. The given problem is completely based on regression analysis. 
Here, I am trying to predict future energy consumption of these (R1-R9) rooms.
The following are key findings (K)
K1: The total consumption of energy in these rooms depends more on temperature than humidity.
K2: I can predict total consumption of any specific day if I have room temperature and humidity data
K3: I can compare actual energy consumption with predicted consumption and use the difference as additional factor for future predictions. Following is the procedure
LSTM deep learning algorithm
LSTM networks are well-suited to classifying, processing and making predictions based on time series data, since there can be lags of unknown duration between important events in a time series. I used Keras python library with Tensor flow backend. I also used scikit-learn packages and anaconda jupyter environment. 
Data Preprocessing
For data preprocessing, scaling and normalization of values has been performed which is universal preprocessing technique especially for LSTM. Following is the code 
# Reshape the data into X and Y for LSTM Model 
X = np.concatenate([R1,R2,R3,R4,R5,R6,R7,R8,R9],axis=0)
X = np.transpose(X)
Y = np.transpose(TotalConsmp)
sc= MinMaxScaler()
sc.fit(X)
X=sc.transform(X)
sc= MinMaxScaler()
sc.fit(Y)
Y=sc.transform(Y)
X = np.reshape(X,(X.shape[0],1,X.shape[1]))

Modelling Process
I’ve used LSTM model to predict the total energy consumption based on two different parameters given in data. 1) Room temperature and 2) Humidity. Showed in different graphs bellow: 
In trained LSTM model, I used activation function as a RELU and mean squared error as a loss function. The optimizer is RMSprop. To evaluate this model, I used Mean Absolute Error (MAE). I trained the designed and trained LSTM model till 100 epochs and got pretty good results as listed above key factors given bellow:  
Results
# Plot the graph
#colnames = ['Epochs','MAE']
data = pd.read_csv('Epoch_humidity.csv')
Epoch= list(data.Epoch)
MAE = data.MAE.tolist()

plt.plot(Epoch, MAE)
plt.xlabel('No. of Epochs')
plt.ylabel('Mean_Absolute Error')
plt.title('MAE for each Epoch on LSTM Model wrt Humidity')
plt.show()

 
Figure 1 : MAE for epoch on LSTM model with respect to Humidity

 
Figure 2 : MAE for epoch on LSTM model with respect to Humidity
So the key finding is, when LSTM model has higher minimization rate when trained with room temperature as compare to humidity levels with increase in epoch level. 
#Plotting the graph for the real values with the predicted value 
plt.figure(1)
Real = plt.plot(Y_test)
Predict = plt.plot(predict)
plt.show()

 
Figure 3 : Prediction of total energy consumption based on Humidity
 
 
Figure 4 : Prediction of total energy consumption based on room temperature
The figure 3 (based on humidity) and 4 (based on room temperature) shows prediction of total energy consumption.  The orange color shows the energy consumption predicted by trained LSTM model whereas blue color shows the real values of energy consumption in data.  

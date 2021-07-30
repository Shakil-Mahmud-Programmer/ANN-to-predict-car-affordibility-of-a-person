#imported packages
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd

data=pd.read_csv("Data/car.csv")
df=data.values
x=df[0:20, 0:2]
y=df[0:20,2:3]
test_x=df[21:25,0:2]
test_y=df[21:25,2:3]


model=Sequential()
model.add(Dense(units=2,activation='relu',input_dim=2))
model.add(Dense(units=1,activation='sigmoid'))

model.compile(optimizer='sgd',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(x,y,epochs=15)
#model.save('model')
p,q=model.evaluate(test_x,test_y)
print("Loss="+str(p)+" Accuracy"+str(q))
a=int(input("Enter the age: "))
b=int(input("Enter afordibility (0/1): "))
if model.predict([[49,1]])>0.5:
    print("Yes he/she might has a car")
else:
    print("He/she might not has a car")


import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVR

# ['temperature_2m (C)','relativehumidity_2m (%)','direct_radiation (W/m)','diffuse_radiation (W/m)','direct_normal_irradiance (W/m)','windspeed_10m (km/h)']
df = pd.read_csv(r'C:\Users\Keerthana\Desktop\solar panel\DA\solar power\dataset\chennaitime.csv')


cols = ['temperature_2m (C)','relativehumidity_2m (%)','direct_radiation (W/m)','diffuse_radiation (W/m)','direct_normal_irradiance (W/m)','windspeed_10m (km/h)']
le = LabelEncoder()
for i in cols:
    df[i] = le.fit_transform(df[i])


# df=df.drop(['Loan_ID','ApplicantIncome','CoapplicantIncome'], axis = 1)

X = df.drop(['temperature_2m (C)','relativehumidity_2m (%)','direct_radiation (W/m)','diffuse_radiation (W/m)','direct_normal_irradiance (W/m)','windspeed_10m (km/h)','power'], axis=1)
y = df['power']
X.head()
y.head()
# model = LogisticRegression()
model= SVR()
# model = RandomForestClassifier()
#model.fit(x_train, y_train)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=20) 

model.fit(X_train, y_train)

file = open(r'C:\Users\Keerthana\Desktop\solar panel\DA\solar power\model2.pkl', 'wb')
pickle.dump(model, file)
y_pred = model.predict(X_test)
print(X_test)
#cm = confusion_matrix(y_test, y_pred)  
print(y_pred)
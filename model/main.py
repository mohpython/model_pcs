import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle5 as pickle



def create_model(data): 
  X = data.drop(['diagnosis'], axis=1)
  y = data['diagnosis']
  
  # La Normalisation des donnees
  scaler = StandardScaler()
  X = scaler.fit_transform(X)
  
  # Division des donnees en train et test
  X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
  )
  
  # entrainement de mon model
  model = LogisticRegression()
  model.fit(X_train, y_train)
  
  # test du model
  y_pred = model.predict(X_test)
  print('Accuracy du model: ', accuracy_score(y_test, y_pred))
  print("Rapport de classification: \n", classification_report(y_test, y_pred))
  
  return model, scaler


def get_clean_data():
    data = pd.read_csv("data/data.csv")

    #Supprimons la colonne unname car tous les elmts sont NaN
    data = data.drop(['Unnamed: 32', 'id'], axis=1)
  
    data['diagnosis'] = data['diagnosis'].map({ 'M': 1, 'B': 0 })
  
    return data

   

def main():
    data = get_clean_data()

    model, scaler = create_model(data)

    # Enregistrons le scaler et le model pour pouvoir l'utiliser dans mon Application web 

    with open('model/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    
    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    

if __name__ == '__main__':
    main()

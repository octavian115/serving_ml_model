import pickle
import pandas as pd

#loading the model
with open('model/model.pkl','rb') as f:
    model = pickle.load(f)

#usually from mlflow
MODEL_VERSION = '1.0.0'

#get class labels from the model(important to match probablities to class names)
class_labels = model.classes_.tolist()

def predict_output(user_input: dict):
    
    df = pd.DataFrame([user_input])

    #predict the class
    predicted_class = model.predict(df)[0]

    #get the probablities for all the classes
    probablities = model.predict_proba(df)[0]
    confidence = max(probablities)
    
    #create mapping:  {class_name: probablity}
    class_probs = dict(zip(class_labels, map(lambda p: round(p, 4), probablities)))

    return {
        "predicted_category": predicted_class,
        "confidence": round(confidence, 4),
        "class_probablities": class_probs
    }
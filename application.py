from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)
model = pickle.load(open(r'models/LassoCVModel.pkl', 'rb'))
df = pd.read_csv(r'notebooks/Housing.csv')

@app.route('/')
def index():
    bedrooms = sorted(df['bedrooms'].unique())
    bathrooms = sorted(df['bathrooms'].unique())
    stories = sorted(df['stories'].unique())
    mainroad = sorted(df['mainroad'].unique())
    guestroom = sorted(df['guestroom'].unique())
    basement = sorted(df['basement'].unique())
    hotwaterheating = sorted(df['hotwaterheating'].unique())
    airconditioning = sorted(df['airconditioning'].unique())
    parking = sorted(df['parking'].unique())
    prefarea = sorted(df['prefarea'].unique())
    furnishingstatus = sorted(df['furnishingstatus'].unique())
    return render_template('index.html', bedrooms=bedrooms, bathrooms= bathrooms, stories=stories, mainroad=mainroad, guestroom=guestroom, basement=basement,hotwaterheating=hotwaterheating, airconditioning=airconditioning, parking=parking, prefarea=prefarea, furnishingstatus=furnishingstatus)

@app.route('/predictdata', methods=['GET', 'POST'])
def predict():
    if request.method=='POST':
        area=int(request.form.get('area'))
        bedrooms = int(request.form.get('bedrooms'))
        bathrooms = int(request.form.get('bathrooms'))
        stories = int(request.form.get('stories'))
        mainroad = str.lower(request.form.get('mainroad'))
        guestroom = str.lower(request.form.get('guestroom'))
        basement = str.lower(request.form.get('basement'))
        hotwaterheating = str.lower(request.form.get('hotwaterheating'))
        airconditioning = str.lower(request.form.get('airconditioning'))
        parking = int(request.form.get('parking'))
        prefarea = str.lower(request.form.get('prefarea'))
        furnishingstatus = str.lower(request.form.get('furnishingstatus'))

        prediction = model.predict(pd.DataFrame([[area, bedrooms, bathrooms, stories, mainroad, guestroom, basement, hotwaterheating,
                                                  airconditioning, parking, prefarea, furnishingstatus]], columns=['area','bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'parking', 'prefarea', 'furnishingstatus']))
        return render_template('index.html', prediction="INR: "+ str(round(prediction[0])))

    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)

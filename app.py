from flask import Flask, request, render_template
from src.pipelines.predict_pipeline import PredictPipeline, CustomData

# Create a flask app
application = Flask(__name__)
app = application

# define the route for homepage
@app.route('/')
def index():
    return render_template('index.html')


# define the route for the prediction pipeline
@app.route('/predict', methods=['GET','POST'])
def predict_datapoint():
    if request.method== 'GET':
        return render_template('home.html')
    elif request.method=='POST':
        # collect the data from frontend
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score')))

        # convert it into df
        data_df = data.get_data_as_df()

        # initialize the prediction pipeline now
        print(data_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(data_df)
        print("after Prediction")
        return render_template('home.html',results=results[0])


if __name__=="__main__":
    app.run(host='0.0.0.0', debug=True)
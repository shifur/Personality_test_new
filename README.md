## Big five personalities

## Dataset
[https://www.kaggle.com/tunguz/big-five-personality-test](https://www.kaggle.com/tunguz/big-five-personality-test)

# Abstract
This free personality test gives you accurate scores for the Big Five personality Traits. 
See exactly how you score for Openness, Conscientiousness, Extraversion, Agreeableness, and Neuroticism with this 
scientific personality assessment. To take the Big Five personality assessment, rate each statement according to how well it describes you. 
Base your ratings on how you really are, not how you would like to be.

# Stack
- React (Frontend)
- Flask (Backend Api)
- Model (Colab Notebook)

## Setup

## Setup
This project requires **Python 3.11**.

```bash
# create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate  # on Windows use "venv\Scripts\activate"

# install dependencies
pip install -r requirements.txt
```

Download `data-final.csv` from Kaggle and place it in the `data/` directory.
Run the training script to create `trained_Model.pkl`:

```bash
python train_model.py
```

### Running the app
Start the Flask backend:

```bash
python server.py
```

Then start the React frontend in another terminal:

```bash
cd client
npm install
npm start
```
Create a `.env` file in the `client` folder (you can copy `.env.example`) and
adjust `REACT_APP_API_URL` if your backend runs on a different URL or port.
### Deploying to Heroku
1. Ensure `requirements.txt`, `Procfile`, and `runtime.txt` are committed.
2. Push the repository to GitHub.
3. Create a free Heroku app and connect it to your repo.
4. Set the buildpack to `heroku/python` and deploy.
5. Once deployed, open the app URL to test the API.


### NumPy compatibility
If you hit errors related to NumPy 2.x incompatibility, reinstall the packages
or downgrade NumPy:

```bash
pip uninstall numpy
pip install "numpy<2"
pip install --force-reinstall -r requirements.txt
```

# Machine learning Algorithms
- Clustering (K-means)
- Classifier (Random Forest)
- PCA (Dimensionality Reduction and Cluster Visualization)

## FAQ

#### Q. What personality traits does this Big Five test measure?<br>
A. The Big Five personality test measures the five personality factors that psychologists have determined are core to our personality makeup. The Five Factors of personality are:

`Openness - How open a person is to new ideas and experiences`<br>
`Conscientiousness - How goal-directed, persistent, and organized a person is`<br>
`Extraversion - How much a person is energized by the outside world`<br>
`Agreeableness - How much a person puts others' interests and needs ahead of their own`<br>
`Neuroticism - How sensitive a person is to stress and negative emotional triggers`<br>
<br>
> The Big Five model of personality is widely considered to be the most scientifically robust way to describe personality differences. It is the basis of most modern personality research.
<br>

#### Q. How long is this test?<br>
A. The test consists of 50 questions and takes about 10-15 minutes to complete.
 
#### Q. What will I learn from my test report?<br>
A. You will first see a brief, free report showing the basic findings of your personality test. Then, you have the option of unlocking your full report for a small fee. To see what you can expect from your full report, check out this sample Big Five report.

#### Q. Is this personality test really free?<br>
A. You do not need to purchase or register to take this test and view an overview of your results. If you would like, you can purchase a more comprehensive full report for a small fee.

#### Q. Is this personality test scientific and/or accurate?<br>
A. This test has been researched extensively to ensure it is valid and reliable. It is based on psychological research into the core of personality, and our own psychometric research. Your scores show you how you compare to the other people in a large, international sample for each of the Big Five personality traits.

#### Q. Can I have my employees, team or group take the Big Five test?<br>
A. Absolutely. Our Truity @ Work platform is designed to make it easy to give the Big Five personality test to your team or group. See discounted group pricing and learn how to quickly and easily set up testing for your group on the Testing for Business page.

#### Q. What is the difference between Big Five, Five Factor, and the OCEAN model of personality?<br>
A. Big Five, Five Factor, and OCEAN are all ways of describing the same theory of personality. Multiple psychological studies have arrived at the conclusion that the differences between people's personalities can be organized into five broad categories, called the Big Five or Five Factors. These are sometimes referred to as the five broad dimensions of personality.

## Training the Model
To generate the `trained_Model.pkl` used by the backend, download `data-final.csv` from Kaggle and place it under `data/`. Then run:
```bash
python train_model.py
```
This will create the pickle file in the project root that the Flask app loads.

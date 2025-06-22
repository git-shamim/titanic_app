This is a Streamlit-based ML web app that predicts survival on the Titanic using user inputs.

## 🚀 Features
- Interactive web UI
- Random Forest model
- Docker-ready for GCP Cloud Run

## 🛠 Files
- `app.py`: Streamlit app
- `train_model.py`: Model training script
- `train.csv`: Titanic dataset
- `titanic_model.pkl`: Trained model
- `Dockerfile`: Deployment file
- `requirements.txt`: Python packages

## 🧪 Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## ☁️ Deploy to GCP Cloud Run
1. Push code to GitHub.
2. Connect repo in Cloud Run.
3. Deploy using Dockerfile.
4. Set `--min-instances=1` to avoid cold start.

## 📦 Build Docker Image (optional)
```bash
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/titanic-app
gcloud run deploy titanic-app --image gcr.io/YOUR_PROJECT_ID/titanic-app --platform managed --allow-unauthenticated

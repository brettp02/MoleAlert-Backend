# Mole Alert

This is the repository for the Backend/Model of [MoleAlert](https://github.com/brettp02/MoleAlert-Web?tab=readme-ov-file). It is a [FastAPI-based REST API](https://github.com/brettp02/MoleAlert-Backend). Which uses computer vision in the form of a Resnet50 model which is trained using transfer learning, containerized with Docker and hosted on AWS. 
This is where you can find the different parts in this repo:
- Exploratory Data Analysis: `/EDA`
- Training and Testing Images: `/data`
- Model Training and Testing: `/code`
- RestAPI code (FastAPI): `/app/main` for the API and `/app/model_starter` for the test logic.
- `requirements.txt` and `Dockerfile` are also included.

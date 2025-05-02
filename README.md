# Brain Tumor Classification

A deep learning model for classifying brain tumors from MRI images into four categories: glioma, meningioma, pituitary, and no tumor. This project includes a Flask API for model inference that can be deployed to Google Kubernetes Engine.

![Brain Tumor Classification](https://img.shields.io/badge/ML-Brain%20Tumor%20Classification-blue)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Flask](https://img.shields.io/badge/Flask-2.x-green)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)
![Kubernetes](https://img.shields.io/badge/Kubernetes-Ready-blue)

## Project Structure

- `data/` - Contains raw, processed, and augmented image datasets
- `models/` - Contains trained models and evaluation results
- `flask/` - API service for inference
  - `app.py` - Flask application
  - `best_model.keras` - Trained model file for inference
  - `classes.txt` - Class names for prediction output
  - `Dockerfile.*` - Docker configuration for different environments
  - `deployment.yaml` - Kubernetes deployment configuration
  - `service.yaml` - Kubernetes service configuration
## GitHub CI/CD Integration

This project uses GitHub Actions for CI/CD to automatically build Docker containers and deploy to Google Kubernetes Engine (GKE).

### Step 1: Set Up Google Cloud Platform

1. **Create Project (if not already done)**:

   ```bash
   gcloud projects create cloud-braintumer --name="Brain Tumor Classification"
   gcloud config set project cloud-braintumer
   ```

2. **Enable Required APIs**:

   ```bash
   gcloud services enable container.googleapis.com artifactregistry.googleapis.com
   ```

3. **Create Service Account for GitHub Actions**:

   ```bash
   # Create service account
   gcloud iam service-accounts create github-actions-sa \
     --display-name="GitHub Actions Service Account"

   # Assign necessary roles
   gcloud projects add-iam-policy-binding cloud-braintumer \
     --member="serviceAccount:github-actions-sa@cloud-braintumer.iam.gserviceaccount.com" \
     --role="roles/artifactregistry.admin"

   gcloud projects add-iam-policy-binding cloud-braintumer \
     --member="serviceAccount:github-actions-sa@cloud-braintumer.iam.gserviceaccount.com" \
     --role="roles/container.admin"

   gcloud projects add-iam-policy-binding cloud-braintumer \
     --member="serviceAccount:github-actions-sa@cloud-braintumer.iam.gserviceaccount.com" \
     --role="roles/storage.admin"
   ```

4. **Create and Download Service Account Key**:
   ```bash
   gcloud iam service-accounts keys create github-sa-key.json \
     --iam-account=github-actions-sa@cloud-braintumer.iam.gserviceaccount.com
   ```

### Step 2: Create GitHub Repository

1. **Initialize Git and Push to GitHub**:

   ```bash
   # Use the provided script
   bash setup_git.sh

   # Then follow instructions to set up your remote repository
   git remote add origin https://github.com/YOUR_USERNAME/BrainTumorClassification.git
   git branch -M main
   git push -u origin main
   ```

### Step 3: Configure GitHub Repository Secrets

1. **Navigate to your GitHub repository**
2. **Go to Settings > Secrets and variables > Actions**
3. **Add the following repository secrets**:
   - `GCP_PROJECT_ID`: `cloud-braintumer` (or your project ID)
   - `GCP_REGION`: `europe-west1` (or your preferred region)
   - `GCP_SA_KEY`: _Paste the entire content of the github-sa-key.json file_

### Step 4: Trigger Workflow

The CI/CD workflow will automatically run when you push to the main or master branch. You can also manually trigger it:

1. **Go to Actions tab in your GitHub repository**
2. **Select "Build and Deploy to GKE" workflow**
3. **Click "Run workflow"**
4. **Select the environment (prod, dev, or test)**
5. **Click "Run workflow" button**

### Manual Deployment

You can also deploy manually using the provided scripts:

```bash
# Set up environment and build images
cd flask
source ./env.sh

# Deploy to Kubernetes
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml

# Check deployment status
kubectl get pods -n prod
kubectl get services -n prod
```

### Workflow Steps Explained

The GitHub Actions workflow performs these steps automatically:

1. **Checkout code**: Pulls your code from GitHub
2. **Set up Docker**: Configures Docker Buildx for multi-platform builds
3. **Set up Google Cloud SDK**: Authenticates with GCP using your service account
4. **Configure Docker for Artifact Registry**: Sets up Docker to push to Google Cloud
5. **Build Docker image**: Builds the container using Dockerfile.prod
6. **Push to Artifact Registry**: Uploads the container to Google Cloud
7. **Deploy to GKE**: Updates the deployment with the new container version
8. **Verify deployment**: Checks that pods are running correctly

## API Usage

Once deployed, the API provides a web interface for uploading and classifying brain MRI images.

- **Endpoint**: `/predict`
- **Method**: POST
- **Input**: Form data with an 'image' field containing the MRI image
- **Output**: JSON with classification results

## Model Information

The deployed model is an EfficientNet-based transfer learning model trained on augmented MRI images with the following classes:

- Glioma
- Meningioma
- No tumor
- Pituitary

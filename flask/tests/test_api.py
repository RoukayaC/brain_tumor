# tests/test_api.py
import requests
import base64
import os
import pytest

# URL of the running Flask application (adjust if running differently)
# For local testing: 'http://127.0.0.1:8080'
# If testing inside docker-compose: 'http://web:8080' (service name 'web')
BASE_URL = "http://127.0.0.1:8080" # Assume local testing for now

# Path to a sample image file (you need to provide one!)
# Create a dummy image or use a real one for testing
SAMPLE_IMAGE_PATH = "sample_test_image.jpg" 
# Make sure SAMPLE_IMAGE_PATH exists relative to where pytest runs
# Or provide an absolute path for simplicity in testing setup

@pytest.fixture(scope="module")
def api_url():
    """Provides the base URL for the API."""
    # Optional: Add logic here to wait for the server to be ready
    # For now, assume server is running before tests execute
    try:
         response = requests.get(f"{BASE_URL}/") # Check health endpoint
         response.raise_for_status() # Raise an exception for bad status codes
    except requests.exceptions.ConnectionError:
         pytest.fail(f"Could not connect to API at {BASE_URL}. Is the server running?")
    except requests.exceptions.RequestException as e:
         pytest.fail(f"API health check failed: {e}")

    return BASE_URL

@pytest.fixture(scope="module")
def sample_image_base64():
    """Reads the sample image and encodes it in base64."""
    if not os.path.exists(SAMPLE_IMAGE_PATH):
         pytest.skip(f"Sample image not found at {SAMPLE_IMAGE_PATH}, skipping API test.")
         return None # Skip test if image not found

    with open(SAMPLE_IMAGE_PATH, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def test_health_check(api_url):
    """Tests the health check endpoint."""
    response = requests.get(f"{api_url}/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_predict_endpoint(api_url, sample_image_base64):
    """Tests the /predict endpoint with a valid image."""
    if sample_image_base64 is None:
        pytest.skip("Sample image base64 not available.") # Skip if fixture skipped

    headers = {'Content-Type': 'application/json'}
    payload = {'image': sample_image_base64}

    response = requests.post(f"{api_url}/predict", json=payload, headers=headers)

    assert response.status_code == 200
    json_response = response.json()
    assert "predicted_class" in json_response
    assert "confidence" in json_response
    assert "probabilities" in json_response
    assert isinstance(json_response["confidence"], float)
    assert isinstance(json_response["probabilities"], dict)
    # Add more specific assertions based on expected classes if needed
    # assert json_response["predicted_class"] in ["glioma", "meningioma", "notumor", "pituitary"]

def test_predict_endpoint_no_image(api_url):
    """Tests the /predict endpoint with missing image data."""
    headers = {'Content-Type': 'application/json'}
    payload = {'wrong_key': 'some_data'}
    response = requests.post(f"{api_url}/predict", json=payload, headers=headers)
    assert response.status_code == 400 # Bad Request
    assert "error" in response.json()

def test_predict_endpoint_invalid_base64(api_url):
    """Tests the /predict endpoint with invalid base64."""
    headers = {'Content-Type': 'application/json'}
    payload = {'image': 'this is not valid base64'}
    response = requests.post(f"{api_url}/predict", json=payload, headers=headers)
    assert response.status_code == 400 # Bad Request
    assert "error" in response.json()

# Add more tests: test the HTML endpoint, test edge cases, etc.
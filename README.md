# Akurasi-naik-signifikan

# ======================================================================================
#
#  Tabular Regression with Amazon SageMaker LightGBM
#  - Enhanced with Latency Calculation, Lambda Function, and API Gateway Integration
#  - ARCHITECTURE UPDATED FOR STATEFUL FEATURE ENGINEERING
#
# ======================================================================================

# ======================================================================================
# 1. SET UP & 2. MODEL TRAINING
# ======================================================================================
# This part of the notebook remains the same. We assume you have successfully
# trained your LightGBM model using your PySpark feature engineering script
# and have a deployed SageMaker endpoint.
# The key is that the data used for training was the output of your complex script.
# Therefore, the data sent for inference must have the exact same structure.

import sagemaker, boto3, json, time
from sagemaker import get_execution_role
from sagemaker.utils import name_from_base

aws_role = get_execution_role()
aws_region = boto3.Session().region_name
sess = sagemaker.Session()

# Assume 'endpoint_name' is the name of your deployed SageMaker endpoint
# endpoint_name = "jumpstart-example-lightgbm-regression-model-prod"
# print(f"Using SageMaker Endpoint Name: {endpoint_name}")

# ======================================================================================
# NEW ARCHITECTURE OVERVIEW
# ======================================================================================
#
# Your PySpark script for lags and rolling windows cannot run inside a simple Lambda.
# The correct architecture is:
#
# 1. BATCH FEATURE ENGINEERING (AWS Glue / EMR)
#    - Run your PySpark script daily to generate features for all entities.
#    - OUTPUT: A table of features for each `uuid` for each day.
#
# 2. FEATURE STORE (Amazon DynamoDB / SageMaker Feature Store)
#    - Store the output from the batch job in a database.
#    - Example DynamoDB Structure:
#      - Table Name: `user-features`
#      - Primary Key (Partition Key): `uuid` (String)
#      - Primary Key (Sort Key): `feature_date` (String, e.g., "2025-07-18")
#      - Attribute: `feature_vector` (String, a CSV of all feature values)
#
# 3. API GATEWAY + LAMBDA FOR INFERENCE (Code below)
#    - The API receives a simple request (e.g., for a specific user).
#    - The Lambda queries the Feature Store to get the pre-computed features.
#    - The Lambda sends these features to the SageMaker endpoint.
#
# ======================================================================================
# 6. REVISED LAMBDA FUNCTION FOR INFERENCE (WITH FEATURE STORE)
# ======================================================================================
# This is the updated Python code for your Lambda function. It now fetches
# pre-calculated features from DynamoDB.

# --- lambda_function.py ---

import os
import boto3
import json
from datetime import datetime

# Grab environment variables
ENDPOINT_NAME = os.environ['ENDPOINT_NAME']
FEATURE_TABLE_NAME = os.environ['FEATURE_TABLE_NAME'] # e.g., 'user-features'

# Initialize AWS clients
sagemaker_runtime = boto3.client('runtime.sagemaker')
dynamodb = boto3.resource('dynamodb')
feature_table = dynamodb.Table(FEATURE_TABLE_NAME)

def lambda_handler(event, context):
    """
    Lambda handler that fetches pre-computed features from DynamoDB
    and invokes the SageMaker endpoint.
    """
    print("Received event: " + json.dumps(event, indent=2))

    try:
        # API Gateway will pass query string parameters
        # Example: /predict?uuid=user-123&date=2025-07-18
        params = event.get('queryStringParameters', {})
        user_uuid = params.get('uuid')
        
        # Use today's date if not provided
        prediction_date = params.get('date', datetime.utcnow().strftime('%Y-%m-%d'))

        if not user_uuid:
            raise ValueError("Query parameter 'uuid' is required.")

        print(f"Fetching features for uuid: {user_uuid} on date: {prediction_date}")

        # --- 1. Fetch features from DynamoDB (Feature Store) ---
        response = feature_table.get_item(
            Key={
                'uuid': user_uuid,
                'feature_date': prediction_date
            }
        )
        
        item = response.get('Item')
        if not item:
            raise ValueError(f"No features found for uuid {user_uuid} on date {prediction_date}")

        # The feature vector is stored as a single CSV string
        feature_payload = item.get('feature_vector')
        if not feature_payload:
             raise ValueError("Feature vector is missing in the database item.")

        print(f"Successfully fetched feature vector: {feature_payload[:100]}...") # Log first 100 chars

        # --- 2. Invoke SageMaker Endpoint ---
        sagemaker_response = sagemaker_runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType='text/csv',
            Body=feature_payload
        )

        print("Received response from SageMaker endpoint.")
        result = json.loads(sagemaker_response['Body'].read().decode())
        print(f"Prediction result: {result}")

        # --- 3. Return a successful response ---
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
            },
            'body': json.dumps(result)
        }

    except Exception as e:
        print(f"ERROR: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }


# --- Steps to Deploy This Architecture ---
#
# 1.  **Create DynamoDB Table:**
#     -   Go to the DynamoDB console.
#     -   Create a table (e.g., `user-features`).
#     -   Set Partition Key: `uuid` (String).
#     -   Set Sort Key: `feature_date` (String).
#
# 2.  **Set up AWS Glue Job:**
#     -   Go to the AWS Glue console.
#     -   Create a new Spark job.
#     -   Use your PySpark script as the source.
#     -   Modify the end of your script to write the final DataFrame (`df_model_input`)
#       to the DynamoDB table you just created.
#     -   Schedule this job to run daily.
#
# 3.  **Create the Lambda Function:**
#     -   Go to the AWS Lambda Console and create a function (`lightgbm-feature-store-invoke`).
#     -   Use the Python code above.
#     -   **Permissions:** The Lambda's execution role needs permissions for:
#         -   `sagemaker:InvokeEndpoint` (for your specific endpoint).
#         -   `dynamodb:GetItem` (for your features table).
#         -   Basic Lambda execution permissions (for CloudWatch Logs).
#     -   **Environment variables:**
#         -   `ENDPOINT_NAME`: Your SageMaker endpoint name.
#         -   `FEATURE_TABLE_NAME`: The name of your DynamoDB table.
#
# 4.  **Create API Gateway:**
#     -   Follow the steps from the previous response to create a REST API.
#     -   Create a resource (e.g., `/predict`).
#     -   Create a `GET` method on that resource.
#     -   Integrate it with your new Lambda function (using Lambda Proxy integration).
#     -   Enable CORS and deploy the API.
#
# ======================================================================================
# 8. USE THE REVISED API GATEWAY URL IN POSTMAN
# ======================================================================================
#
# 1.  **Method:** `GET`
# 2.  **URL:** Paste the Invoke URL from API Gateway and add query parameters.
#     `https://<api_id>.execute-api.<region>.amazonaws.com/prod/predict?uuid=some-user-id&date=2025-07-18`
# 3.  **Authorization/Body:** None needed for a GET request.
# 4.  **Send Request:** Click "Send". The API will now trigger the Lambda, which
#     fetches the pre-computed features from DynamoDB and gets a prediction.
#

























# ======================================================================================
#
#  Tabular Regression with Amazon SageMaker LightGBM
#  - Enhanced with Latency Calculation, Lambda Function, and API Gateway Integration
#  - ARCHITECTURE UPDATED FOR STATEFUL FEATURE ENGINEERING
#
# ======================================================================================

# ======================================================================================
# 1. SET UP & 2. MODEL TRAINING
# ======================================================================================
# This part of the notebook remains the same. We assume you have successfully
# trained your LightGBM model using your PySpark feature engineering script
# and have a deployed SageMaker endpoint.
# The key is that the data used for training was the output of your complex script.
# Therefore, the data sent for inference must have the exact same structure.

import sagemaker, boto3, json, time
from sagemaker import get_execution_role
from sagemaker.utils import name_from_base

aws_role = get_execution_role()
aws_region = boto3.Session().region_name
sess = sagemaker.Session()

# Assume 'endpoint_name' is the name of your deployed SageMaker endpoint
# endpoint_name = "jumpstart-example-lightgbm-regression-model-prod"
# print(f"Using SageMaker Endpoint Name: {endpoint_name}")

# ======================================================================================
# NEW ARCHITECTURE OVERVIEW
# ======================================================================================
#
# Your PySpark script for lags and rolling windows cannot run inside a simple Lambda.
# The correct architecture is:
#
# 1. BATCH FEATURE ENGINEERING (AWS Glue / EMR)
#    - Run your PySpark script daily to generate features for all entities.
#    - OUTPUT: A table of features for each `uuid` for each day.
#
# 2. FEATURE STORE (Amazon DynamoDB / SageMaker Feature Store)
#    - Store the output from the batch job in a database.
#    - Example DynamoDB Structure:
#      - Table Name: `user-features`
#      - Primary Key (Partition Key): `uuid` (String)
#      - Primary Key (Sort Key): `feature_date` (String, e.g., "2025-07-18")
#      - Attribute: `feature_vector` (String, a CSV of all feature values)
#
# 3. API GATEWAY + LAMBDA FOR INFERENCE (Code below)
#    - The API receives a simple request (e.g., for a specific user).
#    - The Lambda queries the Feature Store to get the pre-computed features.
#    - The Lambda sends these features to the SageMaker endpoint.
#
# ======================================================================================
# 6. REVISED LAMBDA FUNCTION FOR INFERENCE (WITH FEATURE STORE)
# ======================================================================================
# This is the updated Python code for your Lambda function. It now fetches
# pre-calculated features from DynamoDB.

# --- lambda_function.py ---

import os
import boto3
import json
from datetime import datetime

# Grab environment variables
ENDPOINT_NAME = os.environ['ENDPOINT_NAME']
FEATURE_TABLE_NAME = os.environ['FEATURE_TABLE_NAME'] # e.g., 'user-features'

# Initialize AWS clients
sagemaker_runtime = boto3.client('runtime.sagemaker')
dynamodb = boto3.resource('dynamodb')
feature_table = dynamodb.Table(FEATURE_TABLE_NAME)

def lambda_handler(event, context):
    """
    Lambda handler that fetches pre-computed features from DynamoDB
    and invokes the SageMaker endpoint.
    """
    print("Received event: " + json.dumps(event, indent=2))

    try:
        # API Gateway will pass query string parameters
        # Example: /predict?uuid=user-123&date=2025-07-18
        params = event.get('queryStringParameters', {})
        user_uuid = params.get('uuid')
        
        # Use today's date if not provided
        prediction_date = params.get('date', datetime.utcnow().strftime('%Y-%m-%d'))

        if not user_uuid:
            raise ValueError("Query parameter 'uuid' is required.")

        print(f"Fetching features for uuid: {user_uuid} on date: {prediction_date}")

        # --- 1. Fetch features from DynamoDB (Feature Store) ---
        response = feature_table.get_item(
            Key={
                'uuid': user_uuid,
                'feature_date': prediction_date
            }
        )
        
        item = response.get('Item')
        if not item:
            raise ValueError(f"No features found for uuid {user_uuid} on date {prediction_date}")

        # The feature vector is stored as a single CSV string
        feature_payload = item.get('feature_vector')
        if not feature_payload:
             raise ValueError("Feature vector is missing in the database item.")

        print(f"Successfully fetched feature vector: {feature_payload[:100]}...") # Log first 100 chars

        # --- 2. Invoke SageMaker Endpoint ---
        sagemaker_response = sagemaker_runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType='text/csv',
            Body=feature_payload
        )

        print("Received response from SageMaker endpoint.")
        result = json.loads(sagemaker_response['Body'].read().decode())
        print(f"Prediction result: {result}")

        # --- 3. Return a successful response ---
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
            },
            'body': json.dumps(result)
        }

    except Exception as e:
        print(f"ERROR: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }


# --- Steps to Deploy This Architecture ---
#
# 1.  **Create DynamoDB Table:**
#     -   Go to the DynamoDB console.
#     -   Create a table (e.g., `user-features`).
#     -   Set Partition Key: `uuid` (String).
#     -   Set Sort Key: `feature_date` (String).
#
# 2.  **Set up AWS Glue Job:**
#     -   Go to the AWS Glue console.
#     -   Create a new Spark job.
#     -   Use your PySpark script as the source.
#     -   Modify the end of your script to write the final DataFrame (`df_model_input`)
#       to the DynamoDB table you just created.
#     -   Schedule this job to run daily.
#
# 3.  **Create the Lambda Function:**
#     -   Go to the AWS Lambda Console and create a function (`lightgbm-feature-store-invoke`).
#     -   Use the Python code above.
#     -   **Permissions:** The Lambda's execution role needs permissions for:
#         -   `sagemaker:InvokeEndpoint` (for your specific endpoint).
#         -   `dynamodb:GetItem` (for your features table).
#         -   Basic Lambda execution permissions (for CloudWatch Logs).
#     -   **Environment variables:**
#         -   `ENDPOINT_NAME`: Your SageMaker endpoint name.
#         -   `FEATURE_TABLE_NAME`: The name of your DynamoDB table.
#
# 4.  **Create API Gateway:**
#     -   Follow the steps from the previous response to create a REST API.
#     -   Create a resource (e.g., `/predict`).
#     -   Create a `GET` method on that resource.
#     -   Integrate it with your new Lambda function (using Lambda Proxy integration).
#     -   Enable CORS and deploy the API.
#
# ======================================================================================
# 8. USE THE REVISED API GATEWAY URL IN POSTMAN
# ======================================================================================
#
# 1.  **Method:** `GET`
# 2.  **URL:** Paste the Invoke URL from API Gateway and add query parameters.
#     `https://<api_id>.execute-api.<region>.amazonaws.com/prod/predict?uuid=some-user-id&date=2025-07-18`
# 3.  **Authorization/Body:** None needed for a GET request.
# 4.  **Send Request:** Click "Send". The API will now trigger the Lambda, which
#     fetches the pre-computed features from DynamoDB and gets a prediction.
#
























To measure the latency of a prediction using the created SageMaker endpoint, you can time the invocation of the endpoint from the client side. This captures the round-trip time for sending the request, processing it on the endpoint, and receiving the response, which is a standard way to quantify prediction latency in this context.

Based on the notebook code in the document (specifically in the inference section on pages 8-9), you can modify the `query_endpoint` function or add timing around an individual invocation. Here's how to do it step by step:

1. **Ensure the necessary imports are present**: In the notebook cell where you handle inference (page 8), make sure you have `import time` added if it's not already there. The document already imports `boto3`, `numpy as np`, and `pandas as pd`, so you'll use those as well.

2. **Prepare a single example for prediction**: Latency is typically measured for individual predictions (or small batches). Use one row from your test data as an example. From the document (page 8), the test data is loaded as a Pandas DataFrame (`test_data`), with features starting from column 1. Encode a single row as CSV:

   ```python
   # Assuming test_data is already loaded as per the notebook
   single_example = test_data.iloc[0, 1:]  # Features only (exclude target)
   encoded_single_example = single_example.to_csv(header=False, index=False).encode('utf-8')
   ```

3. **Time the endpoint invocation**: Use the existing `query_endpoint` function (or inline the code) and wrap it with `time.time()` to measure elapsed time in seconds. Here's the updated code:

   ```python
   import time
   import boto3

   content_type = "text/csv"
   endpoint_name = "jumpstart-example-lightgbm-classification-m"  # Replace with your actual endpoint_name from the notebook (e.g., from page 7)

   client = boto3.client('runtime.sagemaker')

   start_time = time.time()  # Start timer

   response = client.invoke_endpoint(
       EndpointName=endpoint_name,
       ContentType=content_type,
       Body=encoded_single_example
   )

   end_time = time.time()  # End timer

   latency = end_time - start_time  # Latency in seconds
   print(f"Prediction latency: {latency * 1000:.2f} milliseconds")  # Convert to ms for readability
   ```

4. **Parse the response if needed**: After measuring latency, you can still parse the response as in the document (page 9) to get the predictions:

   ```python
   import json

   model_predictions = json.loads(response['Body'].read())
   predicted_probabilities = model_predictions['probabilities']
   predicted_label = np.argmax(predicted_probabilities)
   print(f"Predicted label: {predicted_label}")
   ```

5. **Run multiple times for average latency**: Latency can vary due to network conditions or endpoint load. To get a reliable measure, repeat the invocation in a loop (e.g., 10-100 times) and compute the average:

   ```python
   num_runs = 10
   latencies = []

   for _ in range(num_runs):
       start_time = time.time()
       response = client.invoke_endpoint(
           EndpointName=endpoint_name,
           ContentType=content_type,
           Body=encoded_single_example
       )
       end_time = time.time()
       latencies.append(end_time - start_time)

   avg_latency = sum(latencies) / len(latencies)
   print(f"Average prediction latency over {num_runs} runs: {avg_latency * 1000:.2f} milliseconds")
   ```

### Key Notes:
- This measures **client-side latency**, including network transit time. If you need pure server-side inference time (excluding network), check Amazon CloudWatch metrics for the endpoint (look for `InvocationLatency` under SageMaker metrics), but the document doesn't cover that—it's a separate AWS console or SDK feature.
- Ensure the endpoint is still running (from the deployment in section 3 of the notebook). If deleted (as in the cleanup on page 10), redeploy it first.
- Payload size affects latency; keep it small (e.g., single example) for accurate per-prediction measurement. The document batches due to limits (page 9), but for latency, avoid large batches.
- This approach is fully compatible with the LightGBM model (or CatBoost if switched) and the MNIST example in the document.
- Test in your notebook environment to confirm—no changes to training or deployment are needed.

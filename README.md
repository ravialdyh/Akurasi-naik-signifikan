# Akurasi-naik-signifikan

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

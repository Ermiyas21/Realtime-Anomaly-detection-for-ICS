### Real-Time Anomaly Detection with Uncertainty Estimation 


##### Instructions
This repository contains a setup for running an anomaly detection system for Industrial Control Systems (ICS) using Docker, Kafka, Spark, Grafana, and InfluxDB. 
Follow the steps below to get the system up and running. 
##### Prerequisites
- Docker
- Docker Compose
- Python 3.x 
##### Getting Started 
Step 1: Bring up Docker Containers
First, start the Docker containers using Docker Compose: 
   - docker-compose up
Step 2: Verify Dataset Availability 
Open a new terminal and check if the dataset is available.

Step 3: Run the Kafka Producer
Navigate to the Kafka directory and run the Kafka producer script:  
   - cd kafka
   - python producer.py
Step 4: Check Running Docker Containers
   - docker ps
Step 5: Access the Spark Container
Access the Spark container:
   - docker exec -it <SparkContainerID> bash
 Replace <SparkContainerID> with the actual ID of the Spark container from the docker ps output.
Step 6: Submit the Spark Job
Once inside the Spark container, run the following command to submit the Spark job:
  - spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.0 /sparkScripts/streamingProcess.py

Step 7: Start InfluxDB and Grafana
- Ensure that InfluxDB (http://localhost:8086) and Grafana (http://localhost:3000) are running. They should be started with the Docker Compose command in Step 1. 
- Setup the user name ans passwrd for both InfluxDb and Grafana 

Step 8: Visualize Output with Grafana
Open Grafana in your web browser (at http://localhost:3000). Configure Grafana to connect to InfluxDB, and create dashboards to visualize the output data from the anomaly detection process. 
##### Conclusion
Follow these steps to set up and run the anomaly detection system for ICS. Ensure all prerequisites are installed and configured correctly. For any issues or questions, please open an issue in this repository. 

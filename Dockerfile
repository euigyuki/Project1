FROM python:3.9

WORKDIR /app

# Copy only the requirements file first (for efficient caching)
COPY requirements.txt .

# Install required dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir debugpy


# Copy only the `data/results/` directory into the container
COPY data/results /app/data/results



# Default command
CMD ["python", "/app/data/results/get_fleiss_kappas.py"]

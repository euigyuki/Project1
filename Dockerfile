FROM python:3.9

WORKDIR /app

# Copy only the requirements file first (for efficient caching)
COPY requirements.txt .

# Install required dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir debugpy


COPY data/results /app/data/results
COPY data/picked_captions /app/data/picked_captions/



# Default command
#CMD ["python", "/app/data/results/get_fleiss_kappas.py"]
CMD ["tail", "-f", "/dev/null"]

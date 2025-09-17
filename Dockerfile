
FROM python:3.9-slim

WORKDIR /app

# 1. Copy the requirements file 
COPY requirements.txt .

# 2. Install the Python libraries. This step will be cached
# as long as requirements.txt doesn't change.
RUN pip install --no-cache-dir -r requirements.txt

# 3. Now, copy the rest of your application files
COPY . .

EXPOSE 8080

# The command to run  app using gunicorn
CMD ["gunicorn", "--workers", "1", "--bind", "0.0.0.0:8080", "app:app"]
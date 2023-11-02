docker build -t ekinox-test -f .
docker run  -p 8501:8501 --rm ekinox-test

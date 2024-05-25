from pic_service import app

# nohup gunicorn --bind 0.0.0.0:5000  --limit-request-line 0 --workers 4 --log-file ./logs/gunicorn.log wsgi:app > ./logs/wsgl.out &
if __name__ == "__main__":
    app.run()

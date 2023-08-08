FROM python:3
#LABEL authors="andrey"
RUN pip --no-cache-dir install catboost pandas scikit-learn explainerdashboard waitress

COPY app.py /app/app.py
COPY trained_model.cbm /app/trained_model.cbm
COPY online_shoppers_new.csv /app/online_shoppers_new.csv
WORKDIR /app

#RUN python app.py
EXPOSE 9050
#ENTRYPOINT ["python"]
CMD ["python", "./app.py"]
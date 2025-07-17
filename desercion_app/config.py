from datetime import timedelta

class Config:
    SECRET_KEY = 'clave_segura_123'
    PERMANENT_SESSION_LIFETIME = timedelta(days=1)

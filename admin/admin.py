# -*- coding: utf-8 -*-

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from starlette_admin.contrib.sqla import Admin, ModelView

from admin.model import APIKey
from app.main import app

Base = declarative_base()
engine = create_engine("sqlite:///test.db", connect_args={"check_same_thread": False})


# Create admin
admin = Admin(engine, title="Example: SQLAlchemy")

# Add view
admin.add_view(ModelView(APIKey))

# Mount admin to your app
admin.mount_to(app)

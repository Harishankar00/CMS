from app import app
from database.models import db

# Initialize the app context and create all tables
with app.app_context():
    db.create_all()

print("Database setup completed.")

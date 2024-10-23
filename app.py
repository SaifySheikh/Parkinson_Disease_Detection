from flask import Flask
import os
from app.routes.main_routes import main_routes  # Use absolute import

def create_app():
    app = Flask(__name__, template_folder=os.path.join(os.getcwd(), 'app', 'templates'))

    # Define the upload folder
    app.config['UPLOAD_FOLDER'] = r'C:\Users\LENOVO\Downloads\Mini Project 7th Sem\parkinsons_detection\app\upload'
    app.config['ALLOWED_EXTENSIONS'] = {'mp3', 'jpg', 'jpeg','png'}

    # Create the upload directory if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    app.register_blueprint(main_routes)

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)

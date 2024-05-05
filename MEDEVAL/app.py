import os
import json
import zipfile
import threading
from functools import wraps
from datetime import datetime
from datetime import timedelta
from running_model import evaluate_and_mail
from authlib.integrations.flask_client import OAuth
from flask import Flask, render_template, url_for, request, session, abort, redirect, flash

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/files'
app.config['SESSION_COOKIE_NAME'] = 'google-login-session'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=5)

# Semaphore to limit the number of concurrent threads
max_threads = 3
thread_semaphore = threading.Semaphore(max_threads)

# Check for login
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user = dict(session).get('profile', None)
        if user:
            return f(*args, **kwargs)
        return f"""You aint logged in, no page for u! <button onclick="location.href = '{url_for('welcome')}'">Go Back</button>"""
    return decorated_function

# Function to get user info inside html files
@app.context_processor
def utility_processor():
    def login_logout():
        return dict(session).get('profile', None)
    return dict(profile=login_logout)

# oAuth Setup
f = open('static/client_secret.json')
data = json.load(f)['web']
oauth = OAuth(app)
google = oauth.register(
    name='google',
    client_id=data['client_id'],
    client_secret=data['client_secret'],
    access_token_url='https://accounts.google.com/o/oauth2/token',
    access_token_params=None,
    authorize_url='https://accounts.google.com/o/oauth2/auth',
    authorize_params=None,
    api_base_url='https://www.googleapis.com/oauth2/v1/',
    userinfo_endpoint='https://openidconnect.googleapis.com/v1/userinfo',  # This is only needed if using openId to fetch user info
    client_kwargs={'scope': 'email profile'},
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration'
)

# Initial welcome page
@app.route('/')
def welcome():
    return render_template('login.html')

# login route
@app.route('/login')
def login():
    google = oauth.create_client('google')  # create the google oauth client
    redirect_uri = url_for('authorize', _external=True)
    return google.authorize_redirect(redirect_uri)

# authorise route
@app.route('/authorize')
def authorize():
    google = oauth.create_client('google')  # create the google oauth client
    token = google.authorize_access_token()  # Access token from google (needed to get user info)
    resp = google.get('userinfo')  # userinfo contains stuff u specificed in the scrope
    user_info = resp.json()
    user = oauth.google.userinfo()  # uses openid endpoint to fetch user info
    # Here you use the profile/user data that you got and query your database find/register the user
    # and set ur own data in the session not the profile from google
    session['profile'] = user_info
    session.permanent = True  # make the session permanant so it keeps existing after broweser gets closed
    return redirect('/home')

# logout route
@app.route('/logout')
def logout():
    for key in list(session.keys()):
        session.pop(key)
    return redirect('/')

# home route
@app.route('/home', methods=['GET', 'POST'])
@login_required 
def home():
    if request.method == 'POST':
        # Get selected file from local directory
        filesDict = request.files.to_dict()
        uploadData=request.files['media']
        data_file_name = uploadData.filename
        folder_path = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'])
        file_path = os.path.join(folder_path, data_file_name)
        uploadData.save(file_path)

        # Extract the uploaded zip file
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(os.path.join(folder_path, 'model'))

        # Get the name of file after extracting
        extracted_files = zip_ref.namelist() 
        if len(extracted_files) > 1:
            extracted_file = extracted_files[0].split('/')[0]
        else:
            extracted_file = extracted_files[0]

        # Run the evaluation on different thread so that it would run independently as evaluation takes time
        if thread_semaphore.acquire(blocking=False):
            thread = threading.Thread(target=evaluate_and_mail, args=(os.path.join(folder_path, 'model', extracted_file), request.form.get('choose_attacks'), dict(session).get('profile')))
            thread.start()
            # Flash message
            flash('Thank you for submitting your model. You will get email of evaluation of model.', 'success')
        else:
            flash('Maximum number of concurrent threads reached. Try again later.', 'danger')
        
    return render_template('home.html', date=datetime.now().strftime("%A, %B %d"))

# about route
@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
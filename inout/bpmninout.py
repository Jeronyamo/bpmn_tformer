from flask import Flask, request, render_template 


app = Flask(__name__) 

UPLOAD_FOLDER = 'server/uploads/'

@app.route("/", methods=["GET", "POST"]) 
def index(): 
    if request.method == "POST": 
        file = request.files.get("file") 
        file_content = file.read()
        print(file_content)

        # check if file loaded successfully or not 
        if file_content: 
            return "Upload Successful"
        else: 
            return "Upload Unsuccessful"

    return render_template("index.html") 

if __name__ == "__main__":
    # from waitress import serve
    # serve(app, host="127.0.0.1", port=5000)
    app.run(debug= True)
from flask import Flask, request, jsonify,send_from_directory
from flask_cors import CORS
from tweets_search import tweets_search
from text_classification import text_classification, generate_js_dictionary
import pandas as pd
import utils_conf

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return 'Text Classification Backend!'

@app.route('/api/tweetssearch', methods=['POST'])
def tweetssearch():
    if request.is_json:
        # Get the JSON data
        data = request.get_json()    
        query = data.get('query', None)
        print(query)
        df = tweets_search(query, max_results=10)    
        json_data = df.to_json(orient='records')
        
        return jsonify(json_data=json_data)            
        
    
    return None

@app.route('/api/getclassifications', methods=['GET'])
def getclassifications():
    
    # # Get the JSON data       
    df = text_classification()    
    json_data = df.to_json(orient='records')
    
    #df = pd.read_excel("outputs/text_classification_output.xlsx", sheet_name=1)
    json_data = df.to_json(orient='records')
    
    return jsonify(json_data=json_data)            

@app.route('/api/getchartdata', methods=['GET'])
def getchartdata():
    
    #df = pd.read_excel("outputs/text_classification_output.xlsx", sheet_name=1)    
    #json_data = df.to_json(orient='records')
    
    json_data = generate_js_dictionary()
    return jsonify(json_data=json_data)            

@app.route('/api/getkeys', methods=['GET'])
def getkeys():
    try:
        keys = []
        keys.append(utils_conf.get_api_key('bearer_token'))
        keys.append(utils_conf.get_api_key('OPENAI_KEY'))
    except:
        return jsonify("Erro ao carregar as APIs")            
        
    return jsonify(keys)          

@app.route('/api/downloadsearch')
def downloadsearch(filename='tweets_search_output.xlsx'):
    return send_from_directory('outputs', filename, as_attachment=True)
  
    
@app.route('/api/test1', methods=['GET'])
def test1():
    df = pd.read_excel('outputs/text_classification_output.xlsx')
    df = pd.DataFrame(df['Texto'])    
    json_data = df.to_json(orient='records')
    
    return jsonify(json_data)            

if __name__ == '__main__':
    app.run(debug=True, port=8080)

# DocSimi - Document Similarity

DocSimi is a comprehensive web application built using FastAPI backend that provides various techniques to compute and visualize the similarity between two text documents.

## Techniques Implemented

The following similarity techniques are implemented and made available through an easy-to-use interface:

### Typeface Similarity

- Bag of Words
- TF-IDF (Term Frequency-Inverse Document Frequency)
- Graph Edit Distance based on Levenshtein
- Jaccard

### Semantic Similarity

- Latent Semantic Analysis (LSA)
- Word2Vec

## Running Locally

To run the project locally, make sure you have Python 3.6 or higher installed. You can then clone the repository and install the required dependencies.

```bash
git clone https://github.com/YOUR_USERNAME/DocSimi.git
cd DocSimi
pip install -r requirements.txt
```

To run the application, use the command:
```bash
python app.py
```
The application will start running on your local server, which you can access through http://localhost:3000 on your web browser.

## Contribution

Contributions to improve DocSimi are always welcome. Please feel free to create issues or submit pull requests.

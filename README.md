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

Then, you just run the app.py file, The application will be running at http://localhost:3000.

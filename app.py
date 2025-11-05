from flask import Flask, render_template, request, Response
import pandas as pd
from hybrid_search import HybridSearch, load_synonyms, ensure_column, load_csv
import os

app = Flask(__name__)

# --- Global Search Instance ---
hs = None

def initialize_search_engine():
    """Loads data and initializes the HybridSearch engine."""
    global hs

    # Define file paths and column names
    csv_path = "products.csv"  # Assume the product data is in this file
    synonyms_file = "synonyms.csv"
    code_col_name = "Mã hàng"
    name_col_name = "Tên hàng"

    if not os.path.exists(csv_path):
        print(f"ERROR: Product data file not found at '{csv_path}'.")
        print("Please create this file with your product data.")
        hs = None
        return

    print("Initializing search engine...")
    try:
        load_synonyms(synonyms_file)
        headers, rows = load_csv(csv_path)
        code_col = ensure_column(headers, code_col_name)
        name_col = ensure_column(headers, name_col_name)
        df = pd.DataFrame(rows)
        hs = HybridSearch(df, name_col=name_col, code_col=code_col)
        print("Search engine initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize search engine: {e}")
        hs = None

@app.route('/')
def index():
    """Renders the main search page."""
    return render_template('index.html', results={}, queries=[])

@app.route('/search', methods=['POST'])
def search():
    """Handles the search request from the UI."""
    if hs is None:
        return "<h1>Error: Search engine is not initialized. Check server logs for details.</h1>", 500

    queries_str = request.form.get('queries', '')
    queries = [q.strip() for q in queries_str.splitlines() if q.strip()]

    search_results = {}
    if queries:
        for query in queries:
            results_df = hs.search(query, topk=5)
            search_results[query] = results_df.to_dict('records')

    return render_template('index.html', results=search_results, queries=queries)

@app.route('/export', methods=['POST'])
def export():
    """Handles the export of selected items."""
    if hs is None:
        return "<h1>Error: Search engine is not initialized.</h1>", 500

    # Re-run the search to ensure we have the latest results in memory
    queries_str = request.form.get('queries', '')
    queries = [q.strip() for q in queries_str.splitlines() if q.strip()]

    selected_items = []
    for i, query in enumerate(queries, 1):
        selection_key = f"selection_{i}"
        quantity_key = f"quantity_{i}"

        selected_idx = request.form.get(selection_key)
        quantity = request.form.get(quantity_key, 0)

        try:
            quantity = int(quantity)
        except (ValueError, TypeError):
            quantity = 0

        if selected_idx is not None and quantity > 0:
            try:
                selected_idx = int(selected_idx)
                # Retrieve the full item details from the DataFrame
                item_details = hs.df.iloc[selected_idx].to_dict()
                item_details['Số lượng'] = quantity
                selected_items.append(item_details)
            except (ValueError, IndexError):
                pass  # Ignore invalid index

    if not selected_items:
        return "<h1>No items selected for export.</h1>", 400

    # Create a tab-separated string for the file content
    output_lines = ["Mã hàng\tTên hàng\tSố lượng\n"]
    for item in selected_items:
        code = item.get(hs.code_col, '')
        name = item.get(hs.name_col, '')
        qty = item.get('Số lượng', 0)
        output_lines.append(f"{code}\t{name}\t{qty}\n")

    output_content = "".join(output_lines)

    return Response(
        output_content,
        mimetype="text/plain",
        headers={"Content-Disposition": "attachment;filename=order.txt"}
    )

if __name__ == '__main__':
    initialize_search_engine()
    app.run(debug=True, port=5002)

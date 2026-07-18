# AMR Graph Visualizer — How to Run

## Backend (Python / FastAPI)

```bash
cd graph-visualizer/backend

# Install dependencies (in your venv / conda env that has torch + torch_geometric)
pip install -r requirements.txt

# Start the API server
uvicorn main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`.
Uploaded `.pt` files are stored in `backend/uploads/`.

---

## Frontend (Next.js)

```bash
cd graph-visualizer/frontend

npm install   # first time only
npm run dev
```

Open `http://localhost:3000` in your browser.

---

## Usage

1. Click **Upload .pt file** in the sidebar and select one or more `.pt` graph files.
2. **Check** a file to display it — its AMR graph appears in the main area.
3. Check a **second** file to compare both graphs side-by-side.
4. Toggle **2D / 3D** per panel using the buttons in the panel header.
5. Switch between **Force** layout (physics simulation) and **Semantic** layout (PCA of FinBERT node features).
6. Hover over a node to see its label.
7. Click the **pencil** icon to rename a file inline (saves to disk).
8. Click the **trash** icon to delete a file.

---

## Environment

| Variable | Default | Description |
|---|---|---|
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Backend base URL |

Set in `frontend/.env.local` if your backend runs on a different port/host.

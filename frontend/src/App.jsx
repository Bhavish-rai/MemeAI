import { useState } from "react";
import axios from "axios";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

const examples = [
  "exam tomorrow and I haven't studied",
  "my friend borrowed money and didn't return it",
  "everyone is studying except me",
  "I finished my project successfully",
];

function App() {
  const [query, setQuery] = useState("");
  const [memes, setMemes] = useState([]);
  const [selectedMeme, setSelectedMeme] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [mode, setMode] = useState("text");
  const [selectedFile, setSelectedFile] = useState(null);

  const searchMemes = async () => {
    if (!query.trim()) return;

    try {
      setLoading(true);
      setError("");
      setMemes([]);

      const response = await axios.get(
        `${API_URL}/api/search`,
        {
          params: {
            query: query.trim(),
            limit: 5,
          },
        }
      );

      setMemes(response.data.results);
    } catch (err) {
      setError("Unable to connect to MemeAI backend.");
    } finally {
      setLoading(false);
    }
  };

  const searchImage = async () => {
    if (!selectedFile) return;

    try {
      setLoading(true);
      setError("");
      setMemes([]);

      const formData = new FormData();

      formData.append("file", selectedFile);

      const response = await axios.post(
        `${API_URL}/api/image-search`,
        formData
      );

      if (response.data.error) {
        setError(response.data.error);
        return;
      }

      setMemes(response.data.results);
    } catch (err) {
      console.error(err);
      setError("Unable to process the image.");
    } finally {
      setLoading(false);
    }
  };

  const selectExample = (example) => {
    setQuery(example);
    setError("");
  };

  const closePreview = () => {
    setSelectedMeme(null);
  };

  return (
    <div className="app">

      <header className="navbar">
        <div className="logo">MemeAI</div>

        <div className="nav-text">
          AI-Powered Meme Search
        </div>
      </header>


      <main className="main-content">

        <section className="hero">

          <p className="eyebrow">
            MULTIMODAL MEME SEARCH
          </p>

          <h1>
            Find the meme that
            <br />
            matches your thought.
          </h1>

          <p className="description">
            Search using natural language or upload
            a meme to find visually similar memes.
          </p>


          {/* MODE BUTTONS */}

          <div
            style={{
              display: "flex",
              gap: "10px",
              marginBottom: "15px",
            }}
          >

            <button
              onClick={() => {
                setMode("text");
                setMemes([]);
                setError("");
              }}
              style={{
                padding: "10px 18px",
                borderRadius: "8px",
                border: "1px solid #ddd",
                cursor: "pointer",
                fontWeight:
                  mode === "text"
                    ? "600"
                    : "400",
              }}
            >
              Text Search
            </button>

            <button
              onClick={() => {
                setMode("image");
                setMemes([]);
                setError("");
              }}
              style={{
                padding: "10px 18px",
                borderRadius: "8px",
                border: "1px solid #ddd",
                cursor: "pointer",
                fontWeight:
                  mode === "image"
                    ? "600"
                    : "400",
              }}
            >
              Image Search
            </button>

          </div>


          {/* TEXT SEARCH */}

          {mode === "text" && (
            <>
              <div className="search-box">

                <input
                  type="text"
                  placeholder="Example: exam tomorrow and I haven't studied"
                  value={query}
                  onChange={(e) =>
                    setQuery(e.target.value)
                  }
                  onKeyDown={(e) => {
                    if (e.key === "Enter") {
                      searchMemes();
                    }
                  }}
                />

                <button
                  onClick={searchMemes}
                  disabled={
                    loading ||
                    !query.trim()
                  }
                >
                  {loading
                    ? "Searching..."
                    : "Search"}
                </button>

              </div>


              <div className="examples">

                <span>Try:</span>

                {examples.map((example) => (
                  <button
                    key={example}
                    onClick={() =>
                      selectExample(example)
                    }
                  >
                    {example}
                  </button>
                ))}

              </div>
            </>
          )}


          {/* IMAGE SEARCH */}

          {mode === "image" && (
            <div
              style={{
                padding: "25px",
                border: "1px dashed #cbd5e1",
                borderRadius: "12px",
                marginTop: "10px",
              }}
            >

              <input
                type="file"
                accept="image/*"
                onChange={(e) => {
                  setSelectedFile(
                    e.target.files[0]
                  );
                  setError("");
                }}
              />

              {selectedFile && (
                <p>
                  Selected:{" "}
                  <strong>
                    {selectedFile.name}
                  </strong>
                </p>
              )}

              <button
                onClick={searchImage}
                disabled={
                  loading ||
                  !selectedFile
                }
                style={{
                  marginTop: "12px",
                  padding: "11px 20px",
                  border: "none",
                  borderRadius: "8px",
                  cursor: "pointer",
                }}
              >
                {loading
                  ? "Analyzing..."
                  : "Find Similar Memes"}
              </button>

            </div>
          )}

        </section>


        {error && (
          <div className="error">
            {error}
          </div>
        )}


        {loading && (
          <div className="loading-state">
            <div className="spinner"></div>

            <p>
              Finding matching memes...
            </p>
          </div>
        )}


        {!loading && memes.length > 0 && (
          <section className="results-section">

            <div className="results-header">

              <div>

                <p className="results-label">
                  SEARCH RESULTS
                </p>

                <h2>
                  Matching Memes
                </h2>

              </div>

              <span>
                {memes.length} results
              </span>

            </div>


            <div className="meme-grid">

              {memes.map((meme, index) => (

                <article
                  className="meme-card"
                  key={`${meme.image}-${index}`}
                  onClick={() =>
                    setSelectedMeme(meme)
                  }
                >

                  <div className="image-container">

                    <img
                      src={`${API_URL}${meme.image_url}`}
                      alt={
                        meme.caption ||
                        "Meme"
                      }
                      loading="lazy"
                    />

                  </div>


                  <div className="meme-info">

                    <div className="rank-row">

                      <span className="rank">
                        #{index + 1}
                      </span>

                      <span className="match">
                        {(meme.similarity * 100).toFixed(1)}
                        % match
                      </span>

                    </div>


                    <p className="caption">
                      {meme.caption}
                    </p>


                    <div className="meta">

                      <span>
                        {meme.sentiment}
                      </span>

                      <span>
                        {meme.image}
                      </span>

                    </div>

                  </div>

                </article>

              ))}

            </div>

          </section>
        )}


        {!loading &&
          memes.length === 0 &&
          !error && (
            <section className="empty-state">

              <div className="empty-icon">
                ✦
              </div>

              <h3>
                Search for a meme
              </h3>

              <p>
                Search using text or upload
                an image to find similar memes.
              </p>

            </section>
          )}

      </main>


      {selectedMeme && (

        <div
          className="modal-overlay"
          onClick={closePreview}
        >

          <div
            className="modal"
            onClick={(event) =>
              event.stopPropagation()
            }
          >

            <button
              className="close-button"
              onClick={closePreview}
            >
              ×
            </button>


            <div className="modal-image">

              <img
                src={`${API_URL}${selectedMeme.image_url}`}
                alt={
                  selectedMeme.caption ||
                  "Meme"
                }
              />

            </div>


            <div className="modal-info">

              <div className="modal-top">

                <span>
                  Meme Match
                </span>

                <strong>
                  {(selectedMeme.similarity * 100).toFixed(1)}%
                </strong>

              </div>


              <p className="modal-caption">
                {selectedMeme.caption}
              </p>


              <div className="modal-meta">

                <span>
                  Sentiment:{" "}
                  {selectedMeme.sentiment}
                </span>

                <span>
                  {selectedMeme.image}
                </span>

              </div>

            </div>

          </div>

        </div>

      )}

    </div>
  );
}

export default App;
import { useState, useEffect } from "react";
import axios from "axios";

function App() {

  const [pdf, setPdf] = useState(null);

  const [question, setQuestion] = useState("");

  const [uploaded, setUploaded] = useState(false);

  const [filename, setFilename] = useState("");

  const [loading, setLoading] = useState(false);

  const [uploading, setUploading] = useState(false);

  const [response, setResponse] = useState(null);

  const [selectedModel, setSelectedModel] =
    useState("llama70b");

  const API_BASE =
    "http://localhost:8000";

  useEffect(() => {

    loadStatus();

  }, []);

  const loadStatus = async () => {

    try {

      const res = await axios.get(
        `${API_BASE}/status`
      );

      if (res.data.uploaded) {

        setUploaded(true);

        setFilename(
          res.data.filename
        );
      }

    } catch (err) {

      console.log(err);
    }
  };

  const models = [
    {
      label: "Llama 3.3 70B",
      value: "llama70b"
    },
    {
      label: "Llama 3.1 8B",
      value: "llama8b"
    },
    {
      label: "Gptoss 20B",
      value: "gptoss20b"
    },
    {
      label: "Gptoss 120B",
      value: "gptoss120b"
    }

  ];

  const uploadPDF = async () => {
    if (!pdf) {
      alert("Select PDF");
      return;
    }

    try {

      setUploading(true);

      const formData =
        new FormData();

      formData.append(
        "file",
        pdf
      );

      const res =
        await axios.post(
          `${API_BASE}/upload`,
          formData,
          {
            headers: {
              "Content-Type":
                "multipart/form-data"
            }
          }
        );

      if (res.data.error) {

        alert(res.data.error);

        return;
      }

      setUploaded(true);

      setFilename(
        res.data.filename
      );

      alert(
        `Uploaded: ${res.data.filename}`
      );

    } catch (err) {

      console.log(err);

      alert("Upload failed");

    } finally {

      setUploading(false);
    }
  };

  const askQuestion = async () => {

    if (!uploaded) {

      alert(
        "Upload PDF first"
      );

      return;
    }

    if (!question.trim()) {

      alert(
        "Enter question"
      );

      return;
    }

    try {

      setLoading(true);

      setResponse(null);

      const res =
        await axios.post(
          `${API_BASE}/query`,
          {
            question,
            model_name:
              selectedModel
          }
        );

      if (res.data.error) {

        alert(res.data.error);

        return;
      }

      setResponse(res.data);

    } catch (err) {

      console.log(err);

      alert("Query failed");

    } finally {

      setLoading(false);
    }
  };

  return (

    <div
      style={{
        minHeight: "100vh",
        background: "#f4f4f4",
        padding: "40px",
        fontFamily: "Arial"
      }}
    >

      <div
        style={{
          maxWidth: "1100px",
          margin: "0 auto",
          background: "white",
          padding: "30px",
          borderRadius: "12px",
          boxShadow:
            "0 0 10px rgba(0,0,0,0.1)"
        }}
      >

        <h1>
          RAG System
        </h1>

        <div
          style={{
            marginTop: "30px"
          }}
        >

          <h2>
            Upload PDF
          </h2>

          <input
            type="file"
            accept=".pdf"
            onChange={(e) =>
              setPdf(
                e.target.files[0]
              )
            }
          />

          <button
            onClick={uploadPDF}
            disabled={uploading}
            style={{
              marginLeft: "15px",
              padding:
                "10px 18px"
            }}
          >
            {uploading
              ? "Uploading..."
              : "Upload"}
          </button>

          {uploaded && (

            <div
              style={{
                marginTop: "15px",
                color: "green"
              }}
            >

              <strong>
                Current PDF:
              </strong>{" "}

              {filename}

            </div>
          )}

        </div>

        <div
          style={{
            marginTop: "30px"
          }}
        >

          <h2>
            Select Model
          </h2>

          <select
            value={selectedModel}
            onChange={(e) =>
              setSelectedModel(
                e.target.value
              )
            }
            style={{
              padding: "10px",
              width: "250px"
            }}
          >

            {models.map(
              (model) => (

                <option
                  key={model.value}
                  value={model.value}
                >
                  {model.label}
                </option>
              )
            )}

          </select>

        </div>

        <div
          style={{
            marginTop: "30px"
          }}
        >

          <h2>
            Ask Question
          </h2>

          <textarea
            rows={4}
            value={question}
            onChange={(e) =>
              setQuestion(
                e.target.value
              )
            }
            placeholder="Ask question..."
            style={{
              width: "100%",
              padding: "12px"
            }}
          />

          <button
            onClick={askQuestion}
            disabled={loading}
            style={{
              marginTop: "15px",
              padding:
                "12px 20px"
            }}
          >
            {loading
              ? "Generating..."
              : "Ask"}
          </button>

        </div>

        {response && (

  <div
    style={{
      marginTop: "40px"
    }}
  >

    <h2>
      Results
    </h2>

    {response.top_k_answers.map(
      (item, index) => (

        <div
          key={index}
          style={{
            border:
              "1px solid #ddd",
            padding: "20px",
            borderRadius:
              "10px",
            marginBottom:
              "25px",
            background:
              "white"
          }}
        >

          <h3>
            Rank:
            {" "}
            {item.answer_rank}
          </h3>

          <p>
            <strong>
              Answer:
            </strong>
          </p>

          <p
            style={{
              lineHeight: "1.7"
            }}
          >
            {item.answer}
          </p>

          <p>
            <strong>
              Citation Pages:
            </strong>
            {" "}
            {item.citation_ranges.join(", ")}
          </p>

          <div
            style={{
              marginTop: "20px"
            }}
          >

            <strong>
              Citations:
            </strong>

            {item.citations.map(
              (citation, idx) => (

                <div
                  key={idx}
                  style={{
                    background:
                      "#f5f5f5",
                    padding: "15px",
                    marginTop: "12px",
                    borderRadius: "8px",
                    border:
                      "1px solid #ddd"
                  }}
                >

                  <p>
                    <strong>
                      Page:
                    </strong>
                    {" "}
                    {citation.page_range}
                  </p>

                  <p
                    style={{
                      marginTop: "8px",
                      lineHeight: "1.6"
                    }}
                  >
                    {
                      citation.matched_text
                    }
                  </p>

                </div>
              )
            )}

          </div>

        </div>
      )
    )}

  </div>
)}
      </div>

    </div>
  );
}

export default App;